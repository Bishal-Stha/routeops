"""Real-time MLOps drift monitor and Prometheus metrics registry for RouteOps.

Tracks query distributions, prediction entropy, confidence degradation,
and inference latency across rolling request windows to detect concept drift.
"""

from __future__ import annotations

import collections
import logging
import time
from typing import Any, Deque, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from src.config import RouteOpsConfig, load_config

logger = logging.getLogger("routeops.drift_monitor")


class QueryDriftRecord(BaseModel):
    """Snapshot record for an individual routed query."""

    timestamp: float = Field(default_factory=time.time)
    query_length: int
    routed_class: str
    confidence: float
    entropy: float
    latency_ms: float
    language: str
    is_guardrail_triggered: bool


class DriftSummary(BaseModel):
    """Aggregate statistics over the current rolling monitoring window."""

    window_size: int
    total_records: int
    class_distribution: Dict[str, float]
    mean_confidence: float
    mean_entropy: float
    mean_latency_ms: float
    guardrail_trigger_rate: float
    is_drift_detected: bool
    drift_reasons: List[str]


class DriftMonitor:
    """MLOps monitoring engine for tracking inference behavior and query drift."""

    def __init__(
        self,
        config: Optional[RouteOpsConfig] = None,
        custom_registry: Optional[Any] = None,
    ) -> None:
        """Initialize the drift monitor.

        Args:
            config: RouteOps configuration instance.
            custom_registry: Optional custom Prometheus CollectorRegistry.
        """
        self.config = config or load_config()
        self.window_size = self.config.monitoring.drift_window_size
        self.entropy_threshold = self.config.monitoring.entropy_threshold
        self.latency_slo_ms = self.config.monitoring.latency_slo_ms

        self.window: Deque[QueryDriftRecord] = collections.deque(
            maxlen=self.window_size
        )

        # Baseline class distribution expectations
        self.baseline_distribution: Dict[str, float] = {
            "direct_response": 0.25,
            "rag_retrieval": 0.35,
            "frontier_reasoning": 0.35,
            "guardrail_blocked": 0.05,
        }

        # Setup Prometheus Metrics
        self._init_prometheus(custom_registry)

    def _init_prometheus(self, custom_registry: Optional[Any]) -> None:
        """Initialize Prometheus counters, histograms, and gauges."""
        self.prometheus_available = PROMETHEUS_AVAILABLE
        if not self.prometheus_available:
            logger.warning(
                "prometheus_client not available. Operating with in-memory metrics only."
            )
            return

        from prometheus_client import REGISTRY

        self.registry = custom_registry or REGISTRY

        # Counters & Histograms
        try:
            self.query_counter = Counter(
                "routeops_queries_total",
                "Total number of routed queries",
                ["routed_class", "language", "action"],
                registry=self.registry,
            )
            self.guardrail_counter = Counter(
                "routeops_guardrail_blocks_total",
                "Total number of queries blocked by safety guardrails",
                registry=self.registry,
            )
            self.cost_savings_counter = Counter(
                "routeops_cost_savings_dollars_total",
                "Estimated cumulative cost saved by routing queries away from Frontier models",
                registry=self.registry,
            )
            self.latency_histogram = Histogram(
                "routeops_routing_latency_ms",
                "Semantic routing latency in milliseconds",
                buckets=(1.0, 3.0, 5.0, 10.0, 15.0, 25.0, 50.0, 100.0, 250.0),
                registry=self.registry,
            )
            self.entropy_gauge = Gauge(
                "routeops_prediction_entropy_bits",
                "Current prediction entropy in bits",
                registry=self.registry,
            )
        except ValueError:
            # Handle cases where metrics were already registered in this process
            pass

    def record_query(
        self,
        routed_class: str,
        confidence: float,
        entropy: float,
        latency_ms: float,
        query_text: str,
        language: str = "en",
        action: str = "route",
        is_guardrail_triggered: bool = False,
    ) -> None:
        """Record telemetry for a single query into the rolling buffer and Prometheus.

        Args:
            routed_class: The selected routing category.
            confidence: Prediction confidence score.
            entropy: Calculated Shannon entropy in bits.
            latency_ms: Total routing latency in milliseconds.
            query_text: Raw query text.
            language: Detected dialect.
            action: Specific gateway action taken.
            is_guardrail_triggered: Whether query was blocked by safety filters.
        """
        record = QueryDriftRecord(
            query_length=len(query_text),
            routed_class=routed_class,
            confidence=confidence,
            entropy=entropy,
            latency_ms=latency_ms,
            language=language,
            is_guardrail_triggered=is_guardrail_triggered,
        )
        self.window.append(record)

        if self.prometheus_available:
            try:
                self.query_counter.labels(
                    routed_class=routed_class,
                    language=language,
                    action=action,
                ).inc()

                if is_guardrail_triggered:
                    self.guardrail_counter.inc()

                # Calculate estimated savings relative to frontier baseline ($0.0300 / 1k)
                frontier_cost = self.config.routing.token_costs_per_1k.get(
                    "frontier_reasoning", 0.0300
                )
                actual_cost = self.config.routing.token_costs_per_1k.get(
                    routed_class, 0.0
                )
                savings = max(0.0, frontier_cost - actual_cost)
                self.cost_savings_counter.inc(savings)

                self.latency_histogram.observe(latency_ms)
                self.entropy_gauge.set(entropy)
            except Exception as e:
                logger.debug(f"Prometheus metric update skipped: {e}")

    def get_summary(self) -> DriftSummary:
        """Compute summary statistics and detect drift across the rolling window.

        Returns:
            DriftSummary with current distribution and alerts.
        """
        total = len(self.window)
        if total == 0:
            return DriftSummary(
                window_size=self.window_size,
                total_records=0,
                class_distribution={},
                mean_confidence=1.0,
                mean_entropy=0.0,
                mean_latency_ms=0.0,
                guardrail_trigger_rate=0.0,
                is_drift_detected=False,
                drift_reasons=[],
            )

        counts: Dict[str, int] = collections.defaultdict(int)
        sum_conf = 0.0
        sum_entropy = 0.0
        sum_latency = 0.0
        guardrail_count = 0

        for rec in self.window:
            counts[rec.routed_class] += 1
            sum_conf += rec.confidence
            sum_entropy += rec.entropy
            sum_latency += rec.latency_ms
            if rec.is_guardrail_triggered:
                guardrail_count += 1

        distribution = {cls: count / total for cls, count in counts.items()}
        mean_conf = sum_conf / total
        mean_ent = sum_entropy / total
        mean_lat = sum_latency / total
        guardrail_rate = guardrail_count / total

        # Drift Detection Logic
        is_drift = False
        reasons: List[str] = []

        # 1. High Entropy Alert (Model uncertainty)
        if mean_ent > self.entropy_threshold:
            is_drift = True
            reasons.append(
                f"Elevated Mean Entropy ({mean_ent:.3f} > {self.entropy_threshold:.3f}): "
                "indicates distribution shift or ambiguous input distribution."
            )

        # 2. Latency SLO Breach
        if mean_lat > self.latency_slo_ms:
            reasons.append(
                f"Latency SLO Degradation (Mean {mean_lat:.2f}ms > SLO {self.latency_slo_ms}ms)"
            )

        # 3. Class Imbalance Drift (e.g., sudden spike in guardrail attacks > 25%)
        if guardrail_rate > 0.25:
            is_drift = True
            reasons.append(
                f"Guardrail Attack Spike (Blocked queries = {guardrail_rate * 100:.1f}%)"
            )

        return DriftSummary(
            window_size=self.window_size,
            total_records=total,
            class_distribution=distribution,
            mean_confidence=float(mean_conf),
            mean_entropy=float(mean_ent),
            mean_latency_ms=float(mean_lat),
            guardrail_trigger_rate=float(guardrail_rate),
            is_drift_detected=is_drift,
            drift_reasons=reasons,
        )

    def export_metrics(self) -> bytes:
        """Export raw Prometheus metrics in text format."""
        if self.prometheus_available:
            return generate_latest(self.registry)
        return b"# Prometheus metrics not available\n"
