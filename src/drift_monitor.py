"""Real-time MLOps drift monitor, PSI engine, and Prometheus registry for RouteOps.

Tracks query distributions, prediction entropy, confidence degradation,
Population Stability Index (PSI), Jensen-Shannon Divergence (JSD),
and inference latency across rolling request windows to detect concept drift.
"""

from __future__ import annotations

import collections
import logging
import time
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from src.config import RouteOpsConfig, load_config

logger = logging.getLogger("routeops.drift_monitor")


# -----------------------------------------------------------------------------
# Mathematical Drift Metrics: PSI & Jensen-Shannon Divergence
# -----------------------------------------------------------------------------


def calculate_psi(
    expected: Union[np.ndarray, List[float]],
    actual: Union[np.ndarray, List[float]],
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between two continuous distributions.

    PSI = sum((Actual_b - Expected_b) * ln(Actual_b / Expected_b))

    Standard Interpretations:
        - PSI < 0.10: Stable (No significant shift)
        - 0.10 <= PSI < 0.25: Moderate Shift (Warning, monitor closely)
        - PSI >= 0.25: Significant Shift (Critical alert, trigger model retraining)

    Args:
        expected: Reference/baseline sample values.
        actual: Production/evaluation sample values.
        num_bins: Number of quantile bins to discretize values.
        epsilon: Small smoothing constant to prevent division by zero or log(0).

    Returns:
        Calculated non-negative PSI score.
    """
    exp_arr = np.asarray(expected, dtype=np.float64).flatten()
    act_arr = np.asarray(actual, dtype=np.float64).flatten()

    if len(exp_arr) == 0 or len(act_arr) == 0:
        return 0.0

    # If all values are identical in both sets
    if np.all(exp_arr == exp_arr[0]) and np.all(act_arr == exp_arr[0]):
        return 0.0

    # Determine quantile bin edges from expected distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(exp_arr, percentiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        bin_edges = np.array([exp_arr.min() - 1e-5, exp_arr.max() + 1e-5])

    # Extend boundary bins to catch extreme outliers in actual
    bin_edges[0] = min(bin_edges[0], float(act_arr.min())) - 1e-5
    bin_edges[-1] = max(bin_edges[-1], float(act_arr.max())) + 1e-5

    # Compute bin frequency counts
    exp_counts, _ = np.histogram(exp_arr, bins=bin_edges)
    act_counts, _ = np.histogram(act_arr, bins=bin_edges)

    # Convert to probability distributions with smoothing
    exp_pct = (exp_counts.astype(np.float64) + epsilon) / (
        len(exp_arr) + epsilon * len(exp_counts)
    )
    act_pct = (act_counts.astype(np.float64) + epsilon) / (
        len(act_arr) + epsilon * len(act_counts)
    )

    # Calculate PSI summation
    psi_value = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(max(0.0, psi_value))


def calculate_js_divergence(
    p: Union[Dict[str, float], np.ndarray, List[float]],
    q: Union[Dict[str, float], np.ndarray, List[float]],
) -> float:
    """Calculate symmetric Jensen-Shannon Divergence (JSD) using base-2 logarithm.

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M) where M = 0.5 * (P + Q)
    Bounded in [0.0, 1.0].

    Args:
        p: First probability distribution (dict of class:prob or array).
        q: Second probability distribution.

    Returns:
        JSD score in range [0.0, 1.0].
    """
    if isinstance(p, dict) and isinstance(q, dict):
        all_keys = sorted(set(p.keys()).union(set(q.keys())))
        p_vec = np.array([p.get(k, 0.0) for k in all_keys], dtype=np.float64)
        q_vec = np.array([q.get(k, 0.0) for k in all_keys], dtype=np.float64)
    else:
        p_vec = np.asarray(p, dtype=np.float64).flatten()
        q_vec = np.asarray(q, dtype=np.float64).flatten()

    p_sum = np.sum(p_vec)
    q_sum = np.sum(q_vec)

    if p_sum > 0:
        p_vec /= p_sum
    if q_sum > 0:
        q_vec /= q_sum

    # Intermediate mixture distribution
    m = 0.5 * (p_vec + q_vec)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 1e-12) & (b > 1e-12)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    jsd = 0.5 * _kl(p_vec, m) + 0.5 * _kl(q_vec, m)
    return float(np.clip(jsd, 0.0, 1.0))


def detect_low_confidence_spike(
    confidences: List[float],
    threshold: float = 0.70,
    alert_rate: float = 0.15,
) -> Tuple[bool, float]:
    """Detect if the percentage of low-confidence predictions exceeds alert threshold.

    Args:
        confidences: List of predicted confidence scores.
        threshold: Confidence cutoff below which a prediction is considered low-confidence.
        alert_rate: Proportion of low-confidence queries that triggers an alert (default: 15%).

    Returns:
        Tuple of (is_spike_detected, observed_rate).
    """
    if not confidences:
        return False, 0.0
    low_count = sum(1 for c in confidences if c < threshold)
    rate = low_count / len(confidences)
    return (rate >= alert_rate), float(rate)


# -----------------------------------------------------------------------------
# Data Schemas
# -----------------------------------------------------------------------------


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
    """Aggregate statistics and drift diagnostic analysis over a rolling monitoring window."""

    window_size: int
    total_records: int
    class_distribution: Dict[str, float]
    mean_confidence: float
    mean_entropy: float
    mean_latency_ms: float
    guardrail_trigger_rate: float
    low_confidence_rate: float = 0.0
    psi_token_length: float = 0.0
    psi_confidence: float = 0.0
    js_divergence_class: float = 0.0
    drift_severity: str = "STABLE"  # STABLE, MODERATE, CRITICAL
    is_retrain_recommended: bool = False
    is_drift_detected: bool = False
    drift_reasons: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Production Drift Monitor Engine
# -----------------------------------------------------------------------------


class DriftMonitor:
    """Production MLOps monitoring engine for tracking inference stability, PSI, and drift."""

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

        # Baseline reference expectations
        self.baseline_distribution: Dict[str, float] = {
            "direct_response": 0.25,
            "rag_retrieval": 0.35,
            "frontier_reasoning": 0.35,
            "guardrail_blocked": 0.05,
        }
        self.baseline_lengths: List[float] = [25.0, 35.0, 50.0, 65.0, 80.0, 100.0]
        self.baseline_confidences: List[float] = [0.85, 0.90, 0.92, 0.95, 0.97, 0.98]

        # Setup Prometheus Metrics
        self._init_prometheus(custom_registry)

    def set_reference_distributions(
        self,
        baseline_lengths: List[float],
        baseline_confidences: List[float],
        baseline_class_dist: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update reference baseline distributions from validated training/val sets."""
        if baseline_lengths:
            self.baseline_lengths = list(baseline_lengths)
        if baseline_confidences:
            self.baseline_confidences = list(baseline_confidences)
        if baseline_class_dist:
            self.baseline_distribution = dict(baseline_class_dist)

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
            self.psi_gauge = Gauge(
                "routeops_confidence_psi",
                "Current Population Stability Index on prediction confidence",
                registry=self.registry,
            )
        except ValueError:
            # Handle metrics re-registration gracefully
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
        """Record telemetry for a single query into the rolling buffer and Prometheus."""
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
                logger.debug("Prometheus metric update skipped: %s", e)

    def get_summary(self) -> DriftSummary:
        """Compute aggregate summary, PSI, JSD, and retrain alerts across the window."""
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
                low_confidence_rate=0.0,
                psi_token_length=0.0,
                psi_confidence=0.0,
                js_divergence_class=0.0,
                drift_severity="STABLE",
                is_retrain_recommended=False,
                is_drift_detected=False,
                drift_reasons=[],
            )

        counts: Dict[str, int] = collections.defaultdict(int)
        confidences: List[float] = []
        lengths: List[float] = []
        sum_entropy = 0.0
        sum_latency = 0.0
        guardrail_count = 0

        for rec in self.window:
            counts[rec.routed_class] += 1
            confidences.append(rec.confidence)
            lengths.append(float(rec.query_length))
            sum_entropy += rec.entropy
            sum_latency += rec.latency_ms
            if rec.is_guardrail_triggered:
                guardrail_count += 1

        distribution = {cls: count / total for cls, count in counts.items()}
        mean_conf = float(np.mean(confidences))
        mean_ent = sum_entropy / total
        mean_lat = sum_latency / total
        guardrail_rate = guardrail_count / total

        # 1. Population Stability Index (PSI)
        psi_len = calculate_psi(self.baseline_lengths, lengths)
        psi_conf = calculate_psi(self.baseline_confidences, confidences)

        # 2. Jensen-Shannon Divergence
        jsd_class = calculate_js_divergence(self.baseline_distribution, distribution)

        # 3. Low-Confidence Spike Detection (< 0.70)
        has_low_conf_spike, low_conf_rate = detect_low_confidence_spike(
            confidences, threshold=0.70, alert_rate=0.15
        )

        # Drift Diagnostics & Retrain Trigger Evaluation
        is_drift = False
        reasons: List[str] = []
        severity = "STABLE"

        # Check PSI on confidence
        if psi_conf >= 0.25:
            is_drift = True
            severity = "CRITICAL"
            reasons.append(
                f"Critical Confidence PSI Shift ({psi_conf:.4f} >= 0.25): "
                "Model prediction stability degraded."
            )
        elif psi_conf >= 0.10:
            is_drift = True
            severity = "MODERATE"
            reasons.append(
                f"Moderate Confidence PSI Shift ({psi_conf:.4f} >= 0.10): "
                "Early distribution drift detected."
            )

        # Check PSI on query lengths
        if psi_len >= 0.25:
            is_drift = True
            if severity != "CRITICAL":
                severity = "MODERATE"
            reasons.append(
                f"Query Length PSI Shift ({psi_len:.4f} >= 0.25): "
                "Input token length distribution altered."
            )

        # Check Low Confidence Spike
        if has_low_conf_spike:
            is_drift = True
            severity = "CRITICAL"
            reasons.append(
                f"Low-Confidence Spike ({low_conf_rate * 100:.1f}% queries < 0.70 threshold): "
                "Signals new slang or Out-of-Distribution queries."
            )

        # Check Class Distribution Divergence (JSD > 0.15)
        if jsd_class > 0.15:
            is_drift = True
            if severity != "CRITICAL":
                severity = "MODERATE"
            reasons.append(
                f"Class Distribution Shift (JSD = {jsd_class:.4f} > 0.15): "
                "Target mix deviated from baseline."
            )

        # Check Entropy Alert
        if mean_ent > self.entropy_threshold:
            is_drift = True
            reasons.append(
                f"Elevated Mean Entropy ({mean_ent:.3f} > {self.entropy_threshold:.3f}): "
                "Ambiguity rise."
            )

        # Check Latency SLO Breach
        if mean_lat > self.latency_slo_ms:
            reasons.append(
                f"Latency SLO Degradation (Mean {mean_lat:.2f}ms > SLO {self.latency_slo_ms}ms)."
            )

        # Check Guardrail Attack Surge (> 25%)
        if guardrail_rate > 0.25:
            is_drift = True
            severity = "CRITICAL"
            reasons.append(
                f"Guardrail Attack Spike (Blocked queries = {guardrail_rate * 100:.1f}%)."
            )

        retrain_recommended = (
            (severity == "CRITICAL") or (psi_conf >= 0.25) or (low_conf_rate >= 0.20)
        )

        if self.prometheus_available:
            try:
                self.psi_gauge.set(psi_conf)
            except Exception:
                pass

        return DriftSummary(
            window_size=self.window_size,
            total_records=total,
            class_distribution=distribution,
            mean_confidence=mean_conf,
            mean_entropy=float(mean_ent),
            mean_latency_ms=float(mean_lat),
            guardrail_trigger_rate=float(guardrail_rate),
            low_confidence_rate=float(low_conf_rate),
            psi_token_length=round(float(psi_len), 4),
            psi_confidence=round(float(psi_conf), 4),
            js_divergence_class=round(float(jsd_class), 4),
            drift_severity=severity,
            is_retrain_recommended=retrain_recommended,
            is_drift_detected=is_drift,
            drift_reasons=reasons,
        )

    def export_metrics(self) -> bytes:
        """Export raw Prometheus metrics in text format."""
        if self.prometheus_available:
            return generate_latest(self.registry)
        return b"# Prometheus metrics not available\n"
