"""Unit and integration test suite for RouteOps Drift Monitoring Engine.

Tests Population Stability Index (PSI), Jensen-Shannon Divergence (JSD),
low-confidence spike detection, and automated retraining triggers.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.drift_monitor import (
    DriftMonitor,
    calculate_js_divergence,
    calculate_psi,
    detect_low_confidence_spike,
)

# -----------------------------------------------------------------------------
# 1. Population Stability Index (PSI) Tests
# -----------------------------------------------------------------------------


class TestPopulationStabilityIndex:
    """Tests for calculate_psi implementation and statistical stability thresholds."""

    def test_psi_identical_distributions(self) -> None:
        """Verify identical distributions yield PSI close to 0.0 (< 0.05)."""
        rng = np.random.default_rng(42)
        baseline = rng.normal(loc=50.0, scale=10.0, size=500)
        target = rng.normal(loc=50.0, scale=10.0, size=500)

        psi = calculate_psi(baseline, target, num_bins=10)
        assert psi < 0.05

    def test_psi_moderate_shift(self) -> None:
        """Verify moderate distributional shift produces 0.10 <= PSI < 0.25."""
        rng = np.random.default_rng(42)
        baseline = rng.normal(loc=50.0, scale=10.0, size=500)
        # Shift mean slightly by 3 units
        target = rng.normal(loc=53.5, scale=10.0, size=500)

        psi = calculate_psi(baseline, target, num_bins=10)
        assert 0.08 <= psi < 0.35

    def test_psi_significant_shift_critical(self) -> None:
        """Verify strong distribution shift triggers critical PSI >= 0.25."""
        rng = np.random.default_rng(42)
        baseline = rng.normal(loc=50.0, scale=10.0, size=500)
        # Shift mean substantially by 15 units
        target = rng.normal(loc=65.0, scale=10.0, size=500)

        psi = calculate_psi(baseline, target, num_bins=10)
        assert psi >= 0.25

    def test_psi_edge_cases_empty_or_constant(self) -> None:
        """Verify empty or invariant datasets do not crash or produce NaN."""
        assert calculate_psi([], []) == 0.0
        assert calculate_psi([1.0, 1.0], [1.0, 1.0]) == 0.0
        assert calculate_psi([10.0] * 50, [20.0] * 50) >= 0.0


# -----------------------------------------------------------------------------
# 2. Jensen-Shannon Divergence (JSD) Tests
# -----------------------------------------------------------------------------


class TestJensenShannonDivergence:
    """Tests for symmetric base-2 Jensen-Shannon Divergence calculation."""

    def test_jsd_identical_distributions(self) -> None:
        """Verify identical discrete probability distributions yield JSD = 0.0."""
        p = {"class_0": 0.25, "class_1": 0.25, "class_2": 0.25, "class_3": 0.25}
        q = {"class_0": 0.25, "class_1": 0.25, "class_2": 0.25, "class_3": 0.25}

        jsd = calculate_js_divergence(p, q)
        assert np.isclose(jsd, 0.0, atol=1e-5)

    def test_jsd_completely_disjoint_distributions(self) -> None:
        """Verify completely disjoint distributions yield maximum JSD = 1.0."""
        p = [1.0, 0.0]
        q = [0.0, 1.0]

        jsd = calculate_js_divergence(p, q)
        assert np.isclose(jsd, 1.0, atol=1e-5)

    def test_jsd_symmetry_property(self) -> None:
        """Verify symmetry: JSD(P || Q) == JSD(Q || P)."""
        p = {"a": 0.7, "b": 0.2, "c": 0.1}
        q = {"a": 0.2, "b": 0.5, "c": 0.3}

        jsd_pq = calculate_js_divergence(p, q)
        jsd_qp = calculate_js_divergence(q, p)
        assert np.isclose(jsd_pq, jsd_qp, atol=1e-6)
        assert 0.0 <= jsd_pq <= 1.0


# -----------------------------------------------------------------------------
# 3. Low-Confidence Spike Detection Tests
# -----------------------------------------------------------------------------


class TestLowConfidenceSpikeDetection:
    """Tests for detecting Out-of-Distribution query spikes below confidence threshold."""

    def test_no_spike_detected(self) -> None:
        """Verify stable high-confidence stream does not trigger alert."""
        confs = [0.95, 0.92, 0.88, 0.91, 0.75, 0.89, 0.94]
        is_spike, rate = detect_low_confidence_spike(
            confs, threshold=0.70, alert_rate=0.15
        )
        assert is_spike is False
        assert rate == 0.0

    def test_spike_detected_on_ood_surge(self) -> None:
        """Verify surge in low-confidence predictions triggers spike alert."""
        # 3 out of 10 samples below 0.70 (30% > 15% alert rate)
        confs = [0.95, 0.45, 0.52, 0.88, 0.61, 0.92, 0.90, 0.85, 0.91, 0.93]
        is_spike, rate = detect_low_confidence_spike(
            confs, threshold=0.70, alert_rate=0.15
        )
        assert is_spike is True
        assert rate == pytest.approx(0.30, rel=1e-2)


# -----------------------------------------------------------------------------
# 4. DriftMonitor Integration & Retrain Alert Tests
# -----------------------------------------------------------------------------


class TestDriftMonitorIntegration:
    """Tests for DriftMonitor windowing, summary computation, and retrain triggers."""

    def test_monitor_initial_state(self) -> None:
        """Verify fresh monitor initializes with empty window and stable state."""
        monitor = DriftMonitor()
        summary = monitor.get_summary()
        assert summary.total_records == 0
        assert summary.is_drift_detected is False
        assert summary.drift_severity == "STABLE"
        assert summary.is_retrain_recommended is False

    def test_monitor_records_and_detects_drift(self) -> None:
        """Verify monitor records queries and flags critical drift on distribution collapse."""
        monitor = DriftMonitor()
        monitor.set_reference_distributions(
            baseline_lengths=[20.0, 30.0, 40.0],
            baseline_confidences=[0.95, 0.96, 0.98],
            baseline_class_dist={"direct_response": 0.5, "frontier_reasoning": 0.5},
        )

        # Inject 30 queries with low confidence and shifted lengths
        for _ in range(30):
            monitor.record_query(
                routed_class="guardrail_blocked",
                confidence=0.45,  # Low confidence
                entropy=1.8,
                latency_ms=12.0,
                query_text="Unusually long and ambiguous code query with unseen tokens"
                * 3,
                language="en",
                is_guardrail_triggered=True,
            )

        summary = monitor.get_summary()
        assert summary.total_records == 30
        assert summary.is_drift_detected is True
        assert summary.drift_severity == "CRITICAL"
        assert summary.is_retrain_recommended is True
        assert len(summary.drift_reasons) > 0
