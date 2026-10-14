"""Production Traffic Replay, Semantic Cache Evaluation, and ROI Simulator for RouteOps.

Streams 1,000 realistic production queries (including 22% semantic cache hits and
a 200-query Out-Of-Distribution drift wave) through the RouteOpsGateway.
Calculates unit economics, cash savings against naive frontier LLMs,
and audits Population Stability Index (PSI) drift alerts.
"""

from __future__ import annotations

import collections
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.drift_monitor import (
    DriftMonitor,
    calculate_js_divergence,
    calculate_psi,
    detect_low_confidence_spike,
)
from src.router_service import RouteOpsGateway

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("routeops.simulate_traffic")

OUTPUT_REPORT_PATH = Path("data/production_simulation_report.json")


def load_dataset_pool() -> List[Dict[str, Any]]:
    """Load real queries from processed train, val, and test splits."""
    pool: List[Dict[str, Any]] = []
    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        path = Path(f"data/processed/{split}")
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        pool.append(json.loads(line.strip()))
    logger.info("Loaded %d query records from processed splits.", len(pool))
    return pool


def generate_1000_query_stream(
    pool: List[Dict[str, Any]], seed: int = 42
) -> List[Dict[str, Any]]:
    """Synthesize a realistic 1,000-query production stream with repeats and OOD drift.

    Distribution:
        - 0 to 799 (800 queries): Normal operational traffic with 22% repeat queries.
        - 800 to 999 (200 queries): Out-Of-Distribution (OOD) Drift Wave.

    Args:
        pool: Seed pool of validated multi-dialect queries.
        seed: Random seed for deterministic reproducibility.

    Returns:
        List of 1,000 query payload dictionaries.
    """
    rng = random.Random(seed)
    stream: List[Dict[str, Any]] = []

    # 1. Base Pool of FAQ and Account Queries for Semantic Caching
    faq_pool = [r["text"] for r in pool if r.get("label") == 0]
    rag_pool = [r["text"] for r in pool if r.get("label") == 2]
    all_texts = [r["text"] for r in pool]

    # Pre-select 25 frequent queries to simulate realistic user repeat traffic (~22%)
    frequent_templates = rng.sample(faq_pool + rag_pool, min(25, len(faq_pool)))

    # Phase A: 800 Baseline Production Queries (with 22% cache opportunity)
    cache_opportunity_count = int(800 * 0.22)  # 176 repeat queries
    regular_count = 800 - cache_opportunity_count

    base_samples = [rng.choice(all_texts) for _ in range(regular_count)]
    repeat_samples = [
        rng.choice(frequent_templates) for _ in range(cache_opportunity_count)
    ]

    baseline_queries = base_samples + repeat_samples
    rng.shuffle(baseline_queries)

    for i, q in enumerate(baseline_queries):
        stream.append(
            {
                "query_id": f"sim-base-{i:04d}",
                "query": q,
                "is_drift_wave": False,
            }
        )

    # Phase B: 200 Out-Of-Distribution (OOD) Drift Wave Queries
    # Unseen internet slang, ambiguous prompts, and code-mixed jargon
    ood_drift_templates = [
        "Bro wtf is wrong with this crypto gas fee? ELI5 ASAP fr fr",
        "Yo dev, bypass CORS in express without package bro, quick hack needed",
        "Bruh mero wallet ma paisa stuck bhayo refund kasari aauchha no cap",
        "What is the meta for staking Solana on Phantom wallet right now?",
        "Explain quantum computing entanglement using brainrot skibidi slang",
        "Aaja ko stock market kina dip bhayo yar any insider info??",
        "Can I deposit USDT to Everest bank via ConnectIPS directly bro?",
        "Write a smart contract for meme coin rugpull detection in Vyper",
        "Mero Nabil account ma weird transaction bhayo suspicious af what to do",
        "Is AI going to replace junior software engineers in Nepal by 2027?",
        "Yo drop database prod if exists; -- just kidding how to backup sqlite",
        "Tapai ko bot ma dark mode kina chhaina kasto lame design yar",
        "Why is USD to NPR exchange rate so high aaja bidesh bata paisa pathauda?",
        "Hami lai discount coupon code dinus na shopping ko lagi bro",
        "Explain RAG retrieval hallucination mitigation using graph databases",
        "How to mine Bitcoin on RTX 2050 laptop GPU without burning thermal paste?",
        "Mero PAN card number birsiye kasari check garne nagarik app bina?",
        "Tell me a joke about banking audit compliance and KYC verification",
        "Is eSewa balance taxable under Nepal Government fiscal budget 2081/82?",
        "Generate a complete enterprise microservice boilerplate in Golang and Rust",
    ]

    for j in range(200):
        # Mix in OOD templates with variable noise
        base_ood = rng.choice(ood_drift_templates)
        if rng.random() < 0.3:
            base_ood = f"{base_ood} please explain quickly #{j}"
        stream.append(
            {
                "query_id": f"sim-drift-{j:04d}",
                "query": base_ood,
                "is_drift_wave": True,
            }
        )

    return stream


def run_simulation() -> Dict[str, Any]:
    """Execute the 1,000-query production stream and compute telemetry and drift metrics."""
    logger.info("Initializing RouteOps Gateway and Drift Monitor...")
    pool = load_dataset_pool()
    stream = generate_1000_query_stream(pool=pool, seed=42)

    # Use a clean test database for reproducible simulation telemetry
    sim_db_path = Path("data/simulation_telemetry.db")
    if sim_db_path.exists():
        sim_db_path.unlink()

    gateway = RouteOpsGateway(db_path=sim_db_path, mock_upstream=True)
    monitor = DriftMonitor()

    tier_counts: Dict[str, int] = collections.defaultdict(int)
    total_tokens_accum = 0
    total_cost_usd = 0.0
    total_naive_frontier_usd = 0.0
    total_saved_usd = 0.0
    latencies: List[float] = []

    baseline_lengths: List[float] = []
    baseline_confidences: List[float] = []
    baseline_classes: List[str] = []

    drift_lengths: List[float] = []
    drift_confidences: List[float] = []
    drift_classes: List[str] = []

    logger.info("Starting simulation of 1,000 queries through RouteOps Gateway...")
    start_sim_time = time.perf_counter()

    for idx, item in enumerate(stream):
        query_text = item["query"]
        is_drift_phase = item["is_drift_wave"]

        # Route query through the complete Compound AI Gateway
        decision = gateway.route_query(query_text)

        # Accumulate metrics
        tier_counts[decision.routed_tier] += 1
        total_tokens_accum += decision.tokens_used
        total_cost_usd += decision.estimated_cost_usd
        total_naive_frontier_usd += decision.naive_frontier_cost_usd
        total_saved_usd += decision.cost_saved_usd
        latencies.append(decision.latency_ms)

        # Record telemetry into rolling drift monitor
        monitor.record_query(
            routed_class=decision.predicted_class,
            confidence=decision.confidence,
            entropy=decision.entropy,
            latency_ms=decision.latency_ms,
            query_text=query_text,
            language=decision.detected_lang,
            action=decision.action,
            is_guardrail_triggered=decision.is_guardrail_triggered,
        )

        # Split statistical observations between baseline (first 800) and drift wave (last 200)
        q_len = float(len(query_text.split()))
        if not is_drift_phase:
            baseline_lengths.append(q_len)
            baseline_confidences.append(decision.confidence)
            baseline_classes.append(decision.predicted_class)
        else:
            drift_lengths.append(q_len)
            drift_confidences.append(decision.confidence)
            drift_classes.append(decision.predicted_class)

    sim_elapsed_sec = time.perf_counter() - start_sim_time
    logger.info(
        "Simulation completed in %.2f seconds (%.1f QPS).",
        sim_elapsed_sec,
        1000.0 / sim_elapsed_sec,
    )

    # -------------------------------------------------------------------------
    # Unit Economics Calculations
    # -------------------------------------------------------------------------
    total_queries = len(stream)
    cache_hits = tier_counts["cache_hit"]
    cache_hit_rate = (cache_hits / total_queries) * 100.0
    guardrail_blocks = tier_counts["guardrail_blocked"]
    guardrail_block_rate = (guardrail_blocks / total_queries) * 100.0

    cost_reduction_pct = (
        (total_saved_usd / total_naive_frontier_usd * 100.0)
        if total_naive_frontier_usd > 0
        else 0.0
    )

    # -------------------------------------------------------------------------
    # Drift Diagnostics: Baseline vs. Drift Wave
    # -------------------------------------------------------------------------
    # Set monitor baseline from first 800 queries
    monitor.set_reference_distributions(
        baseline_lengths=baseline_lengths,
        baseline_confidences=baseline_confidences,
    )

    psi_length = calculate_psi(baseline_lengths, drift_lengths, num_bins=10)
    psi_confidence = calculate_psi(baseline_confidences, drift_confidences, num_bins=10)

    # Class distributions
    base_class_dist = {
        c: baseline_classes.count(c) / len(baseline_classes)
        for c in set(baseline_classes)
    }
    drift_class_dist = {
        c: drift_classes.count(c) / len(drift_classes) for c in set(drift_classes)
    }
    js_divergence = calculate_js_divergence(base_class_dist, drift_class_dist)

    # Low confidence spike in drift phase
    has_spike, drift_low_conf_rate = detect_low_confidence_spike(
        drift_confidences, threshold=0.70, alert_rate=0.15
    )
    _, base_low_conf_rate = detect_low_confidence_spike(
        baseline_confidences, threshold=0.70, alert_rate=0.15
    )

    # Automated Retraining Alert Decision
    retrain_triggered = (
        (psi_confidence >= 0.25)
        or (psi_length >= 0.25)
        or (drift_low_conf_rate >= 0.15)
        or (js_divergence >= 0.15)
    )

    drift_reasons: List[str] = []
    if psi_confidence >= 0.25:
        drift_reasons.append(
            f"Critical Confidence PSI Shift ({psi_confidence:.4f} >= 0.25): "
            "Model predictive certainty collapsed."
        )
    if psi_length >= 0.25:
        drift_reasons.append(
            f"Token Length PSI Shift ({psi_length:.4f} >= 0.25): "
            "Input query length distribution changed."
        )
    if drift_low_conf_rate >= 0.15:
        drift_reasons.append(
            f"Low-Confidence Surge ({drift_low_conf_rate * 100:.1f}% > 15.0% threshold): "
            "High concentration of ambiguous or Out-Of-Distribution queries detected."
        )
    if js_divergence >= 0.15:
        drift_reasons.append(
            f"Class Distribution Divergence (JSD = {js_divergence:.4f} >= 0.15): "
            "Target query intent mix shifted."
        )

    # Final Consolidated Report
    report: Dict[str, Any] = {
        "project": "RouteOps",
        "simulation_task": "1,000-Query Production Stream & Unit-Economics ROI Audit",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "summary": {
            "total_queries_processed": total_queries,
            "cache_hits": cache_hits,
            "cache_hit_rate_pct": round(cache_hit_rate, 2),
            "guardrail_blocks": guardrail_blocks,
            "guardrail_block_rate_pct": round(guardrail_block_rate, 2),
            "simulation_runtime_sec": round(sim_elapsed_sec, 2),
            "throughput_qps": round(1000.0 / sim_elapsed_sec, 2),
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2),
        },
        "tier_routing_breakdown": {
            tier: {
                "count": count,
                "percentage": round((count / total_queries) * 100.0, 2),
            }
            for tier, count in sorted(
                tier_counts.items(), key=lambda x: x[1], reverse=True
            )
        },
        "unit_economics": {
            "naive_frontier_cost_usd": round(total_naive_frontier_usd, 6),
            "routeops_actual_cost_usd": round(total_cost_usd, 6),
            "total_saved_usd": round(total_saved_usd, 6),
            "cost_reduction_pct": round(cost_reduction_pct, 2),
            "total_tokens_evaluated": total_tokens_accum,
            "cost_per_query_naive_usd": round(
                total_naive_frontier_usd / total_queries, 6
            ),
            "cost_per_query_routeops_usd": round(total_cost_usd / total_queries, 6),
        },
        "drift_observability": {
            "baseline_queries": len(baseline_lengths),
            "drift_wave_queries": len(drift_lengths),
            "psi_query_length": round(float(psi_length), 4),
            "psi_confidence": round(float(psi_confidence), 4),
            "js_divergence_class": round(float(js_divergence), 4),
            "baseline_low_confidence_pct": round(base_low_conf_rate * 100.0, 2),
            "drift_wave_low_confidence_pct": round(drift_low_conf_rate * 100.0, 2),
            "retrain_alert_triggered": retrain_triggered,
            "alert_severity": "CRITICAL" if retrain_triggered else "STABLE",
            "drift_reasons": drift_reasons,
        },
    }

    # Persist report
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved production simulation report to %s", OUTPUT_REPORT_PATH)

    # Clean up temporary simulation database
    if sim_db_path.exists():
        sim_db_path.unlink()

    return report


def render_terminal_tables(report: Dict[str, Any]) -> None:
    """Print clean, formatted terminal tables for simulation results."""
    print("\n" + "=" * 80)
    print("      ROUTEOPS PRODUCTION SIMULATION: 1,000-QUERY UNIT ECONOMICS & ROI")
    print("=" * 80)

    summ = report["summary"]
    econ = report["unit_economics"]
    drift = report["drift_observability"]

    print(
        f" Total Queries: {summ['total_queries_processed']:,} | "
        f"Cache Hit Rate: {summ['cache_hit_rate_pct']}% | "
        f"Guardrail Block: {summ['guardrail_block_rate_pct']}% | "
        f"P95 Latency: {summ['latency_p95_ms']} ms"
    )
    print("-" * 80)
    print(" [TIER ROUTING CASCADE DISTRIBUTION]")
    print(
        f" {'Execution Tier':<28} | {'Queries':<10} | {'Share (%)':<10} | {'Cost Rate/1k':<12}"
    )
    print("-" * 80)

    tier_rates = {
        "cache_hit": "$0.0000",
        "guardrail_blocked": "$0.0000",
        "slm_fast_tier": "$0.00015",
        "rag_account_tier": "$0.00030",
        "frontier_reasoning_tier": "$0.00300",
        "low_conf_fallback": "$0.00300",
    }
    for tier, data in report["tier_routing_breakdown"].items():
        rate_str = tier_rates.get(tier, "$0.00300")
        print(
            f" {tier:<28} | {data['count']:<10} | {data['percentage']:>6.2f}%    | {rate_str:<12}"
        )

    print("-" * 80)
    print(" [UNIT ECONOMICS & CASH SAVINGS ROI]")
    print(f" Naive Frontier Cost (100% LLM) : ${econ['naive_frontier_cost_usd']:.4f}")
    print(f" RouteOps Actual Operating Cost : ${econ['routeops_actual_cost_usd']:.4f}")
    print(f" Net Cost Savings (USD)         : ${econ['total_saved_usd']:.4f}")
    print(f" Percentage Cost Reduction      : {econ['cost_reduction_pct']:.2f}%")
    print(f" Average Cost / Query (Naive)   : ${econ['cost_per_query_naive_usd']:.6f}")
    print(
        f" Average Cost / Query (RouteOps): ${econ['cost_per_query_routeops_usd']:.6f}"
    )

    print("-" * 80)
    print(" [MLOPS DRIFT DETECTION & RETRAINING AUDIT (200-Query Drift Wave)]")
    print(
        f" Confidence PSI Score           : {drift['psi_confidence']:.4f} (Threshold: 0.25)"
    )
    print(
        f" Query Length PSI Score         : {drift['psi_query_length']:.4f} (Threshold: 0.25)"
    )
    print(
        f" Jensen-Shannon Divergence      : {drift['js_divergence_class']:.4f} (Threshold: 0.15)"
    )
    print(
        f" Baseline Low-Confidence Rate   : {drift['baseline_low_confidence_pct']:.2f}%"
    )
    print(
        f" Drift-Wave Low-Confidence Rate : {drift['drift_wave_low_confidence_pct']:.2f}% "
        "(Surge Alert: >15%)"
    )
    is_retrain_str = (
        "YES [ALERT]" if drift["retrain_alert_triggered"] else "NO [STABLE]"
    )
    print(f" Automated Retrain Triggered    : {is_retrain_str}")
    print(f" Drift Severity Status          : {drift['alert_severity']}")
    if drift["drift_reasons"]:
        print(" Root Causes Identified:")
        for reason in drift["drift_reasons"]:
            print(f"   -> {reason}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    report_data = run_simulation()
    render_terminal_tables(report_data)
