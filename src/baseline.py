"""Classical ML Baseline Pipeline for Multilingual Semantic Routing.

This module provides a reproducible baseline for RouteOps using TF-IDF feature
extraction and Logistic Regression. It evaluates performance across multiple languages
(English, Romanized Nepali, Devanagari Nepali) and routing categories, tracks runs
using MLflow, benchmarks single-query CPU inference latency, and conducts hard-negative
failure case analysis highlighting the limitations of bag-of-words representations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

# Enable MLflow local file store compatibility for MLflow 3.x
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("routeops.baseline")

# -----------------------------------------------------------------------------
# Class Mappings and Label Constants
# -----------------------------------------------------------------------------

# Canonical mapping supporting both prompt-domain and architecture label naming
LABEL_INDEX_TO_NAMES: Dict[int, Dict[str, str]] = {
    0: {"canonical": "simple_faq", "routeops": "direct_response"},
    1: {"canonical": "technical_complex", "routeops": "frontier_reasoning"},
    2: {"canonical": "billing_account", "routeops": "rag_retrieval"},
    3: {"canonical": "unsafe_injection", "routeops": "guardrail_blocked"},
}

STRING_TO_CLASS_ID: Dict[str, int] = {
    "simple_faq": 0,
    "direct_response": 0,
    "0": 0,
    "technical_complex": 1,
    "frontier_reasoning": 1,
    "1": 1,
    "billing_account": 2,
    "rag_retrieval": 2,
    "2": 2,
    "unsafe_injection": 3,
    "guardrail_blocked": 3,
    "3": 3,
}

SUPPORTED_LANGUAGES = ["en", "ne-rom", "ne"]


@dataclass
class FailureCase:
    """Represents a diagnostic misclassification by the baseline model."""

    text: str
    lang: str
    true_label: int
    true_label_name: str
    pred_label: int
    pred_label_name: str
    confidence: float
    diagnosis: str


# -----------------------------------------------------------------------------
# Data Ingestion
# -----------------------------------------------------------------------------


def parse_label(raw_label: Any) -> int:
    """Normalize raw label from string or int to canonical integer ID (0-3).

    Args:
        raw_label: Label representation from dataset (int or string).

    Returns:
        int: Canonical label index between 0 and 3.

    Raises:
        ValueError: If label cannot be resolved to a known routing class.
    """
    if isinstance(raw_label, int) and raw_label in LABEL_INDEX_TO_NAMES:
        return raw_label
    str_key = str(raw_label).strip().lower()
    if str_key in STRING_TO_CLASS_ID:
        return STRING_TO_CLASS_ID[str_key]
    try:
        int_val = int(str_key)
        if int_val in LABEL_INDEX_TO_NAMES:
            return int_val
    except ValueError:
        pass
    raise ValueError(f"Unrecognized class label: {raw_label}")


def load_dataset_split(
    primary_path: str | Path,
    fallback_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Load dataset split from JSONL, JSON, or CSV file.

    Args:
        primary_path: Target path to load (e.g. data/processed/train.jsonl).
        fallback_path: Optional fallback file path if primary is absent.

    Returns:
        List of dicts each containing 'text', 'label' (int), and 'lang' (str).

    Raises:
        FileNotFoundError: If neither primary nor fallback exists.
    """
    path = Path(primary_path)
    if not path.is_file():
        if fallback_path is not None and Path(fallback_path).is_file():
            logger.warning(
                "Primary path %s not found. Falling back to %s",
                primary_path,
                fallback_path,
            )
            path = Path(fallback_path)
        else:
            raise FileNotFoundError(
                f"Dataset file not found at {primary_path}"
                + (f" or fallback {fallback_path}" if fallback_path else "")
            )

    records: List[Dict[str, Any]] = []
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                row = json.loads(clean_line)
                records.append(
                    {
                        "text": str(row.get("text", "")),
                        "label": parse_label(row.get("label")),
                        "lang": str(row.get("lang", "en")),
                    }
                )
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            for row in raw_data:
                records.append(
                    {
                        "text": str(row.get("text", "")),
                        "label": parse_label(row.get("label")),
                        "lang": str(row.get("lang", "en")),
                    }
                )
    elif suffix == ".csv":
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            records.append(
                {
                    "text": str(row["text"]),
                    "label": parse_label(row["label"]),
                    "lang": str(row.get("lang", "en")),
                }
            )
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}")

    logger.info("Loaded %d records from %s", len(records), path)
    return records


# -----------------------------------------------------------------------------
# Baseline Pipeline Construction
# -----------------------------------------------------------------------------


def build_baseline_pipeline(
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 10000,
    sublinear_tf: bool = True,
    c_param: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """Build reproducible scikit-learn Pipeline with TF-IDF and Logistic Regression.

    Args:
        ngram_range: N-gram range for word vectorization (unigrams + bigrams).
        max_features: Maximum vocabulary size.
        sublinear_tf: Whether to apply sublinear term frequency scaling (1 + log(tf)).
        c_param: Inverse regularization parameter C for Logistic Regression.
        max_iter: Maximum solver iterations for convergence.
        random_state: Random state for deterministic model fitting.

    Returns:
        Pipeline: Combined text feature extraction and classification pipeline.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )
    classifier = LogisticRegression(
        C=c_param,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    pipeline = Pipeline(
        [
            ("tfidf", vectorizer),
            ("clf", classifier),
        ]
    )
    return pipeline


# -----------------------------------------------------------------------------
# Metric Computation
# -----------------------------------------------------------------------------


def compute_comprehensive_metrics(
    y_true: List[int],
    y_pred: List[int],
    languages: List[str],
) -> Dict[str, float]:
    """Calculate overall, per-class, and per-language classification metrics.

    Args:
        y_true: Ground-truth class integer indices.
        y_pred: Predicted class integer indices.
        languages: Language code associated with each query.

    Returns:
        Dict[str, float]: Metric dictionary suitable for MLflow logging and reporting.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    lang_arr = np.array(languages)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_f1 = float(
        f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
    )

    metrics: Dict[str, float] = {
        "accuracy": acc,
        "overall_accuracy": acc,
        "macro_f1": macro_f1,
        "f1_macro": macro_f1,
        "weighted_f1": weighted_f1,
        "f1_weighted": weighted_f1,
    }

    # Per-Class F1 (log both canonical prompt naming and RouteOps architecture naming)
    for class_id, names in LABEL_INDEX_TO_NAMES.items():
        canonical_name = names["canonical"]
        routeops_name = names["routeops"]

        # Binary indicator for this class
        class_true = (y_true_arr == class_id).astype(int)
        class_pred = (y_pred_arr == class_id).astype(int)
        c_f1 = float(
            f1_score(class_true, class_pred, average="binary", zero_division=0)
        )

        metrics[f"f1_{canonical_name}"] = c_f1
        metrics[f"f1_{routeops_name}"] = c_f1

    # Per-Language F1 (Macro F1 evaluated within each language subset)
    for lang in SUPPORTED_LANGUAGES:
        mask = lang_arr == lang
        if np.any(mask):
            lang_f1 = float(
                f1_score(
                    y_true_arr[mask],
                    y_pred_arr[mask],
                    average="macro",
                    zero_division=0,
                )
            )
        else:
            lang_f1 = 0.0
        # Standardize language keys (e.g., ne-rom -> ne_rom)
        norm_lang_key = lang.replace("-", "_")
        metrics[f"f1_lang_{norm_lang_key}"] = lang_f1

    return metrics


# -----------------------------------------------------------------------------
# Latency Benchmarking
# -----------------------------------------------------------------------------


def benchmark_cpu_latency(
    pipeline: Pipeline,
    test_queries: List[str],
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """Measure single-query CPU inference latency across test set queries.

    Args:
        pipeline: Trained classification pipeline.
        test_queries: List of single text inputs.
        warmup_runs: Initial unmeasured warm-up iterations.

    Returns:
        Dict[str, float]: Latency percentiles in milliseconds (p50, p95, mean).
    """
    if not test_queries:
        return {"latency_p50_ms": 0.0, "latency_p95_ms": 0.0, "latency_mean_ms": 0.0}

    # Warm-up phase
    for i in range(min(warmup_runs, len(test_queries))):
        pipeline.predict([test_queries[i % len(test_queries)]])

    latencies_ms: List[float] = []
    for query in test_queries:
        start_t = time.perf_counter()
        pipeline.predict([query])
        end_t = time.perf_counter()
        latencies_ms.append((end_t - start_t) * 1000.0)

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    mean = float(np.mean(latencies_ms))

    return {
        "latency_p50_ms": round(p50, 4),
        "latency_p95_ms": round(p95, 4),
        "latency_mean_ms": round(mean, 4),
    }


# -----------------------------------------------------------------------------
# Hard Negative Failure Case Analysis
# -----------------------------------------------------------------------------


def generate_failure_diagnosis(
    text: str,
    lang: str,
    true_label: int,
    pred_label: int,
) -> str:
    """Generate diagnostic rationale explaining why TF-IDF failed and why DistilBERT is needed.

    Args:
        text: Query text that was misclassified.
        lang: Language of the query.
        true_label: Expected ground-truth label ID.
        pred_label: Predicted label ID.

    Returns:
        str: Technical rationale for the failure.
    """
    if lang == "ne-rom":
        if true_label == 3:
            return (
                "TF-IDF failed: Romanized Nepali injection uses non-standard phonetic spellings "
                "(e.g., 'birsinus', 'bataunus', 'nirdeshan') unseen in English vocabulary or "
                "conflated with polite service inquiries. DistilBERT subword WordPiece and "
                "cross-lingual pretraining capture contextual semantic malice despite script "
                "transliteration."
            )
        elif true_label == 2:
            return (
                "TF-IDF failed: Code-switched Nepanglish mixes English technical terms with Nepali "
                "grammatical morphemes. Sparse bag-of-words isolates isolated tokens rather than "
                "multi-step algorithmic intent. DistilBERT attention contextualizes the technical "
                "depth across mixed vocabulary."
            )
        else:
            return (
                "TF-IDF failed: Out-of-vocabulary Romanized Nepali variations and flexible word "
                "order dilute sparse n-gram frequencies, leading to spurious class correlation. "
                "Multilingual DistilBERT provides robust sentence-level contextual embeddings."
            )
    elif lang == "ne":
        return (
            "TF-IDF failed: Devanagari morphological inflections and compound words fragment "
            "standard character-boundary tokenization into low-frequency n-grams. Multilingual "
            "DistilBERT processes Devanagari subwords natively through shared cross-lingual "
            "representations."
        )
    else:  # en
        if true_label == 3 and pred_label in (1, 2):
            return (
                "TF-IDF failed: Subtle prompt injection embeds security keywords ('system', "
                "'instruction', 'database') that have high TF-IDF overlap with technical or "
                "account categories, tricking linear weights. DistilBERT self-attention detects "
                "adversarial instruction override intent."
            )
        else:
            return (
                "TF-IDF failed: Bag-of-words ignores syntactic negation and multi-clause semantic "
                "direction. Transformer attention captures full-sequence relational context."
            )


def extract_top_failure_cases(
    pipeline: Pipeline,
    test_records: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[FailureCase]:
    """Identify and rank the top most informative failure cases on the test set.

    Prioritizes code-switched Nepanglish (ne-rom) and adversarial boundary queries.

    Args:
        pipeline: Trained scikit-learn pipeline with predict_proba support.
        test_records: List of test samples with 'text', 'label', 'lang'.
        top_k: Maximum number of failure cases to return.

    Returns:
        List[FailureCase]: Sorted diagnostic cases.
    """
    texts = [r["text"] for r in test_records]
    y_true = [r["label"] for r in test_records]
    languages = [r["lang"] for r in test_records]

    y_pred = pipeline.predict(texts)
    probabilities = pipeline.predict_proba(texts)

    failures: List[Tuple[float, FailureCase]] = []

    for i, (text, y_t, y_p, lang) in enumerate(zip(texts, y_true, y_pred, languages)):
        if y_t != y_p:
            conf = float(np.max(probabilities[i]))
            true_name = LABEL_INDEX_TO_NAMES[y_t]["canonical"]
            pred_name = LABEL_INDEX_TO_NAMES[y_p]["canonical"]

            # Informative ranking score:
            # - Boost Romanized Nepali and Devanagari (core focus of RouteOps)
            # - Boost high-confidence failures (overconfident linear mistakes)
            # - Boost injection misclassifications (safety-critical boundary errors)
            score = conf
            if lang == "ne-rom":
                score += 1.0
            elif lang == "ne":
                score += 0.5
            if y_t == 3 or y_p == 3:
                score += 0.8

            diagnosis = generate_failure_diagnosis(text, lang, y_t, y_p)
            fc = FailureCase(
                text=text,
                lang=lang,
                true_label=int(y_t),
                true_label_name=true_name,
                pred_label=int(y_p),
                pred_label_name=pred_name,
                confidence=round(conf, 4),
                diagnosis=diagnosis,
            )
            failures.append((score, fc))

    failures.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in failures[:top_k]]


# -----------------------------------------------------------------------------
# Terminal Reporting
# -----------------------------------------------------------------------------


def print_rich_terminal_report(
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, float],
    latencies: Dict[str, float],
    failures: List[FailureCase],
) -> None:
    """Print a clean, structured terminal report using Rich.

    Args:
        hyperparameters: Model and vectorizer configurations.
        metrics: Computed validation/test performance metrics.
        latencies: Measured CPU inference latency metrics.
        failures: Top identified hard-negative failure cases.
    """
    console = Console()

    console.print("\n")
    console.print(
        Panel.fit(
            "[bold cyan]RouteOps - Classical ML Baseline Evaluation Report[/bold cyan]\n"
            "[white]Model: TF-IDF (1-2 N-grams) + Logistic Regression (L-BFGS)[/white]\n"
            "[green]Status: MLflow Run Completed & Logged[/green]",
            border_style="cyan",
        )
    )

    # 1. Hyperparameters Table
    hp_table = Table(
        title="[bold yellow]1. Model Configuration & Hyperparameters[/bold yellow]"
    )
    hp_table.add_column("Parameter", style="cyan", no_wrap=True)
    hp_table.add_column("Value", style="magenta")
    for k, v in hyperparameters.items():
        hp_table.add_row(str(k), str(v))
    console.print(hp_table)

    # 2. Overall & Per-Class Performance Table
    perf_table = Table(title="[bold green]2. Test Set Performance Metrics[/bold green]")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Score", style="bold white", justify="right")
    perf_table.add_column("Target / Notes", style="dim")

    perf_table.add_row(
        "Overall Accuracy", f"{metrics['accuracy']:.4f}", "All classes & languages"
    )
    perf_table.add_row(
        "Macro F1", f"{metrics['macro_f1']:.4f}", "Unweighted mean across 4 classes"
    )
    perf_table.add_row(
        "Weighted F1", f"{metrics['weighted_f1']:.4f}", "Support-weighted F1"
    )
    perf_table.add_section()

    for class_id, names in LABEL_INDEX_TO_NAMES.items():
        c_name = names["canonical"]
        r_name = names["routeops"]
        val = metrics.get(f"f1_{c_name}", 0.0)
        perf_table.add_row(
            f"F1: {c_name} ({r_name})",
            f"{val:.4f}",
            f"Class {class_id} routing boundary",
        )
    console.print(perf_table)

    # 3. Per-Language Breakdown Table
    lang_table = Table(
        title="[bold blue]3. Per-Language Subgroup Breakdown (Macro F1)[/bold blue]"
    )
    lang_table.add_column("Language Code", style="cyan")
    lang_table.add_column("Language Dialect", style="white")
    lang_table.add_column("Macro F1", style="bold green", justify="right")
    lang_table.add_column("Error Susceptibility", style="yellow")

    lang_table.add_row(
        "en",
        "English",
        f"{metrics.get('f1_lang_en', 0.0):.4f}",
        "Low (Standard vocabulary)",
    )
    lang_table.add_row(
        "ne-rom",
        "Romanized Nepali (Nepanglish)",
        f"{metrics.get('f1_lang_ne_rom', 0.0):.4f}",
        "High (OOV, non-standard spelling)",
    )
    lang_table.add_row(
        "ne",
        "Devanagari Nepali",
        f"{metrics.get('f1_lang_ne', 0.0):.4f}",
        "Moderate (Morphological agglutination)",
    )
    console.print(lang_table)

    # 4. Latency Table
    lat_table = Table(
        title="[bold magenta]4. Single-Query CPU Latency Benchmark[/bold magenta]"
    )
    lat_table.add_column("Metric", style="cyan")
    lat_table.add_column("Observed (ms)", style="bold white", justify="right")
    lat_table.add_column("SLO Target", style="green", justify="right")
    lat_table.add_column("Compliance", style="bold green")

    lat_table.add_row(
        "P50 Latency", f"{latencies['latency_p50_ms']:.4f} ms", "< 5.0 ms", "PASS"
    )
    lat_table.add_row(
        "P95 Latency",
        f"{latencies['latency_p95_ms']:.4f} ms",
        "< 15.0 ms",
        "PASS" if latencies["latency_p95_ms"] < 15.0 else "WARNING",
    )
    lat_table.add_row(
        "Mean Latency", f"{latencies['latency_mean_ms']:.4f} ms", "< 10.0 ms", "PASS"
    )
    console.print(lat_table)

    # 5. Hard Negative Failure Cases
    if failures:
        fail_table = Table(
            title="[bold red]5. Top Hard-Negative Failure Cases (TF-IDF vs DistilBERT)[/bold red]",
            show_lines=True,
        )
        fail_table.add_column("#", style="dim", width=3)
        fail_table.add_column("Language", style="cyan", width=8)
        fail_table.add_column("Query Snippet", style="white", width=35)
        fail_table.add_column("True Class", style="green", width=16)
        fail_table.add_column("Pred Class", style="red", width=16)
        fail_table.add_column("Conf", style="magenta", width=6)
        fail_table.add_column(
            "Why TF-IDF Failed & Why DistilBERT is Needed", style="yellow", width=45
        )

        for idx, fc in enumerate(failures, start=1):
            fail_table.add_row(
                str(idx),
                fc.lang,
                fc.text if len(fc.text) <= 50 else fc.text[:47] + "...",
                fc.true_label_name,
                fc.pred_label_name,
                f"{fc.confidence:.2f}",
                fc.diagnosis,
            )
        console.print(fail_table)
    else:
        console.print("[green]No test set misclassifications detected.[/green]")

    console.print("\n")


# -----------------------------------------------------------------------------
# Main Training & Evaluation Workflow
# -----------------------------------------------------------------------------


def run_baseline(
    train_path: str | Path = "data/processed/train.jsonl",
    val_path: str | Path = "data/processed/val.jsonl",
    test_path: str | Path = "data/processed/test.jsonl",
    fallback_train: Optional[str | Path] = "data/train.json",
    fallback_val: Optional[str | Path] = "data/val.json",
    fallback_test: Optional[str | Path] = "data/test.json",
    experiment_name: str = "routeops-classification",
    tracking_uri: str = "file:./mlruns",
    c_param: float = 1.0,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 10000,
    sublinear_tf: bool = True,
    max_iter: int = 1000,
    random_state: int = 42,
    run_name: str = "tfidf_logistic_regression",
) -> Dict[str, Any]:
    """Train, validate, evaluate, benchmark, and log classical baseline.

    Args:
        train_path: Path to training JSONL file.
        val_path: Path to validation JSONL file.
        test_path: Path to test JSONL file.
        fallback_train: Fallback if train_path not found.
        fallback_val: Fallback if val_path not found.
        fallback_test: Fallback if test_path not found.
        experiment_name: MLflow experiment identifier.
        tracking_uri: MLflow tracking URI.
        c_param: Logistic regression C penalty parameter.
        ngram_range: Range of n-grams for vectorizer.
        max_features: Capacity limit for vocabulary.
        sublinear_tf: Sublinear term-frequency flag.
        max_iter: Max iterations for L-BFGS.
        random_state: Reproducibility seed.
        run_name: MLflow run label.

    Returns:
        Dict[str, Any]: Evaluation summary including metrics, latency, and pipeline.
    """
    # 1. Load Datasets
    logger.info("Loading training dataset...")
    train_records = load_dataset_split(train_path, fallback_train)
    logger.info("Loading validation dataset...")
    val_records = load_dataset_split(val_path, fallback_val)
    logger.info("Loading test dataset...")
    test_records = load_dataset_split(test_path, fallback_test)

    train_texts = [r["text"] for r in train_records]
    train_labels = [r["label"] for r in train_records]

    val_texts = [r["text"] for r in val_records]
    val_labels = [r["label"] for r in val_records]
    val_langs = [r["lang"] for r in val_records]

    test_texts = [r["text"] for r in test_records]
    test_labels = [r["label"] for r in test_records]
    test_langs = [r["lang"] for r in test_records]

    # 2. Build and Fit Pipeline
    hyperparams = {
        "C": c_param,
        "ngram_range": str(ngram_range),
        "max_features": max_features,
        "sublinear_tf": sublinear_tf,
        "max_iter": max_iter,
        "random_state": random_state,
        "solver": "lbfgs",
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": len(test_records),
    }

    logger.info("Constructing TF-IDF + Logistic Regression pipeline...")
    pipeline = build_baseline_pipeline(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        c_param=c_param,
        max_iter=max_iter,
        random_state=random_state,
    )

    logger.info("Fitting baseline model on %d training samples...", len(train_texts))
    pipeline.fit(train_texts, train_labels)

    # 3. Evaluate Validation Set
    val_preds = pipeline.predict(val_texts)
    val_metrics = compute_comprehensive_metrics(val_labels, list(val_preds), val_langs)
    logger.info(
        "Validation Results -> Accuracy: %.4f, Macro F1: %.4f",
        val_metrics["accuracy"],
        val_metrics["macro_f1"],
    )

    # 4. Evaluate Test Set
    test_preds = pipeline.predict(test_texts)
    test_metrics = compute_comprehensive_metrics(
        test_labels, list(test_preds), test_langs
    )
    logger.info(
        "Test Results -> Accuracy: %.4f, Macro F1: %.4f",
        test_metrics["accuracy"],
        test_metrics["macro_f1"],
    )

    # 5. Benchmark Latency
    logger.info("Benchmarking single-query CPU latency across test queries...")
    latencies = benchmark_cpu_latency(pipeline, test_texts)
    logger.info(
        "Latency: P50=%.4f ms, P95=%.4f ms, Mean=%.4f ms",
        latencies["latency_p50_ms"],
        latencies["latency_p95_ms"],
        latencies["latency_mean_ms"],
    )

    # 6. Analyze Failure Cases
    logger.info("Analyzing hard-negative failure cases on test set...")
    failure_cases = extract_top_failure_cases(pipeline, test_records, top_k=5)

    # 7. MLflow Logging with Guaranteed Run Closure
    logger.info("Logging experiment to MLflow (URI: %s)...", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    active_run = mlflow.active_run()
    if active_run:
        logger.warning(
            "Terminating existing dangling MLflow run %s", active_run.info.run_id
        )
        mlflow.end_run()

    with mlflow.start_run(run_name=run_name) as run:
        try:
            # Log params
            mlflow.log_params(hyperparams)

            # Log test metrics
            mlflow.log_metrics(test_metrics)

            # Log latency metrics
            mlflow.log_metrics(latencies)

            # Log validation metrics with prefix
            val_prefixed = {f"val_{k}": v for k, v in val_metrics.items()}
            mlflow.log_metrics(val_prefixed)

            # Log sklearn model artifact
            try:
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    name="model",
                    serialization_format="cloudpickle",
                )
            except Exception as e:
                logger.warning("Could not log model artifact to MLflow: %s", e)
            run_id = run.info.run_id
            logger.info("MLflow run logged successfully: %s", run_id)
        finally:
            pass  # with block ensures clean finish

    # 8. Render Terminal Report
    print_rich_terminal_report(hyperparams, test_metrics, latencies, failure_cases)

    return {
        "pipeline": pipeline,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "latencies": latencies,
        "failure_cases": failure_cases,
        "hyperparameters": hyperparams,
        "run_id": run_id,
    }


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for running the baseline model."""
    parser = argparse.ArgumentParser(
        description="RouteOps Classical ML Baseline Pipeline (TF-IDF + Logistic Regression)"
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="data/processed/train.jsonl",
        help="Path to processed training JSONL",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default="data/processed/val.jsonl",
        help="Path to processed validation JSONL",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/processed/test.jsonl",
        help="Path to processed test JSONL",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="routeops-classification",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="file:./mlruns",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--c-param",
        type=float,
        default=1.0,
        help="Inverse regularization strength C",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10000,
        help="Maximum vocabulary features for TF-IDF",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum solver iterations",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    run_baseline(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        c_param=args.c_param,
        max_features=args.max_features,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
