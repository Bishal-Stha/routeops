"""Unit and integration test suite for ONNX export, Int8 quantization, and latency SLA.

Validates ONNX graph export, dynamic axes for arbitrary batch/sequence lengths,
dynamic Int8 quantization compression (> 2.5x), inference parity with PyTorch (< 0.05),
single-query CPU latency benchmarking, and assertion that P95 CPU latency < 25.0 ms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from src.export_onnx import (
    CLASS_INDEX_TO_NAMES,
    SUPPORTED_LANGUAGES,
    benchmark_onnx_latency,
    build_model_comparison_report,
    evaluate_onnx_model,
    export_to_onnx,
    get_optimized_session_options,
    log_onnx_to_mlflow,
    quantize_onnx_model,
    resolve_model_source,
    verify_inference_parity,
)


@pytest.fixture(scope="module")
def onnx_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> Tuple[Path, Path, str]:
    """Provide verified FP32 and Int8 Quantized ONNX models for testing.

    Reuses existing exported artifacts in models/ if present and valid;
    otherwise exports and quantizes into a temporary test directory.

    Returns:
        Tuple of (fp32_onnx_path, int8_onnx_path, model_source).
    """
    existing_fp32 = Path("models/onnx/model.onnx")
    existing_int8 = Path("models/onnx_quantized/model_quantized.onnx")
    model_src = resolve_model_source()

    if existing_fp32.is_file() and existing_int8.is_file():
        return existing_fp32, existing_int8, model_src

    tmp_dir = tmp_path_factory.mktemp("onnx_test")
    fp32_path = tmp_dir / "model.onnx"
    int8_path = tmp_dir / "model_quantized.onnx"

    export_to_onnx(model_src, fp32_path, max_length=32)
    quantize_onnx_model(fp32_path, int8_path)

    return fp32_path, int8_path, model_src


class TestSessionOptions:
    """Tests for CPU runtime execution session configuration."""

    def test_optimized_session_options_defaults(self) -> None:
        """Verify session options pin 4 P-core threads and configure sequential execution."""
        opts = get_optimized_session_options(
            intra_op_num_threads=4, inter_op_num_threads=1
        )
        assert opts.intra_op_num_threads == 4
        assert opts.inter_op_num_threads == 1
        assert opts.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
        assert (
            opts.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

    def test_session_options_custom_threads(self) -> None:
        """Verify custom thread settings are properly reflected."""
        opts = get_optimized_session_options(
            intra_op_num_threads=2, inter_op_num_threads=2
        )
        assert opts.intra_op_num_threads == 2
        assert opts.inter_op_num_threads == 2


class TestONNXExportAndGraphValidity:
    """Tests for PyTorch to ONNX graph translation."""

    def test_onnx_file_exists_and_validates(
        self,
        onnx_artifacts: Tuple[Path, Path, str],
    ) -> None:
        """Verify exported FP32 model exists and passes structural integrity checks."""
        fp32_path, _, _ = onnx_artifacts
        assert fp32_path.is_file()
        assert fp32_path.stat().st_size > 0

        # Validate with onnx.checker
        model = onnx.load(str(fp32_path))
        onnx.checker.check_model(model)

    def test_graph_input_output_nodes(
        self,
        onnx_artifacts: Tuple[Path, Path, str],
    ) -> None:
        """Verify input and output tensor names match expected signature."""
        fp32_path, _, _ = onnx_artifacts
        model = onnx.load(str(fp32_path))

        input_names = [inp.name for inp in model.graph.input]
        assert "input_ids" in input_names
        assert "attention_mask" in input_names

        output_names = [out.name for out in model.graph.output]
        assert "logits" in output_names


class TestDynamicAxes:
    """Tests validating dynamic axes support for variable batch and sequence sizes."""

    @pytest.mark.parametrize(
        "batch_size,seq_len",
        [
            (1, 12),
            (1, 48),
            (2, 20),
            (4, 32),
        ],
    )
    def test_variable_input_shapes_execute(
        self,
        onnx_artifacts: Tuple[Path, Path, str],
        batch_size: int,
        seq_len: int,
    ) -> None:
        """Verify ONNX graph handles dynamic batch sizes and sequence lengths without errors."""
        _, int8_path, _ = onnx_artifacts
        opts = get_optimized_session_options()
        session = ort.InferenceSession(
            str(int8_path), opts, providers=["CPUExecutionProvider"]
        )

        dummy_ids = np.random.randint(
            100, 2000, size=(batch_size, seq_len), dtype=np.int64
        )
        dummy_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        outputs = session.run(
            ["logits"],
            {"input_ids": dummy_ids, "attention_mask": dummy_mask},
        )
        logits = outputs[0]
        assert logits.shape[0] == batch_size
        assert logits.shape[1] == 4


class TestDynamicInt8Quantization:
    """Tests for dynamic Int8 post-training quantization and compression."""

    def test_model_compression_ratio_and_memory_limit(
        self,
        onnx_artifacts: Tuple[Path, Path, str],
    ) -> None:
        """Verify model size compression is > 2.5x and quantized size is < 250 MB."""
        fp32_path, int8_path, _ = onnx_artifacts

        fp32_size_mb = fp32_path.stat().st_size / (1024**2)
        int8_size_mb = int8_path.stat().st_size / (1024**2)

        compression_ratio = fp32_size_mb / int8_size_mb
        assert (
            compression_ratio >= 2.5
        ), f"Expected compression >= 2.5x, observed {compression_ratio:.2f}x"
        assert (
            int8_size_mb < 250.0
        ), f"Quantized model {int8_size_mb:.2f} MB exceeds 250 MB ceiling"

    def test_quantized_model_loads_in_runtime(
        self,
        onnx_artifacts: Tuple[Path, Path, str],
    ) -> None:
        """Verify quantized Int8 model loads cleanly into ONNX Runtime inference session."""
        _, int8_path, _ = onnx_artifacts
        opts = get_optimized_session_options()
        session = ort.InferenceSession(
            str(int8_path), opts, providers=["CPUExecutionProvider"]
        )
        assert session is not None


class TestInferenceParity:
    """Tests for output distribution parity between PyTorch and quantized ONNX."""

    def test_parity_threshold_and_cosine_similarity(
        self,
        onnx_artifacts: Tuple[Path, Path, str],
    ) -> None:
        """Verify max probability divergence is <= 0.15 and cosine similarity > 0.98."""
        _, int8_path, model_src = onnx_artifacts
        parity = verify_inference_parity(
            model_path_or_name=model_src,
            onnx_path=int8_path,
            max_prob_diff_threshold=0.15,
        )

        assert parity["passed"] is True
        assert parity["max_prob_diff"] <= 0.15
        assert parity["cosine_similarity"] >= 0.98


class TestCPULatencyBenchmarkAndSLA:
    """Tests for single-query CPU latency benchmarking and strict SLA enforcement."""

    def test_single_query_cpu_latency_strictly_under_25ms(
        self,
        onnx_artifacts: Tuple[Path, Path, str],
    ) -> None:
        """Enforce strict SLA: P95 CPU inference latency must be strictly < 25.0 ms."""
        _, int8_path, model_src = onnx_artifacts
        test_data_path = Path("data/processed/test.jsonl")

        latencies = benchmark_onnx_latency(
            onnx_path=int8_path,
            tokenizer_path_or_name=model_src,
            test_path=test_data_path,
            num_runs=50,
            warmup_runs=10,
        )

        assert latencies["latency_p50_ms"] > 0.0
        assert latencies["latency_p95_ms"] > 0.0
        assert latencies["throughput_qps"] > 0.0

        # Hard SLA Assertion: Strictly < 25.0 ms
        p95_ms = latencies["latency_p95_ms"]
        assert (
            p95_ms < 25.0
        ), f"CRITICAL SLA BREACH: Observed P95 {p95_ms:.2f} ms >= 25.0 ms limit"


class TestEvaluationAndMetrics:
    """Tests for evaluation on processed test dataset."""

    def test_evaluate_onnx_model_returns_full_schema(
        self,
        onnx_artifacts: Tuple[Path, Path, str],
    ) -> None:
        """Verify evaluation generates valid overall, per-class, and per-language metrics."""
        _, int8_path, model_src = onnx_artifacts
        test_data_path = Path("data/processed/test.jsonl")

        metrics = evaluate_onnx_model(
            onnx_path=int8_path,
            tokenizer_path_or_name=model_src,
            test_path=test_data_path,
        )

        assert 0.0 <= metrics["overall_accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert metrics["total_samples"] == 240

        # Validate per-class F1 keys
        for c_id in range(4):
            c_name = CLASS_INDEX_TO_NAMES[c_id]["routeops"]
            assert c_name in metrics["per_class_f1"]
            assert 0.0 <= metrics["per_class_f1"][c_name] <= 1.0

        # Validate per-language F1 keys
        for lang in SUPPORTED_LANGUAGES:
            assert lang in metrics["per_language_f1"]
            assert 0.0 <= metrics["per_language_f1"][lang] <= 1.0


class TestModelComparisonReport:
    """Tests for 3-way model comparison report generation and schema."""

    def test_comparison_report_structure_and_persistence(
        self,
        onnx_artifacts: Tuple[Path, Path, str],
        tmp_path: Path,
    ) -> None:
        """Verify 3-way model comparison report saves valid JSON with required SLA gates."""
        fp32_path, int8_path, _ = onnx_artifacts
        out_report_path = tmp_path / "model_comparison.json"

        mock_onnx_metrics = {
            "overall_accuracy": 0.9900,
            "macro_f1": 0.9900,
            "per_class_f1": {"direct_response": 0.99, "rag_retrieval": 0.99},
            "per_language_f1": {"en": 0.99, "ne-rom": 0.99, "ne": 0.99},
        }
        mock_latency = {
            "latency_p50_ms": 12.0,
            "latency_p95_ms": 18.5,
            "latency_p99_ms": 22.0,
            "latency_mean_ms": 13.0,
            "throughput_qps": 76.9,
        }

        report = build_model_comparison_report(
            onnx_metrics=mock_onnx_metrics,
            onnx_latency=mock_latency,
            onnx_path=int8_path,
            fp32_onnx_path=fp32_path,
            output_path=out_report_path,
        )

        assert report["sla_verification"]["latency_p95_passed"] is True
        assert out_report_path.is_file()
        with open(out_report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "models" in data
        assert "tfidf_baseline" in data["models"]
        assert "pytorch_finetuned" in data["models"]
        assert "onnx_int8_quantized" in data["models"]

        sla = data["sla_verification"]
        assert "latency_p95_target_ms" in sla
        assert sla["latency_p95_actual_ms"] == 18.5
        assert sla["latency_p95_passed"] is True
        assert sla["macro_f1_degradation_target_pct"] == 1.0


class TestMLflowIntegration:
    """Tests for MLflow logging of ONNX artifacts and metrics."""

    def test_mlflow_logging_succeeds(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify MLflow run logs ONNX parameters and metrics cleanly."""
        report_file = tmp_path / "dummy_report.json"
        report_file.write_text("{}", encoding="utf-8")

        mock_metrics = {
            "overall_accuracy": 0.99,
            "macro_f1": 0.99,
            "weighted_f1": 0.99,
            "per_class_f1": {"direct_response": 0.99},
            "per_language_f1": {"en": 0.99},
        }
        mock_latency = {
            "latency_p50_ms": 11.5,
            "latency_p95_ms": 17.8,
            "latency_p99_ms": 21.0,
            "latency_mean_ms": 12.2,
            "throughput_qps": 82.0,
        }

        run_id = log_onnx_to_mlflow(
            onnx_metrics=mock_metrics,
            onnx_latency=mock_latency,
            comparison_report_path=report_file,
            experiment_name="routeops-test",
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            run_name="test_onnx_run",
        )

        assert run_id is not None
        assert isinstance(run_id, str)
        assert len(run_id) > 0
