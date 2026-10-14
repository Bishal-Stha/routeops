"""Configuration management module for RouteOps.

Provides typed Pydantic models for system parameters, hardware constraints,
model fine-tuning hyperparameters, routing thresholds, and monitoring thresholds.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """Project-level metadata."""

    name: str = "RouteOps"
    version: str = "0.1.0"
    description: str = "Multilingual Semantic Router"


class HardwareConfig(BaseModel):
    """Edge workstation hardware constraints and training budget."""

    device: str = "cuda"
    vram_budget_mb: int = 3072  # 3.0 GB ceiling on RTX 2050 (4 GB)
    cpu_threads: int = 8
    fp16_enabled: bool = True


class ModelConfig(BaseModel):
    """Model architecture and checkpoint locations."""

    name: str = "distilbert-base-multilingual-cased"
    tokenizer_name: str = "distilbert-base-multilingual-cased"
    max_length: int = 128
    num_labels: int = 4
    output_dir: str = "./models/routeops-distilbert"
    onnx_output_path: str = "./models/onnx/model.onnx"
    quantized_onnx_path: str = "./models/onnx/model_quantized.onnx"


class TrainingConfig(BaseModel):
    """Fine-tuning hyperparameters strictly bounded by the 3GB VRAM envelope."""

    learning_rate: float = 3.0e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    seed: int = 42


class LanguageInfo(BaseModel):
    """Language descriptor for supported dialects."""

    code: str
    name: str


class RoutingConfig(BaseModel):
    """Routing label mapping, confidence thresholds, and token economics."""

    labels: Dict[int, str] = Field(
        default_factory=lambda: {
            0: "direct_response",
            1: "rag_retrieval",
            2: "frontier_reasoning",
            3: "guardrail_blocked",
        }
    )
    descriptions: Dict[str, str] = Field(
        default_factory=lambda: {
            "direct_response": "Simple chit-chat, greetings, deterministic queries",
            "rag_retrieval": "Domain knowledge, Nepali banking/civic Q&A, documentation",
            "frontier_reasoning": "Complex logic, coding, mathematical, multi-step problem solving",
            "guardrail_blocked": "Prompt injection, jailbreak attempts, profanity, toxic queries",
        }
    )
    thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "direct_response": 0.80,
            "rag_retrieval": 0.75,
            "frontier_reasoning": 0.70,
            "guardrail_flag": 0.60,
        }
    )
    token_costs_per_1k: Dict[str, float] = Field(
        default_factory=lambda: {
            "direct_response": 0.0001,
            "rag_retrieval": 0.0015,
            "frontier_reasoning": 0.0300,
            "guardrail_blocked": 0.0000,
        }
    )
    supported_languages: List[LanguageInfo] = Field(
        default_factory=lambda: [
            LanguageInfo(code="en", name="English"),
            LanguageInfo(code="ne-rom", name="Romanized Nepali"),
            LanguageInfo(code="ne", name="Devanagari Nepali"),
        ]
    )

    def get_label_name(self, index: int) -> str:
        """Return the label string for an integer class index."""
        return self.labels.get(index, "frontier_reasoning")

    def get_label_id(self, name: str) -> int:
        """Return the integer index for a label string."""
        for idx, lbl in self.labels.items():
            if lbl == name:
                return idx
        return 2  # Default to frontier reasoning on mismatch


class MonitoringConfig(BaseModel):
    """Drift detection and Prometheus metrics configuration."""

    prometheus_port: int = 8000
    drift_window_size: int = 1000
    entropy_threshold: float = 1.2
    latency_slo_ms: float = 15.0


class GatewayConfig(BaseModel):
    """FastAPI REST Gateway server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


class RouteOpsConfig(BaseModel):
    """Root configuration object representing the full system specification."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)


def load_config(config_path: Optional[str | Path] = None) -> RouteOpsConfig:
    """Load configuration from a YAML file, falling back to defaults.

    Args:
        config_path: Path to the YAML configuration file. If None, checks the
                     ROUTEOPS_CONFIG environment variable or defaults to
                     configs/config.yaml.

    Returns:
        RouteOpsConfig: Validated typed configuration instance.
    """
    if config_path is None:
        env_path = os.getenv("ROUTEOPS_CONFIG")
        if env_path and os.path.isfile(env_path):
            config_path = env_path
        else:
            default_path = (
                Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            )
            if default_path.is_file():
                config_path = default_path

    if config_path and Path(config_path).is_file():
        with open(config_path, "r", encoding="utf-8") as file:
            raw_data: Dict[str, Any] = yaml.safe_load(file) or {}
        return RouteOpsConfig(**raw_data)

    return RouteOpsConfig()
