"""Fine-tuning pipeline for multilingual semantic routing under strict 3GB VRAM budget.

Uses HuggingFace Transformers & Datasets with PyTorch FP16 mixed precision,
per-device micro-batching (batch_size=4), gradient accumulation (steps=8),
and active CUDA peak memory tracking to protect the 4GB RTX 2050 envelope.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.config import RouteOpsConfig, load_config
from src.data_gen import create_splits, generate_synthetic_samples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("routeops.train")


class VRAMMonitorCallback(TrainerCallback):
    """Callback to track peak CUDA VRAM consumption and enforce the 3GB budget."""

    def __init__(self, vram_budget_mb: int = 3072) -> None:
        super().__init__()
        self.vram_budget_mb = vram_budget_mb
        self.peak_vram_recorded_mb: float = 0.0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Inspect CUDA memory after each optimizer/accumulation step."""
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            self.peak_vram_recorded_mb = max(self.peak_vram_recorded_mb, peak_mb)

            if peak_mb > self.vram_budget_mb:
                logger.error(
                    f"CRITICAL VRAM BUDGET EXCEEDED: Peak {peak_mb:.1f} MB > "
                    f"Budget {self.vram_budget_mb} MB! Aborting training."
                )
                control.should_training_stop = True

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Log VRAM summary at epoch boundary."""
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            logger.info(
                f"[Epoch {state.epoch:.0f}] Peak VRAM: {peak_mb:.2f} MB / "
                f"{self.vram_budget_mb} MB ({ (peak_mb / self.vram_budget_mb) * 100:.1f}%)"
            )


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """Compute classification metrics for evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    recall_macro = recall_score(labels, preds, average="macro", zero_division=0)

    return {
        "accuracy": float(acc),
        "f1": float(f1_macro),
        "precision": float(precision_macro),
        "recall": float(recall_macro),
    }


def prepare_datasets(
    config: RouteOpsConfig,
    data_dir: str = "./data",
) -> DatasetDict:
    """Load or generate tokenized datasets for training and evaluation.

    Args:
        config: System configuration object.
        data_dir: Directory where data is cached.

    Returns:
        DatasetDict containing train, val, and test splits.
    """
    train_json = Path(data_dir) / "train.json"
    val_json = Path(data_dir) / "val.json"

    if train_json.is_file() and val_json.is_file():
        import json

        with open(train_json, "r", encoding="utf-8") as f:
            train_raw = json.load(f)
        with open(val_json, "r", encoding="utf-8") as f:
            val_raw = json.load(f)
        test_json = Path(data_dir) / "test.json"
        if test_json.is_file():
            with open(test_json, "r", encoding="utf-8") as f:
                test_raw = json.load(f)
        else:
            test_raw = val_raw
    else:
        logger.info("[*] Generating synthetic datasets on the fly...")
        raw_samples = generate_synthetic_samples(
            samples_per_class=150, seed=config.training.seed
        )
        train_raw, val_raw, test_raw = create_splits(
            raw_samples, seed=config.training.seed
        )

    return DatasetDict(
        {
            "train": Dataset.from_list(train_raw),
            "validation": Dataset.from_list(val_raw),
            "test": Dataset.from_list(test_raw),
        }
    )


def train_model(config: RouteOpsConfig) -> str:
    """Execute model fine-tuning within the 3GB VRAM envelope.

    Args:
        config: System configuration specification.

    Returns:
        Path to the saved model directory.
    """
    logger.info("=" * 60)
    logger.info("Starting RouteOps Training Pipeline")
    logger.info(f"Target Architecture: {config.model.name}")
    logger.info(f"VRAM Budget Ceiling: {config.hardware.vram_budget_mb} MB")
    logger.info("=" * 60)

    # 1. Device and Hardware Verification
    use_cuda = torch.cuda.is_available() and config.hardware.device == "cuda"
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    logger.info(f"Training Device: {device_name} (CUDA Available: {use_cuda})")

    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        logger.info(f"Detected GPU Dedicated Memory: {total_vram_mb:.1f} MB")

    # 2. Tokenizer and Data Preparation
    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_name)
    raw_datasets = prepare_datasets(config)

    def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=config.model.max_length,
        )

    logger.info("[*] Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(tokenize_batch, batched=True)

    # 3. Model Initialization
    id2label = config.routing.labels
    label2id = {v: k for k, v in id2label.items()}

    logger.info(
        f"[*] Initializing model {config.model.name} with {len(id2label)} labels..."
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    # 4. Training Arguments configuration for RTX 2050 4GB (strict 3.0GB budget)
    output_dir = Path(config.model.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=config.training.num_train_epochs,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        fp16=config.training.fp16 and use_cuda,
        logging_steps=config.training.logging_steps,
        eval_strategy=config.training.eval_strategy,
        save_strategy=config.training.save_strategy,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        seed=config.training.seed,
        report_to="none",
        dataloader_num_workers=2 if use_cuda else 0,
        dataloader_pin_memory=use_cuda,
    )

    vram_callback = VRAMMonitorCallback(vram_budget_mb=config.hardware.vram_budget_mb)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[vram_callback],
    )

    # 5. Execute Training
    logger.info("[*] Executing fine-tuning loop...")
    _ = trainer.train()

    # 6. Evaluate Model
    logger.info("[*] Running final evaluation on validation set...")
    eval_metrics = trainer.evaluate()
    logger.info(f"[+] Validation Metrics: {eval_metrics}")

    # 7. Save Final Model Artifacts
    logger.info(f"[*] Saving best model checkpoint to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if use_cuda:
        final_peak = torch.cuda.max_memory_allocated() / (1024**2)
        logger.info(
            f"[+] Peak VRAM usage during training: {final_peak:.2f} MB "
            f"(Budget: {config.hardware.vram_budget_mb} MB)"
        )

    logger.info(
        f"[+] RouteOps model training complete. Artifacts saved in: {output_dir}"
    )
    return str(output_dir)


def main() -> None:
    """CLI Entry point for fine-tuning."""
    parser = argparse.ArgumentParser(
        description="RouteOps Multilingual Model Fine-Tuning"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_model(cfg)


if __name__ == "__main__":
    main()
