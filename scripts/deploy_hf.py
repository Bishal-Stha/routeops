#!/usr/bin/env python3
"""Helper script to deploy RouteOps model artifacts to Hugging Face Hub.

Uploads quantized ONNX models and tokenizer configuration to a Hugging Face Space
or Model repository, allowing zero-cost cloud deployments on Hugging Face Spaces.

Usage:
    export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
    python scripts/deploy_hf.py --repo-id <username>/routeops-model
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("routeops.deploy_hf")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deploy RouteOps quantized ONNX models and artifacts to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=os.getenv("HF_MODEL_REPO", ""),
        help="Hugging Face Model repository ID (e.g. 'username/routeops-model').",
    )
    parser.add_argument(
        "--space-id",
        type=str,
        default=os.getenv("HF_SPACE_ID", ""),
        help="Optional Hugging Face Space repository ID for Docker deploy.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN", ""),
        help="Hugging Face API token (or read from HF_TOKEN env var).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/onnx_quantized",
        help="Directory containing the quantized ONNX model.",
    )
    return parser.parse_args()


def upload_model_artifacts(repo_id: str, model_dir: Path, token: str) -> None:
    """Upload ONNX model artifacts to Hugging Face model hub."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        )
        sys.exit(1)

    api = HfApi(token=token)

    logger.info("Verifying or creating Hugging Face model repo: %s", repo_id)
    create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)

    quantized_path = model_dir / "model_quantized.onnx"
    if not quantized_path.is_file():
        # Fallback check
        quantized_path = Path("models/onnx/model_quantized.onnx")

    if quantized_path.is_file():
        logger.info(
            "Uploading quantized ONNX model (%s, %.1f MB) to %s...",
            quantized_path.name,
            quantized_path.stat().st_size / (1024 * 1024),
            repo_id,
        )
        api.upload_file(
            path_or_fileobj=str(quantized_path),
            path_in_repo="model_quantized.onnx",
            repo_id=repo_id,
            repo_type="model",
            commit_message="feat: upload Int8 quantized ONNX router model",
        )
        logger.info("Successfully uploaded model_quantized.onnx!")
    else:
        logger.warning(
            "Quantized model not found at %s. Please run 'make export-onnx' first.",
            model_dir,
        )


def main() -> None:
    """Main deployment orchestration."""
    args = parse_args()

    if not args.token:
        logger.error("No HF_TOKEN provided. Please export HF_TOKEN or pass --token.")
        sys.exit(1)

    if not args.repo_id:
        logger.error(
            "No --repo-id provided. Specify destination Hugging Face model repository."
        )
        sys.exit(1)

    model_dir = Path(args.model_dir)
    upload_model_artifacts(args.repo_id, model_dir, args.token)

    logger.info("=" * 60)
    logger.info("RouteOps Model Deployment Completed Successfully!")
    logger.info("To connect your live backend on Hugging Face Spaces / Docker:")
    logger.info("  Set Environment Variable: HF_MODEL_REPO=%s", args.repo_id)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
