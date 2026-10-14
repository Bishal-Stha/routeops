# RouteOps - Multilingual Semantic Router & MLOps Gateway

RouteOps is a high-performance, cost-aware semantic router that classifies incoming user queries (English, Romanized Nepali `ne-rom`, and Devanagari Nepali `ne`) into optimal LLM tiers or execution paths: Direct Response, RAG Retrieval, Frontier Reasoning, or Guardrail Blocked.

---

## 1. Hardware Envelope & Constraints

All engineering, training, and benchmarking in RouteOps must strictly respect the edge host hardware envelope:

- **CPU**: Intel Core i5-12450H (8 cores / 12 threads)
- **System Memory**: 16 GB System RAM
- **GPU**: NVIDIA GeForce RTX 2050 Laptop GPU
  - **Total Dedicated VRAM**: 4096 MiB (4.0 GB)
  - **Operating Driver/CUDA**: Driver 595.x / CUDA 12.x-13.x

### Training Envelope Rules:
- **Maximum VRAM Budget**: Strictly capped at **3.0 GB (3072 MiB)** during model fine-tuning.
- **Precision**: `fp16=True` (or bf16 where supported, fp16 on RTX 2050)
- **Sequence Length**: `max_length=128` (optimal semantic length for prompt routing)
- **Batch Size**: `per_device_train_batch_size=4`
- **Gradient Accumulation**: `gradient_accumulation_steps=8` (effective batch size = 32)
- **Memory Safety**: Clean PyTorch CUDA cache between runs (`torch.cuda.empty_cache()`), dynamic memory tracking, fail-safe early warnings if VRAM approaches 3.0 GB.

### Inference & Serving Envelope:
- **Runtime**: ONNX Runtime with Int8 Dynamic Quantization on CPU/GPU.
- **Latency Target**: P95 < 15ms per routing classification.
- **Memory Footprint**: < 250 MB RAM for quantized ONNX runtime inference session.

---

## 2. Architecture & Routing Categories

RouteOps maps queries to 4 distinct execution classes:

| Class ID | Label | Intent / Target | Fallback & Cost Model |
| :--- | :--- | :--- | :--- |
| `0` | `direct_response` | Chit-chat, deterministic FAQs, greetings, simple responses | Local template / small model ($0.0001 / 1k tokens) |
| `1` | `rag_retrieval` | Domain knowledge, banking/civic Q&A, documentation | Vector search + Mid-tier LLM ($0.0015 / 1k tokens) |
| `2` | `frontier_reasoning` | Multi-step reasoning, coding, mathematical logic | Frontier LLM (Claude 3.5 Sonnet / GPT-4o) ($0.0300 / 1k tokens) |
| `3` | `guardrail_blocked` | Prompt injection, jailbreak attempts, toxic/offensive inputs | Immediate block / rejection ($0.0000 / zero LLM cost) |

---

## 3. Engineering & Portfolio Standards

- **Clean Architecture & Separation of Concerns**: Core engine logic (`router_service.py`), training pipeline (`train.py`), monitoring (`drift_monitor.py`), configuration management (`config.py`), and REST API (`api.py`) are strictly decoupled.
- **Strict Typing**: Python type hints (`mypy` compliant) throughout. Pydantic models for configuration, request/response validation, and metric schemas.
- **Modularity & Testability**: Every component must be unit-testable without external network dependencies (mock external calls where appropriate).
- **Docstrings & Google Style**: Every class, function, and module must contain clear docstrings with Args, Returns, and descriptive context.
- **Zero Secrets Policy**: Never commit `.env`, API keys, tokens, or credentials. Use environment variable loading with safe fallbacks.

---

## 4. Conventional Commit Standards & Git Shipping

All commits must follow the **Conventional Commits** specification with frequent, atomic git commits:
- `feat:` New features or capabilities
- `perf:` Performance optimizations (latency, ONNX quantization, memory reduction)
- `test:` Unit tests, integration tests, smoke benchmarks
- `chore:` Dependency updates, Makefile recipes, CI configuration
- `fix:` Bug fixes and error handling improvements
- `docs:` Documentation, README, architectural diagrams

### @git-shipper Rules:
- Before committing, verify zero secrets, credentials, or binary models are staged.
- Ensure all tests pass (`make test` or `pytest`).
- Maintain atomic, descriptive commit messages with conventional commit prefixes.
- **Push Policy**: Always immediately push every verified commit to `origin main` (`git push origin main`) to keep the public engineering portfolio continuously synchronized.

---

## 5. Development & Testing Conventions

```bash
# Code formatting and linting
make format
make lint

# Run unit tests with coverage
make test

# Run 4GB VRAM safety smoke test
make smoke-test

# Fine-tune multilingual classifier within 3GB VRAM budget
make train

# Export model to ONNX and quantize to Int8
make export-onnx

# Run latency & throughput benchmarks
make benchmark
```

---

## 6. Team Division of Ownership

- **Teammate 1 (`infra-architect`)**:
  - `CLAUDE.md`, `configs/config.yaml`, `requirements.txt`, `Makefile`, `README.md`, `.claude/agents/git-shipper.md`
  - Core codebase: `src/__init__.py`, `src/config.py`, `src/data_gen.py`, `src/train.py`, `src/export_onnx.py`, `src/router_service.py`, `src/drift_monitor.py`, `src/api.py`
- **Teammate 2 (`mlops-benchmarker`)**:
  - Smoke tests & Benchmarking suite: `src/smoke_test_4gb.py`, `tests/test_smoke.py`
  - DO NOT modify files owned by `mlops-benchmarker`.
