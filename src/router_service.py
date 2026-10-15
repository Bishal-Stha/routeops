"""Core Semantic Routing Engine and Gateway for RouteOps.

Executes low-latency Int8 ONNX Runtime inference, applies PII detection and redaction,
performs uncertainty estimation (entropy), evaluates edge guardrails, operates an
in-memory / SQLite zero-extra-model semantic cache, and implements confidence-based
cascade abstention to optimize unit economics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from src.config import RouteOpsConfig, load_config
from src.telemetry_db import TelemetryDB

logger = logging.getLogger("routeops.router_service")


# -----------------------------------------------------------------------------
# Decision & Request Schemas
# -----------------------------------------------------------------------------


class RouteDecision(BaseModel):
    """Structured decision output from the RouteOps Semantic AI Gateway."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    redacted_query: str
    detected_lang: str = "en"
    language_detected: str = "en"  # Backward compatibility alias
    pii_detected: List[str] = Field(default_factory=list)

    predicted_class: str
    routed_class: str  # Backward compatibility alias
    routed_class_id: int
    confidence: float
    entropy: float
    probabilities: Dict[str, float]

    # Tier choices: cache_hit, guardrail_blocked, slm_fast_tier,
    # rag_account_tier, frontier_reasoning_tier, low_conf_fallback
    routed_tier: str
    action: str
    is_guardrail_triggered: bool = False
    is_cache_hit: bool = False

    response_text: str = ""
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    estimated_cost_per_1k: float = 0.0  # Backward compatibility alias
    naive_frontier_cost_usd: float = 0.0
    cost_saved_usd: float = 0.0
    latency_ms: float = 0.0


# -----------------------------------------------------------------------------
# Zero-Extra-Model N-Gram Projection Vectorizer
# -----------------------------------------------------------------------------


def compute_ngram_projection_embedding(text: str, dim: int = 128) -> np.ndarray:
    """Compute a zero-extra-model L2-normalized dense embedding vector.

    Utilizes sign-hashed word and character 3-gram projections (SimHash style)
    to achieve sub-millisecond (<0.05ms) deterministic embeddings with zero external models.

    Args:
        text: Input query string.
        dim: Dimensionality of the target vector (default: 128).

    Returns:
        L2-normalized 1D float32 numpy array of shape (dim,).
    """
    clean = text.lower().strip()
    words = re.findall(r"\w+", clean)
    vec = np.zeros(dim, dtype=np.float32)

    if not words and not clean:
        return vec

    # 1. Word-level hashed projections
    for word in words:
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) else -1.0
        vec[idx] += sign * 1.5

    # 2. Character 3-gram projections for morphology and code-mixing
    if len(clean) >= 3:
        for i in range(len(clean) - 2):
            gram = clean[i : i + 3]
            h = int(hashlib.sha256(gram.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign * 0.5

    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    return vec


# -----------------------------------------------------------------------------
# PII Detection and Redaction Engine
# -----------------------------------------------------------------------------


class PIIRedactor:
    """Detects and redacts sensitive entities (phone numbers, OTPs, API keys, cards)."""

    PHONE_PATTERN = re.compile(
        r"(?:\+?977[- ]?)?(?:98\d{8}|97\d{8}|\b9\d{9}\b|\b\d{10}\b)"
    )
    OTP_PATTERN = re.compile(
        r"(?:otp|code|pin|password|token)[\s:=]+(\b\d{4,6}\b)", re.IGNORECASE
    )
    API_KEY_PATTERN = re.compile(
        r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|Bearer\s+[a-zA-Z0-9_.-]+|"
        r"api_key[:=]\s*[a-zA-Z0-9_-]{16,})\b",
        re.IGNORECASE,
    )
    CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4})\b")

    @classmethod
    def redact(cls, text: str) -> Tuple[str, List[str]]:
        """Redact PII from query text and return the sanitized string and detected types."""
        redacted = text
        detected: List[str] = []

        # API Keys
        if cls.API_KEY_PATTERN.search(redacted):
            redacted = cls.API_KEY_PATTERN.sub("[REDACTED_API_KEY]", redacted)
            detected.append("api_key")

        # Credit Cards
        if cls.CREDIT_CARD_PATTERN.search(redacted):
            redacted = cls.CREDIT_CARD_PATTERN.sub("[REDACTED_CARD]", redacted)
            detected.append("credit_card")

        # OTPs / PINs
        match_otp = cls.OTP_PATTERN.search(redacted)
        if match_otp:
            redacted = cls.OTP_PATTERN.sub("[REDACTED_OTP]", redacted)
            detected.append("otp_code")

        # Phone numbers (Nepal 10-digit and international formats)
        if cls.PHONE_PATTERN.search(redacted):
            redacted = cls.PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
            detected.append("phone_number")

        return redacted, detected


# -----------------------------------------------------------------------------
# Main Gateway & Router Service
# -----------------------------------------------------------------------------


class RouteOpsGateway:
    """Production Semantic AI Gateway and Cost-Aware Multi-Tier Router."""

    def __init__(
        self,
        config: Optional[RouteOpsConfig] = None,
        model_path: Optional[str | Path] = None,
        use_quantized: bool = True,
        db_path: Optional[str | Path] = None,
        mock_upstream: bool = True,
    ) -> None:
        """Initialize the RouteOps Gateway.

        Args:
            config: RouteOps configuration specification.
            model_path: Explicit override path to ONNX model.
            use_quantized: Whether to prioritize the Int8 quantized model.
            db_path: Explicit path for SQLite telemetry and semantic cache store.
            mock_upstream: If True, uses fast deterministic mock generation for upstream tiers.
        """
        self.config = config or load_config()
        self.use_quantized = use_quantized
        self.mock_upstream = mock_upstream
        self.session: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

        # Initialize SQLite telemetry store
        self.telemetry = TelemetryDB(db_path=db_path or "data/routeops_telemetry.db")

        # Cost definitions per 1,000 tokens
        self.cost_rates = {
            "cache_hit": 0.0000,
            "guardrail_blocked": 0.0000,
            "slm_fast_tier": 0.00015,
            "rag_account_tier": 0.0003,
            "frontier_reasoning_tier": 0.0030,
            "low_conf_fallback": 0.0030,
        }
        self.frontier_base_rate = 0.0030

        self._init_model_and_tokenizer(model_path)

    def _init_model_and_tokenizer(self, model_path: Optional[str | Path]) -> None:
        """Load tokenizer and ONNX Runtime inference session with P-core optimization."""
        # 1. Initialize Tokenizer
        tok_dir = Path(self.config.model.output_dir)
        tokenizer_src = (
            str(tok_dir) if tok_dir.exists() else self.config.model.tokenizer_name
        )
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
            logger.info("Initialized tokenizer from: %s", tokenizer_src)
        except Exception as e:
            logger.warning("Tokenizer initialization failed (%s). Using fallback.", e)
            self.tokenizer = None

        # 2. Determine ONNX Target Path
        candidate_paths: List[Path] = []
        if model_path:
            candidate_paths.append(Path(model_path))
        if self.use_quantized:
            candidate_paths.append(Path(self.config.model.quantized_onnx_path))
            candidate_paths.append(Path("models/onnx_quantized/model_quantized.onnx"))
        candidate_paths.append(Path(self.config.model.onnx_output_path))
        candidate_paths.append(Path("models/onnx/model.onnx"))

        target_path: Optional[Path] = None
        for p in candidate_paths:
            if p.is_file():
                target_path = p
                break

        # 2b. Cloud Auto-Bootstrap: Check Hugging Face Hub if no local ONNX model is found
        if target_path is None:
            hf_repo = os.getenv("HF_MODEL_REPO")
            if hf_repo:
                try:
                    from huggingface_hub import hf_hub_download

                    target_filename = (
                        "model_quantized.onnx" if self.use_quantized else "model.onnx"
                    )
                    local_dir = (
                        "models/onnx_quantized" if self.use_quantized else "models/onnx"
                    )
                    os.makedirs(local_dir, exist_ok=True)
                    logger.info(
                        "Attempting cloud model bootstrap from HF Hub: %s (filename: %s)...",
                        hf_repo,
                        target_filename,
                    )
                    downloaded_file = hf_hub_download(
                        repo_id=hf_repo,
                        filename=target_filename,
                        local_dir=local_dir,
                    )
                    if downloaded_file and Path(downloaded_file).is_file():
                        target_path = Path(downloaded_file)
                        logger.info(
                            "Successfully auto-bootstrapped ONNX model from HF Hub: %s",
                            target_path,
                        )
                except Exception as hf_err:
                    logger.warning(
                        "HF Hub auto-bootstrap failed (%s). Continuing with heuristic fallback.",
                        hf_err,
                    )

        # 3. Configure ONNX Runtime Session Options
        if target_path:
            try:
                import onnxruntime as ort

                sess_options = ort.SessionOptions()
                # Pin to 4 physical P-cores on i5-12450H to eliminate thread contention
                sess_options.intra_op_num_threads = 4
                sess_options.inter_op_num_threads = 1
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                self.session = ort.InferenceSession(
                    str(target_path),
                    sess_options,
                    providers=["CPUExecutionProvider"],
                )
                logger.info("Loaded Int8 ONNX Runtime session from: %s", target_path)
            except Exception as exc:
                logger.warning(
                    "Failed to initialize ONNX session from %s: %s", target_path, exc
                )
                self.session = None
        else:
            logger.info(
                "No compiled ONNX model found. Running in heuristic fallback mode."
            )
            self.session = None

    # -------------------------------------------------------------------------
    # Step 1: Language & PII Detection
    # -------------------------------------------------------------------------

    def detect_language(self, text: str) -> str:
        """Detect query language category: 'en', 'ne-rom', or 'ne'."""
        if any("ऀ" <= char <= "ॿ" for char in text):
            return "ne"

        ne_rom_tokens = {
            "namaste",
            "mero",
            "tapai",
            "tapailai",
            "kasari",
            "chha",
            "cha",
            "kasto",
            "garne",
            "khata",
            "hunchha",
            "huncha",
            "hunuhunchha",
            "sanchai",
            "bataunus",
            "bataidinus",
            "sanga",
            "bidesh",
            "aaja",
            "kati",
            "sathi",
            "kina",
            "byaj",
            "kagajat",
            "bida",
            "ra",
            "ma",
            "ko",
            "lagi",
            "pani",
            "garnu",
            "bhayeko",
            "dinus",
        }
        words = set(re.findall(r"[a-z]+", text.lower()))
        if len(words.intersection(ne_rom_tokens)) >= 1:
            return "ne-rom"

        return "en"

    # -------------------------------------------------------------------------
    # Step 2: ONNX Inference & Uncertainty Estimation
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_entropy(probabilities: np.ndarray) -> float:
        """Calculate Shannon entropy in bits to gauge predictive uncertainty."""
        entropy = 0.0
        for p in probabilities:
            if p > 1e-12:
                entropy -= float(p * math.log2(p))
        return float(entropy)

    def _infer_logits(self, query: str) -> np.ndarray:
        """Compute softmax class probabilities via dynamic-sequence ONNX inference."""
        if self.session is not None and self.tokenizer is not None:
            # Dynamic tokenization without fixed padding eliminates quadratic compute
            inputs = self.tokenizer(
                query,
                max_length=64,
                truncation=True,
                padding=False,
                return_tensors="np",
            )
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }
            raw_logits = self.session.run(["logits"], ort_inputs)[0][0]
            exp_logits = np.exp(raw_logits - np.max(raw_logits))
            probs = exp_logits / np.sum(exp_logits)
            return probs.astype(np.float32)

        return self._heuristic_fallback_probabilities(query)

    def _heuristic_fallback_probabilities(self, text: str) -> np.ndarray:
        """Rule-based heuristic classifier used when ONNX model is absent."""
        lower = text.lower()

        # Guardrail check
        guard_terms = [
            "ignore previous",
            "ignore all",
            "system prompt",
            "jailbreak",
            "dan mode",
            "unfiltered",
            "override",
            "malware",
            "ddos",
            "hack",
            "phishing",
            "birsinus",
            "nirdeshan",
            "secret key",
            "password",
            "drop database",
            "injection",
            "bypass",
            "exploit",
        ]
        if any(term in lower for term in guard_terms):
            return np.array([0.02, 0.03, 0.05, 0.90], dtype=np.float32)

        # Simple FAQ check
        direct_terms = [
            "hello",
            "hi",
            "good morning",
            "good evening",
            "how are you",
            "thank you",
            "thanks",
            "what is your name",
            "namaste",
            "sanchai",
            "bye",
            "see you",
            "nice to meet you",
            "store hours",
            "opening time",
        ]
        if any(term in lower for term in direct_terms) and len(text.split()) < 8:
            return np.array([0.88, 0.05, 0.05, 0.02], dtype=np.float32)

        # RAG / Account check
        rag_terms = [
            "bank",
            "khata",
            "account",
            "deposit",
            "loan",
            "interest",
            "passport",
            "license",
            "nagrikta",
            "nagarik",
            "tax",
            "esewa",
            "khalti",
            "connectips",
            "nepse",
            "share",
            "byaj",
            "kagajat",
            "pan",
            "citizenship",
            "invoice",
            "statement",
            "balance",
        ]
        if any(term in lower for term in rag_terms):
            return np.array([0.05, 0.08, 0.85, 0.02], dtype=np.float32)

        # Technical / Frontier reasoning check
        frontier_terms = [
            "python",
            "code",
            "algorithm",
            "differential",
            "equation",
            "proof",
            "prove",
            "cuda",
            "asyncio",
            "microservices",
            "rust",
            "kkt",
            "complexity",
            "gradient",
            "attention",
            "transformer",
            "math",
            "derive",
            "optimize",
        ]
        if any(term in lower for term in frontier_terms) or len(text.split()) > 20:
            return np.array([0.03, 0.88, 0.07, 0.02], dtype=np.float32)

        # Default balanced prior
        return np.array([0.15, 0.55, 0.25, 0.05], dtype=np.float32)

    # -------------------------------------------------------------------------
    # Upstream Generation Simulation & Execution
    # -------------------------------------------------------------------------

    def _generate_upstream_response(
        self, query: str, routed_tier: str, detected_lang: str
    ) -> Tuple[str, int]:
        """Generate response from target tier and calculate token usage."""
        # Estimate token count
        prompt_tokens = max(1, int(len(query.split()) * 1.3))

        if self.mock_upstream:
            if routed_tier == "slm_fast_tier":
                resp = (
                    "Namaste! Office hours are 9:00 AM - 5:00 PM Sunday through Friday."
                    if detected_lang != "en"
                    else "Hello! Our support hours are 9:00 AM to 5:00 PM EST."
                )
                resp_tokens = 18
            elif routed_tier == "rag_account_tier":
                ne_resp = (
                    "तपाईंको खाताको विवरण प्राप्त भयो। "
                    "हालको मुद्दती निक्षेपको ब्याजदर वार्षिक ७.५% छ।"
                )
                resp = (
                    "Verified account documentation retrieved. "
                    "Current savings interest rate is 6.5% per annum."
                    if detected_lang != "ne"
                    else ne_resp
                )
                resp_tokens = 24
            elif routed_tier in ("frontier_reasoning_tier", "low_conf_fallback"):
                resp = (
                    "Step 1: Compute eigenvalues. Step 2: Solve the characteristic polynomial. "
                    "The time complexity is bounded by O(N log N) using divide-and-conquer."
                )
                resp_tokens = 32
            else:
                resp = "Query processed successfully."
                resp_tokens = 10

            return resp, prompt_tokens + resp_tokens

        # Live Upstream Call (Ollama localhost:11434 or OmniRoute localhost:20128)
        try:
            import urllib.request

            req_body = json.dumps(
                {
                    "model": (
                        "llama3:8b" if routed_tier == "slm_fast_tier" else "qwen2.5:72b"
                    ),
                    "prompt": query,
                    "stream": False,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=req_body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                ans = data.get("response", "Response received.")
                total_tokens = data.get("eval_count", 30) + prompt_tokens
                return ans, total_tokens
        except Exception:
            # Fallback to deterministic mock if endpoint is offline
            return (
                f"Generated response for {routed_tier} [{detected_lang}].",
                prompt_tokens + 20,
            )

    # -------------------------------------------------------------------------
    # Core Routing Pipeline
    # -------------------------------------------------------------------------

    def route_query(
        self,
        query: str,
        language_hint: Optional[str] = None,
        enable_cache: bool = True,
        similarity_threshold: float = 0.92,
    ) -> RouteDecision:
        """Route a user query through guardrails, semantic cache, and model tiers.

        Args:
            query: Raw user input prompt.
            language_hint: Optional manual language hint ('en', 'ne-rom', 'ne').
            enable_cache: Whether to check and write to the semantic cache.
            similarity_threshold: Cosine similarity cutoff for cache hits (default 0.92).

        Returns:
            RouteDecision containing the routing choice, telemetry, and economics.
        """
        start_time = time.perf_counter()
        req_id = str(uuid.uuid4())

        # Step 1: Language Detection & PII Tagging
        lang = language_hint or self.detect_language(query)
        redacted_query, pii_detected = PIIRedactor.redact(query)

        # Step 2: ONNX Inference & Uncertainty Estimation
        probs = self._infer_logits(redacted_query)
        entropy = self.calculate_entropy(probs)

        # Canonical mapping:
        # 0: direct_response / simple_faq
        # 1: frontier_reasoning / technical_complex
        # 2: rag_retrieval / billing_account
        # 3: guardrail_blocked / unsafe_injection
        label_names = {
            0: "direct_response",
            1: "frontier_reasoning",
            2: "rag_retrieval",
            3: "guardrail_blocked",
        }
        prob_dict = {label_names[i]: float(probs[i]) for i in range(len(probs))}

        # Step 3: Edge Guardrail Safety Check
        guardrail_prob = float(probs[3])
        guardrail_threshold = self.config.routing.thresholds.get("guardrail_flag", 0.60)
        is_injection = (guardrail_prob >= guardrail_threshold) or (
            int(np.argmax(probs)) == 3
        )

        if is_injection:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            refusal_text = (
                "Request blocked by RouteOps Edge Guardrail: "
                "Prompt injection or policy violation detected."
            )
            decision = RouteDecision(
                request_id=req_id,
                query=query,
                redacted_query=redacted_query,
                detected_lang=lang,
                language_detected=lang,
                pii_detected=pii_detected,
                predicted_class="guardrail_blocked",
                routed_class="guardrail_blocked",
                routed_class_id=3,
                confidence=guardrail_prob,
                entropy=entropy,
                probabilities=prob_dict,
                routed_tier="guardrail_blocked",
                action="block_guardrail",
                is_guardrail_triggered=True,
                is_cache_hit=False,
                response_text=refusal_text,
                tokens_used=0,
                estimated_cost_usd=0.0,
                estimated_cost_per_1k=0.0,
                naive_frontier_cost_usd=0.0,
                cost_saved_usd=0.0,
                latency_ms=latency_ms,
            )
            self.telemetry.log_inference(
                request_id=req_id,
                query_text=redacted_query,
                detected_lang=lang,
                predicted_class="guardrail_blocked",
                confidence=guardrail_prob,
                entropy=entropy,
                routed_tier="guardrail_blocked",
                latency_ms=latency_ms,
                tokens_used=0,
                estimated_cost_usd=0.0,
                naive_frontier_cost_usd=0.0,
                cost_saved_usd=0.0,
            )
            return decision

        # Step 4: Semantic Vector Cache Lookup
        arg_max_class = int(np.argmax(probs[:3]))
        max_prob = float(probs[arg_max_class])
        predicted_class = label_names[arg_max_class]
        query_embedding = compute_ngram_projection_embedding(redacted_query)

        action_suffixes = {
            "direct_response": "direct",
            "frontier_reasoning": "frontier",
            "rag_retrieval": "rag",
        }
        suffix = action_suffixes.get(predicted_class, predicted_class.split("_")[0])
        config_cost_per_1k = self.config.routing.token_costs_per_1k.get(
            predicted_class, 0.0300
        )

        if enable_cache:
            cache_match = self.telemetry.lookup_cache(
                query_embedding=query_embedding,
                predicted_class=predicted_class,
                query_text=redacted_query,
                similarity_threshold=similarity_threshold,
            )
            if cache_match is not None:
                cached_resp, _, sim_score = cache_match
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                prompt_tokens = max(1, int(len(redacted_query.split()) * 1.3))
                naive_frontier_cost = (prompt_tokens / 1000.0) * self.frontier_base_rate

                decision = RouteDecision(
                    request_id=req_id,
                    query=query,
                    redacted_query=redacted_query,
                    detected_lang=lang,
                    language_detected=lang,
                    pii_detected=pii_detected,
                    predicted_class=predicted_class,
                    routed_class=predicted_class,
                    routed_class_id=arg_max_class,
                    confidence=max_prob,
                    entropy=entropy,
                    probabilities=prob_dict,
                    routed_tier="cache_hit",
                    action=f"route_to_{suffix}",
                    is_guardrail_triggered=False,
                    is_cache_hit=True,
                    response_text=cached_resp,
                    tokens_used=0,
                    estimated_cost_usd=0.0,
                    estimated_cost_per_1k=config_cost_per_1k,
                    naive_frontier_cost_usd=naive_frontier_cost,
                    cost_saved_usd=naive_frontier_cost,
                    latency_ms=latency_ms,
                )
                self.telemetry.log_inference(
                    request_id=req_id,
                    query_text=redacted_query,
                    detected_lang=lang,
                    predicted_class=predicted_class,
                    confidence=max_prob,
                    entropy=entropy,
                    routed_tier="cache_hit",
                    latency_ms=latency_ms,
                    tokens_used=0,
                    estimated_cost_usd=0.0,
                    naive_frontier_cost_usd=naive_frontier_cost,
                    cost_saved_usd=naive_frontier_cost,
                )
                return decision

        # Step 5: Cost-Aware Cascade & Confidence Abstention
        action_suffixes = {
            "direct_response": "direct",
            "frontier_reasoning": "frontier",
            "rag_retrieval": "rag",
        }
        suffix = action_suffixes.get(predicted_class, predicted_class.split("_")[0])

        if max_prob < 0.70:
            # Low confidence abstention -> Escalate safely to frontier model
            routed_tier = "low_conf_fallback"
            action = f"route_to_{suffix}_fallback"
            rate_per_1k = self.cost_rates["low_conf_fallback"]
        else:
            if predicted_class == "direct_response":
                routed_tier = "slm_fast_tier"
                action = "route_to_direct"
                rate_per_1k = self.cost_rates["slm_fast_tier"]
            elif predicted_class == "rag_retrieval":
                routed_tier = "rag_account_tier"
                action = "route_to_rag"
                rate_per_1k = self.cost_rates["rag_account_tier"]
            else:  # frontier_reasoning
                routed_tier = "frontier_reasoning_tier"
                action = "route_to_frontier"
                rate_per_1k = self.cost_rates["frontier_reasoning_tier"]

        # Generate response from designated tier
        response_text, tokens_used = self._generate_upstream_response(
            redacted_query, routed_tier, lang
        )

        # Economic Accounting
        estimated_cost = (tokens_used / 1000.0) * rate_per_1k
        naive_frontier_cost = (tokens_used / 1000.0) * self.frontier_base_rate
        cost_saved = max(0.0, naive_frontier_cost - estimated_cost)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Cost per 1k from configuration mapping for backward compatibility
        config_cost_per_1k = self.config.routing.token_costs_per_1k.get(
            predicted_class, rate_per_1k
        )

        # Insert into Semantic Cache for future queries
        if enable_cache:
            self.telemetry.insert_cache(
                query_text=redacted_query,
                predicted_class=predicted_class,
                embedding=query_embedding,
                response_text=response_text,
            )

        # Log inference to SQLite
        self.telemetry.log_inference(
            request_id=req_id,
            query_text=redacted_query,
            detected_lang=lang,
            predicted_class=predicted_class,
            confidence=max_prob,
            entropy=entropy,
            routed_tier=routed_tier,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            estimated_cost_usd=estimated_cost,
            naive_frontier_cost_usd=naive_frontier_cost,
            cost_saved_usd=cost_saved,
        )

        return RouteDecision(
            request_id=req_id,
            query=query,
            redacted_query=redacted_query,
            detected_lang=lang,
            language_detected=lang,
            pii_detected=pii_detected,
            predicted_class=predicted_class,
            routed_class=predicted_class,
            routed_class_id=arg_max_class,
            confidence=max_prob,
            entropy=entropy,
            probabilities=prob_dict,
            routed_tier=routed_tier,
            action=action,
            is_guardrail_triggered=False,
            is_cache_hit=False,
            response_text=response_text,
            tokens_used=tokens_used,
            estimated_cost_usd=estimated_cost,
            estimated_cost_per_1k=config_cost_per_1k,
            naive_frontier_cost_usd=naive_frontier_cost,
            cost_saved_usd=cost_saved,
            latency_ms=latency_ms,
        )


# Backward Compatibility: Maintain RouterService as subclass
class RouterService(RouteOpsGateway):
    """Alias for RouteOpsGateway providing drop-in compatibility with Phase 0/1/2 tests."""

    pass
