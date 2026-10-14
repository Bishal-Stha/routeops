"""Core Semantic Routing Engine for RouteOps.

Executes low-latency ONNX Runtime inference, applies multi-class softmax probabilities,
evaluates guardrail flags, applies confidence thresholds, and decides optimal routing paths.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel

from src.config import RouteOpsConfig, load_config

logger = logging.getLogger("routeops.router_service")


class RouteDecision(BaseModel):
    """Structured decision output from the semantic router."""

    query: str
    routed_class: str
    routed_class_id: int
    confidence: float
    probabilities: Dict[str, float]
    action: str
    estimated_cost_per_1k: float
    latency_ms: float
    entropy: float
    language_detected: str = "en"
    is_guardrail_triggered: bool = False


class RouterService:
    """Production semantic router managing ONNX inference and routing policy."""

    def __init__(
        self,
        config: Optional[RouteOpsConfig] = None,
        model_path: Optional[str | Path] = None,
        use_quantized: bool = True,
    ) -> None:
        """Initialize the router service.

        Args:
            config: Configuration object. If None, loaded automatically.
            model_path: Explicit override path to the ONNX model file.
            use_quantized: If True, prioritizes quantized Int8 model.
        """
        self.config = config or load_config()
        self.use_quantized = use_quantized
        self.session: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

        self._init_model_and_tokenizer(model_path)

    def _init_model_and_tokenizer(self, model_path: Optional[str | Path]) -> None:
        """Load tokenizer and ONNX Runtime inference session."""
        # 1. Initialize Tokenizer
        tokenizer_src = (
            self.config.model.output_dir
            if Path(self.config.model.output_dir).exists()
            else self.config.model.tokenizer_name
        )
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
            logger.info(f"Initialized tokenizer from: {tokenizer_src}")
        except Exception as e:
            logger.warning(
                f"Could not load HuggingFace tokenizer ({e}). Using regex fallback."
            )
            self.tokenizer = None

        # 2. Determine ONNX Model Path
        candidate_paths: List[Path] = []
        if model_path:
            candidate_paths.append(Path(model_path))
        if self.use_quantized:
            candidate_paths.append(Path(self.config.model.quantized_onnx_path))
        candidate_paths.append(Path(self.config.model.onnx_output_path))

        target_path: Optional[Path] = None
        for p in candidate_paths:
            if p.is_file():
                target_path = p
                break

        # 3. Initialize ONNX Session
        if target_path:
            try:
                import onnxruntime as ort

                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = self.config.hardware.cpu_threads
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                self.session = ort.InferenceSession(
                    str(target_path),
                    sess_options,
                    providers=["CPUExecutionProvider"],
                )
                logger.info(f"Loaded ONNX inference session from: {target_path}")
            except Exception as exc:
                logger.warning(f"Failed to load ONNX session from {target_path}: {exc}")
                self.session = None
        else:
            logger.info(
                "No ONNX model file found. Running in rule-heuristic fallback mode."
            )
            self.session = None

    def detect_language(self, text: str) -> str:
        """Detect query language category (en, ne-rom, or ne).

        Args:
            text: Query string.

        Returns:
            Language code string ('en', 'ne-rom', or 'ne').
        """
        # Check for Devanagari Unicode range (0900-097F)
        if any("ऀ" <= char <= "ॿ" for char in text):
            return "ne"

        # Heuristic Romanized Nepali keywords
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
            "bataunus",
            "bataidinus",
            "sanga",
            "bidesh",
            "aaja",
            "kati",
            "k k",
            "ko lagi",
            "sathi",
            "kina",
        }
        lower_words = set(text.lower().split())
        if len(lower_words.intersection(ne_rom_tokens)) >= 1:
            return "ne-rom"

        return "en"

    def _fallback_predict_probabilities(self, text: str) -> np.ndarray:
        """Rule-based heuristic classifier used when ONNX model is not yet compiled."""
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
            "illegal",
            "secret key",
            "password",
            "drop database",
            "injection",
            "bypass",
            "exploit",
        ]
        if any(term in lower for term in guard_terms):
            return np.array([0.02, 0.03, 0.05, 0.90], dtype=np.float32)

        # Direct response check
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
        ]
        if any(term in lower for term in direct_terms) and len(text.split()) < 8:
            return np.array([0.88, 0.05, 0.05, 0.02], dtype=np.float32)

        # RAG domain knowledge check (banking, passport, tax, SSF, license)
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
        ]
        if any(term in lower for term in rag_terms):
            return np.array([0.05, 0.08, 0.85, 0.02], dtype=np.float32)

        # Frontier reasoning check (code, math, complex algorithms)
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
        ]
        if any(term in lower for term in frontier_terms) or len(text.split()) > 20:
            return np.array([0.03, 0.88, 0.07, 0.02], dtype=np.float32)

        # Default balanced prior
        return np.array([0.15, 0.55, 0.25, 0.05], dtype=np.float32)

    def _infer_logits(self, query: str) -> np.ndarray:
        """Compute raw logits via ONNX inference or heuristic fallback."""
        if self.session is not None and self.tokenizer is not None:
            inputs = self.tokenizer(
                query,
                max_length=self.config.model.max_length,
                truncation=True,
                return_tensors="np",
            )
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }
            logits = self.session.run(["logits"], ort_inputs)[0][0]
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            return probs.astype(np.float32)

        return self._fallback_predict_probabilities(query)

    @staticmethod
    def calculate_entropy(probabilities: np.ndarray) -> float:
        """Calculate Shannon entropy in bits to gauge model uncertainty."""
        entropy = 0.0
        for p in probabilities:
            if p > 1e-12:
                entropy -= float(p * math.log2(p))
        return float(entropy)

    def route_query(
        self,
        query: str,
        language_hint: Optional[str] = None,
    ) -> RouteDecision:
        """Route a user query through guardrails and semantic classifiers.

        Args:
            query: The text query to classify and route.
            language_hint: Optional manual language hint.

        Returns:
            RouteDecision containing the routing choice, probabilities, and cost.
        """
        start_time = time.perf_counter()

        lang = language_hint or self.detect_language(query)
        probs = self._infer_logits(query)
        entropy = self.calculate_entropy(probs)

        # Label mapping:
        # 0: direct_response, 1: rag_retrieval, 2: frontier_reasoning, 3: guardrail_blocked
        prob_dict = {
            self.config.routing.labels[i]: float(probs[i]) for i in range(len(probs))
        }

        guardrail_threshold = self.config.routing.thresholds.get("guardrail_flag", 0.60)

        # 1. Guardrail Safety Check
        guardrail_prob = float(probs[3])
        if guardrail_prob >= guardrail_threshold:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return RouteDecision(
                query=query,
                routed_class="guardrail_blocked",
                routed_class_id=3,
                confidence=guardrail_prob,
                probabilities=prob_dict,
                action="block_guardrail",
                estimated_cost_per_1k=self.config.routing.token_costs_per_1k.get(
                    "guardrail_blocked", 0.0
                ),
                latency_ms=latency_ms,
                entropy=entropy,
                language_detected=lang,
                is_guardrail_triggered=True,
            )

        # 2. Semantic Routing Logic with Dynamic Class Configuration
        arg_max_class = int(np.argmax(probs[:3]))
        max_prob = float(probs[arg_max_class])
        routed_class = self.config.routing.labels.get(
            arg_max_class, "frontier_reasoning"
        )
        routed_id = arg_max_class
        threshold = self.config.routing.thresholds.get(routed_class, 0.70)

        action_suffixes = {
            "direct_response": "direct",
            "frontier_reasoning": "frontier",
            "rag_retrieval": "rag",
        }
        suffix = action_suffixes.get(routed_class, routed_class.split("_")[0])

        if max_prob >= threshold:
            action = f"route_to_{suffix}"
        else:
            action = f"route_to_{suffix}_fallback"

        cost = self.config.routing.token_costs_per_1k.get(routed_class, 0.0300)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return RouteDecision(
            query=query,
            routed_class=routed_class,
            routed_class_id=routed_id,
            confidence=max_prob,
            probabilities=prob_dict,
            action=action,
            estimated_cost_per_1k=cost,
            latency_ms=latency_ms,
            entropy=entropy,
            language_detected=lang,
            is_guardrail_triggered=False,
        )
