"""FastAPI Semantic Gateway service for RouteOps.

Provides high-throughput, low-latency REST endpoints for multilingual semantic query
routing (/v1/route), operational health inspections (/health), Prometheus metric
scraping (/metrics), and real-time concept drift summaries (/v1/drift).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import load_config
from src.drift_monitor import DriftMonitor, DriftSummary
from src.router_service import RouteDecision, RouterService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("routeops.api")

# Global singletons
router_service: Optional[RouterService] = None
drift_monitor: Optional[DriftMonitor] = None
start_time: float = time.time()


def get_router_service() -> RouterService:
    """Retrieve or lazily initialize the singleton RouterService instance."""
    global router_service
    if router_service is None:
        config = load_config()
        router_service = RouterService(config=config)
    return router_service


def get_drift_monitor() -> DriftMonitor:
    """Retrieve or lazily initialize the singleton DriftMonitor instance."""
    global drift_monitor
    if drift_monitor is None:
        config = load_config()
        drift_monitor = DriftMonitor(config=config)
    return drift_monitor


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""
    global start_time
    start_time = time.time()
    logger.info("Initializing RouteOps Gateway runtime services...")

    # Preload singletons on startup
    get_router_service()
    get_drift_monitor()

    logger.info("RouteOps Gateway initialized and ready to receive traffic.")
    yield
    logger.info("Shutting down RouteOps Gateway...")


app = FastAPI(
    title="RouteOps Semantic Gateway",
    description="Cost-optimized multilingual query router with edge MLOps observability",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
_config = load_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_config.gateway.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------


class RouteRequest(BaseModel):
    """Incoming query routing payload."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="User text query in English, Romanized Nepali, or Devanagari",
        examples=["Mero bank account ko balance check garne tarika k ho?"],
    )
    language_hint: Optional[str] = Field(
        default=None,
        description="Optional language hint ('en', 'ne-rom', 'ne')",
        examples=["ne-rom"],
    )
    enforce_guardrail_block: bool = Field(
        default=False,
        description="If True, throws HTTP 403 Forbidden when guardrails are triggered",
    )


class HealthResponse(BaseModel):
    """Gateway health check status payload."""

    status: str
    uptime_seconds: float
    model_loaded: bool
    quantized_mode: bool
    total_queries_served: int
    vram_budget_mb: int


# -----------------------------------------------------------------------------
# REST Endpoints
# -----------------------------------------------------------------------------


@app.post(
    "/v1/route",
    response_model=RouteDecision,
    status_code=status.HTTP_200_OK,
    summary="Route query to optimal model tier",
)
async def route_query_endpoint(request: RouteRequest) -> RouteDecision:
    """Classify and route an incoming query across Direct, RAG, Frontier, or Guardrails."""
    router = get_router_service()
    monitor = get_drift_monitor()

    decision = router.route_query(
        query=request.query,
        language_hint=request.language_hint,
    )

    # Record telemetry into drift monitor and Prometheus
    monitor.record_query(
        routed_class=decision.routed_class,
        confidence=decision.confidence,
        entropy=decision.entropy,
        latency_ms=decision.latency_ms,
        query_text=decision.query,
        language=decision.language_detected,
        action=decision.action,
        is_guardrail_triggered=decision.is_guardrail_triggered,
    )

    if request.enforce_guardrail_block and decision.is_guardrail_triggered:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "message": "Query blocked by RouteOps safety guardrail.",
                "confidence": decision.confidence,
                "action": decision.action,
            },
        )

    return decision


@app.post(
    "/v1/route/stream",
    status_code=status.HTTP_200_OK,
    summary="Route query and stream execution steps and tokens via SSE",
)
async def route_query_stream_endpoint(request: RouteRequest) -> StreamingResponse:
    """Classify query, yield live execution steps and streaming tokens via SSE."""
    router = get_router_service()
    monitor = get_drift_monitor()

    decision = router.route_query(
        query=request.query,
        language_hint=request.language_hint,
    )

    # Record telemetry into drift monitor and Prometheus
    monitor.record_query(
        routed_class=decision.routed_class,
        confidence=decision.confidence,
        entropy=decision.entropy,
        latency_ms=decision.latency_ms,
        query_text=decision.query,
        language=decision.language_detected,
        action=decision.action,
        is_guardrail_triggered=decision.is_guardrail_triggered,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        # 1. Yield step 1: Language detection & PII
        step_1_data = json.dumps(
            {
                "step": "language_and_pii",
                "detected_lang": decision.detected_lang,
                "pii_detected": decision.pii_detected,
                "redacted_query": decision.redacted_query,
            }
        )
        yield f"data: {step_1_data}\n\n"
        await asyncio.sleep(0.04)

        # 2. Yield step 2: ONNX inference & confidence
        step_2_data = json.dumps(
            {
                "step": "onnx_classification",
                "predicted_class": decision.predicted_class,
                "confidence": decision.confidence,
                "entropy": decision.entropy,
                "probabilities": decision.probabilities,
            }
        )
        yield f"data: {step_2_data}\n\n"
        await asyncio.sleep(0.04)

        # 3. Yield step 3: Guardrail & Cache check
        step_3_data = json.dumps(
            {
                "step": "guardrail_and_cache",
                "is_guardrail_triggered": decision.is_guardrail_triggered,
                "is_cache_hit": decision.is_cache_hit,
                "routed_tier": decision.routed_tier,
                "action": decision.action,
            }
        )
        yield f"data: {step_3_data}\n\n"
        await asyncio.sleep(0.04)

        # 4. Stream response tokens word by word
        response_text = decision.response_text or "Query processed successfully."
        words = response_text.split(" ")
        token_chunk_size = max(1, len(words) // 8)

        for i in range(0, len(words), token_chunk_size):
            chunk = " ".join(words[i : i + token_chunk_size])
            if i + token_chunk_size < len(words):
                chunk += " "
            token_data = json.dumps(
                {
                    "step": "streaming_token",
                    "token": chunk,
                    "is_complete": False,
                }
            )
            yield f"data: {token_data}\n\n"
            await asyncio.sleep(0.03)

        # 5. Final summary event with full metadata
        final_data = json.dumps(
            {
                "step": "complete",
                "request_id": decision.request_id,
                "routed_class": decision.routed_class,
                "confidence": decision.confidence,
                "latency_ms": decision.latency_ms,
                "estimated_cost_usd": decision.estimated_cost_usd,
                "cost_saved_usd": decision.cost_saved_usd,
                "is_complete": True,
            }
        )
        yield f"data: {final_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service health inspection",
)
async def health_check() -> HealthResponse:
    """Return health metrics, runtime status, and total requests served."""
    router = get_router_service()
    monitor = get_drift_monitor()
    total_queries = len(monitor.window)
    is_model_ready = router.session is not None or router.tokenizer is not None

    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - start_time, 2),
        model_loaded=is_model_ready,
        quantized_mode=router.use_quantized,
        total_queries_served=total_queries,
        vram_budget_mb=_config.hardware.vram_budget_mb,
    )


@app.get(
    "/v1/drift",
    response_model=DriftSummary,
    status_code=status.HTTP_200_OK,
    summary="Inspect rolling query distribution and concept drift",
)
async def get_drift_status() -> DriftSummary:
    """Retrieve concept drift metrics across the current rolling request window."""
    monitor = get_drift_monitor()
    return monitor.get_summary()


@app.get(
    "/metrics",
    summary="Scrape Prometheus metrics",
)
async def prometheus_metrics() -> Response:
    """Expose Prometheus telemetry metrics for scraping."""
    monitor = get_drift_monitor()
    raw_metrics = monitor.export_metrics()
    return Response(content=raw_metrics, media_type="text/plain; version=0.0.4")


# Mount static UI dashboard at root
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
