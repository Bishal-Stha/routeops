"""RouteOps: Multilingual Semantic Router & Edge MLOps Gateway.

A production-grade semantic routing engine optimized for multilingual queries
(English, Romanized Nepali, Devanagari Nepali) with dynamic cost optimization,
guardrail filtering, and real-time MLOps drift observability.
"""

__version__ = "0.1.0"
__author__ = "RouteOps Team"

from src.config import RouteOpsConfig, load_config
from src.drift_monitor import DriftMonitor, QueryDriftRecord
from src.router_service import RouteDecision, RouterService

__all__ = [
    "__version__",
    "load_config",
    "RouteOpsConfig",
    "RouterService",
    "RouteDecision",
    "DriftMonitor",
    "QueryDriftRecord",
]
