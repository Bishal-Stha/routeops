"""Vercel Python Serverless Function entrypoint for RouteOps API Gateway.

Wraps the FastAPI application to serve /v1/route, /health, /metrics, /v1/drift,
and documentation endpoints on Vercel serverless deployments.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api import app
