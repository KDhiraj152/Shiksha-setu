"""
Auto-generated TypeScript types from Pydantic schemas.

This module provides the infrastructure for contract-first API development:
1. Pydantic models define the single source of truth
2. TypeScript types are generated at build time
3. OpenAPI spec is derived from the same models

Usage:
    python -m backend.schemas.generated.export_schemas
"""

from pathlib import Path

GENERATED_DIR = Path(__file__).parent
TYPESCRIPT_OUTPUT = GENERATED_DIR / "api-types.ts"
OPENAPI_OUTPUT = GENERATED_DIR / "openapi.json"

__all__ = ["GENERATED_DIR", "TYPESCRIPT_OUTPUT", "OPENAPI_OUTPUT"]
