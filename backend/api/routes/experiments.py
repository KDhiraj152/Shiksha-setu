"""
Experiments API routes for A/B testing and feature experiments.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any
import uuid
from datetime import datetime, timezone

from ...utils.auth import get_current_user, TokenData

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])


@router.post("/", status_code=201)
async def create_experiment(
    experiment_data: Dict[str, Any],
    current_user: TokenData = Depends(get_current_user)
):
    """Create a new A/B test experiment."""
    experiment_id = str(uuid.uuid4())
    
    return {
        "id": experiment_id,
        "name": experiment_data.get("name", "New Experiment"),
        "description": experiment_data.get("description", ""),
        "status": "created",
        "variants": experiment_data.get("variants", ["control", "variant_a"]),
        "created_at": datetime.now(timezone.utc).isoformat()
    }


@router.post("/{experiment_id}/start")
async def start_experiment(
    experiment_id: str,
    current_user: TokenData = Depends(get_current_user)
):
    """Start an experiment."""
    return {
        "id": experiment_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat()
    }


@router.get("/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    current_user: TokenData = Depends(get_current_user)
):
    """Get experiment details."""
    return {
        "id": experiment_id,
        "name": "Sample Experiment",
        "status": "running",
        "metrics": {
            "participants": 0,
            "conversions": 0
        }
    }


@router.post("/{experiment_id}/stop")
async def stop_experiment(
    experiment_id: str,
    current_user: TokenData = Depends(get_current_user)
):
    """Stop a running experiment."""
    return {
        "id": experiment_id,
        "status": "stopped",
        "stopped_at": datetime.now(timezone.utc).isoformat()
    }


@router.get("/{experiment_id}/assign")
async def assign_variant(
    experiment_id: str,
    user_id: str = Query(..., description="User ID for variant assignment"),
    current_user: TokenData = Depends(get_current_user)
):
    """Assign a user to an experiment variant (GET method for idempotent assignment)."""
    return {
        "experiment_id": experiment_id,
        "user_id": user_id,
        "variant_id": "control",  # Simple mock assignment - should use consistent hashing in production
        "assigned_at": datetime.now(timezone.utc).isoformat()
    }


@router.post("/{experiment_id}/track")
async def track_event(
    experiment_id: str,
    event_data: Dict[str, Any],
    current_user: TokenData = Depends(get_current_user)
):
    """Track an event for experiment analysis."""
    return {
        "experiment_id": experiment_id,
        "event_type": event_data.get("event_type"),
        "tracked": True,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/{experiment_id}/results")
async def get_results(
    experiment_id: str,
    current_user: TokenData = Depends(get_current_user)
):
    """Get experiment results."""
    return {
        "id": experiment_id,
        "variants": [
            {"name": "control", "participants": 50, "conversions": 15},
            {"name": "variant_a", "participants": 50, "conversions": 20}
        ],
        "winner": "variant_a"
    }
