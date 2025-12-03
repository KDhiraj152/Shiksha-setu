"""
Policy Module - Configurable Content Policy Engine
===================================================

This module consolidates all hardcoded content filtering, curriculum alignment,
and safety logic into a single, configurable policy engine.

Environment Variables:
- ALLOW_UNRESTRICTED_MODE: When true, bypasses all policy filters (default: false)
- POLICY_FILTERS_ENABLED: Enable/disable content filtering (default: true)
- SENSITIVE_RESPONSE_BLOCKING: Block sensitive responses (default: true)  
- CURRICULUM_ENFORCEMENT: Enforce curriculum alignment (default: true)
- ALLOW_EXTERNAL_CALLS: Allow external API calls (default: false)
- EXTERNAL_CALL_AUDIT: Log all external calls (default: true)

Usage:
    from backend.policy import get_policy_engine, PolicyMode
    
    engine = get_policy_engine()
    
    # Check current mode
    if engine.mode == PolicyMode.UNRESTRICTED:
        # No filtering applied
        pass
    
    # Apply policy to input
    result = engine.apply_input_policy(user_input)
    if result.blocked:
        return result.rejection_message
    
    # Apply policy to output
    output = engine.apply_output_policy(model_output)
"""

from .policy_module import (
    PolicyEngine,
    PolicyMode,
    PolicyConfig,
    PolicyCheckResult,
    get_policy_engine,
    print_startup_banner,
    reset_policy_engine,
)

__all__ = [
    "PolicyEngine",
    "PolicyMode", 
    "PolicyConfig",
    "PolicyCheckResult",
    "get_policy_engine",
    "print_startup_banner",
    "reset_policy_engine",
]
