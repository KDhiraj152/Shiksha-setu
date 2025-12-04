"""
Model Collaboration System - Inter-Model Communication & Orchestration
=======================================================================

v2.1.0 - Refactored into modular package (December 2025)

This module has been refactored into the `collaboration` package for better
maintainability. All exports remain backward compatible.

REFACTORED STRUCTURE:
- collaboration/types.py: Enums, dataclasses, constants
- collaboration/model_accessors.py: Lazy model loading
- collaboration/helpers.py: Utility functions
- collaboration/patterns/: Individual pattern implementations
  - base.py: Chain, Verify, Semantic patterns
  - translation.py: Back-translate pattern
  - ensemble.py: Ensemble, Iterative, Debate patterns
  - audio.py: Audio verify pattern
  - document.py: Document chain pattern
  - rerank.py: Rerank pattern
- collaboration/orchestrator.py: Main ModelCollaborator class
- collaboration/convenience.py: High-level convenience functions

MODELS PARTICIPATING (ALL 8):
- Qwen2.5-3B (Orchestrator): Main LLM, coordinates and generates
- IndicTrans2 (Translator): Translation with back-translation verification
- BGE-M3 (Semantic Judge): Embedding similarity for semantic preservation
- BGE-Reranker (Quality Ranker): Ranks multiple outputs to select best
- Gemma-2-2B (Validator): Quality scoring and curriculum alignment
- MMS-TTS (Audio Generator): Text-to-Speech for audio content
- Whisper (Audio Verifier): STT to verify TTS output accuracy
- GOT-OCR2 (Document Reader): OCR for images and documents

COLLABORATION PATTERNS (10 TOTAL):
1. Chain: A → B → C (sequential with context passing)
2. Verify: A generates, B validates, A refines
3. Back-Translate: Translate → Back-translate → Compare
4. Ensemble: A, B, C all evaluate, consensus decision
5. Debate: A and B propose, C judges winner
6. Iterative: A and B take turns improving output
7. Semantic Check: BGE-M3 validates meaning preservation
8. Audio Verify: TTS → Whisper → Compare text
9. Document Chain: OCR → Simplify → Translate → Audio
10. Rerank: Generate multiple → BGE-Reranker selects best

Usage:
    # Import from the new package
    from backend.services.pipeline import (
        ModelCollaborator,
        CollaborationPattern,
        get_model_collaborator,
        collaborate_and_simplify,
    )

    # Or import from this module for backward compatibility
    from backend.services.pipeline.model_collaboration import (
        ModelCollaborator,
        CollaborationPattern,
        get_model_collaborator,
    )
"""

# Re-export everything from the collaboration package for backward compatibility
from .collaboration import (
    ALL_MODELS,
    MODEL_BGE_M3,
    MODEL_BGE_RERANKER,
    MODEL_GEMMA2,
    MODEL_GOT_OCR2,
    MODEL_INDICTRANS2,
    MODEL_MMS_TTS,
    # Constants
    MODEL_QWEN25,
    MODEL_WHISPER,
    CollaborationConfig,
    # Types
    CollaborationPattern,
    CollaborationResult,
    CollaboratorConfig,
    # Orchestrator
    ModelCollaborator,
    ModelMessage,
    ModelRole,
    collaborate_and_simplify,
    collaborate_and_translate,
    ensemble_evaluate,
    full_educational_pipeline,
    generate_best_output,
    # Convenience functions
    get_model_collaborator,
    process_document,
    verify_audio_output,
)

__all__ = [
    "ALL_MODELS",
    "MODEL_BGE_M3",
    "MODEL_BGE_RERANKER",
    "MODEL_GEMMA2",
    "MODEL_GOT_OCR2",
    "MODEL_INDICTRANS2",
    "MODEL_MMS_TTS",
    # Constants
    "MODEL_QWEN25",
    "MODEL_WHISPER",
    "CollaborationConfig",
    # Types
    "CollaborationPattern",
    "CollaborationResult",
    "CollaboratorConfig",
    # Orchestrator
    "ModelCollaborator",
    "ModelMessage",
    "ModelRole",
    "collaborate_and_simplify",
    "collaborate_and_translate",
    "ensemble_evaluate",
    "full_educational_pipeline",
    "generate_best_output",
    # Convenience functions
    "get_model_collaborator",
    "process_document",
    "verify_audio_output",
]
