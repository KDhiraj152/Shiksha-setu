 # AI / ML Pipeline — Shiksha Setu

 Overview
- The content pipeline is staged: OCR → Simplification → Translation → Validation → Text-to-Speech.
- Pipeline location: `backend/pipeline/` and stage implementations in `backend/simplify/`, `backend/translate/`, `backend/validate/`, `backend/speech/`.

Stages
1. OCR
   - Tools: PyMuPDF (for PDF text extraction) and Tesseract for image OCR when needed.
   - Output: cleaned UTF-8 text. Handle fonts and complex scripts (Indic scripts) carefully.

2. Simplification
   - Model: Flan-T5 family (or hosted HF endpoint).
   - Purpose: reduce reading complexity to a target grade-level while preserving key facts.
   - Implementation: `backend/simplify/simplifier.py` and `backend/simplify/analyzer.py`.

3. Translation
   - Model: IndicTrans2 (AI4Bharat) or other HF models for Indian languages.
   - Strategy: perform quality checks and fallback to cloud translation if local model fails.

4. Validation
   - Model: BERT-based semantic similarity check to ensure NCERT alignment and factual consistency.
   - Thresholds: NCERT alignment >=80% (configurable). Implement soft-fail vs. hard-fail policies.

5. Text-to-Speech
   - Engines: VITS/Bhashini or Coqui TTS / MMS-TTS. Use GPU for speed when available.
   - Ensure consistent sample rates and chunked audio streaming support.

Orchestration & resilience
- Orchestrator: `backend/pipeline/orchestrator.py` (or similar) implements retry/backoff (2^n) with per-stage max retries.
- Use Celery for background execution with Redis as broker.
- Make models lazy-load and cache model instances across tasks to reduce cold-start overhead.
- Use batching for inference when possible; group small requests to utilize GPUs efficiently.

Optimization & cost control
- Cache model outputs for idempotent inputs.
- Use ASR-based TTS validation to compute `audio_accuracy` (>=90% recommended).
- Fall back to lower-cost cloud inference only when local models give low confidence.

Resource & production notes
- Assign GPU-enabled worker nodes for TTS and large transformer inference.
- Monitor GPU memory and use mixed precision (FP16) where safe.
- Pin model versions and include `model_card` references.

Testing
- Unit test each stage and run integration tests for the full pipeline in `tests/`.
