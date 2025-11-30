# AI Pipeline Architecture

Four-stage content processing pipeline with model routing, sentence-level optimization, and async task orchestration.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT                                          │
│                    Text / PDF / Image Upload                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 0: OCR (Optional)                             │
│                                                                             │
│  Model: GOT-OCR2 (ucaslcl/GOT-OCR2_0)                                      │
│  Input: PDF/Image → Output: Extracted text                                  │
│  Features: 95%+ accuracy on Indian scripts, formula recognition            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 1: SIMPLIFICATION                                │
│                                                                             │
│  Model: Llama-3.2-3B-Instruct (vLLM serving)                               │
│  Input: Complex text + Grade level → Output: Simplified text               │
│  Optimization: Sentence-level batching, INT4 quantization                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STAGE 2: TRANSLATION                                  │
│                                                                             │
│  Model: IndicTrans2-1B (ai4bharat/indictrans2-en-indic-1B)                 │
│  Input: Simplified text + Target language → Output: Translated text        │
│  Languages: Hindi, Tamil, Telugu, Bengali, Marathi, +5 more               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STAGE 3: VALIDATION                                   │
│                                                                             │
│  Model: Gemma-2-2B-it + BERT embeddings                                    │
│  Input: Original + Translated text → Output: Quality scores                │
│  Checks: Semantic accuracy (≥80%), NCERT alignment (≥80%), script          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 4: TEXT-TO-SPEECH                                  │
│                                                                             │
│  Model: AI4Bharat Indic-TTS                                                │
│  Input: Translated text + Language → Output: Audio file (MP3)              │
│  Features: Technical term handling, prosody optimization                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT                                         │
│        ProcessedContent (simplified + translated + validated + audio)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Sequence Diagram

```
User          API           Celery         Pipeline        Models
  │            │              │               │               │
  │──Request──▶│              │               │               │
  │            │──Queue Task─▶│               │               │
  │◀──TaskID───│              │               │               │
  │            │              │──Process──────▶               │
  │            │              │               │               │
  │            │              │               │──OCR─────────▶│
  │            │              │               │◀─────Text─────│
  │            │              │               │               │
  │            │              │               │──Simplify────▶│
  │            │              │               │◀────Text──────│
  │            │              │               │               │
  │            │              │               │──Translate───▶│
  │            │              │               │◀────Text──────│
  │            │              │               │               │
  │            │              │               │──Validate────▶│
  │            │              │               │◀───Scores─────│
  │            │              │               │               │
  │            │              │               │──TTS─────────▶│
  │            │              │               │◀────Audio─────│
  │            │              │               │               │
  │            │              │◀──Complete────│               │
  │──Poll──────▶              │               │               │
  │◀──Result───│              │               │               │
```

---

## Model Stack

### 2025 Optimal Configuration

| Stage | Model | Size | Quantization | Serving |
|-------|-------|------|--------------|---------|
| OCR | `ucaslcl/GOT-OCR2_0` | 1.5B | FP16 | Transformers |
| Simplification | `meta-llama/Llama-3.2-3B-Instruct` | 3B | INT4/AWQ | vLLM |
| Translation | `ai4bharat/indictrans2-en-indic-1B` | 1B | FP16 | Transformers |
| Embeddings | `BAAI/bge-m3` | 568M | FP16 | Transformers |
| Reranker | `BAAI/bge-reranker-v2-m3` | 568M | FP16 | Transformers |
| Validation | `google/gemma-2-2b-it` | 2B | INT8 | Transformers |
| TTS | `ai4bharat/indic-tts` | - | - | VITS |

### Total Memory Footprint

| Device | Configuration | VRAM Required |
|--------|--------------|---------------|
| Local (M4) | INT4 quantization | ~10GB |
| Production (GPU) | AWQ quantization | ~16GB |

---

## Pipeline Orchestrator

### Location

`backend/services/pipeline/orchestrator.py`

### Core Class

```python
class ContentPipelineOrchestrator:
    """
    Four-stage content processing pipeline.
    """
    
    SUPPORTED_LANGUAGES = ['Hindi', 'Tamil', 'Telugu', 'Bengali', 'Marathi']
    SUPPORTED_SUBJECTS = ['Mathematics', 'Science', 'Social Studies', 'English']
    MIN_GRADE, MAX_GRADE = 5, 12
    
    MAX_RETRIES = 3
    NCERT_ALIGNMENT_THRESHOLD = 0.80
    
    def process_content(
        self,
        input_data: str,
        target_language: str,
        grade_level: int,
        subject: str,
        output_format: str = 'both'
    ) -> ProcessedContentResult:
        """Execute full pipeline."""
        ...
```

### Stage Execution

Each stage follows this pattern:

```python
def _execute_stage(self, stage: PipelineStage, func, *args):
    """Execute a pipeline stage with retry logic."""
    start_time = time.time()
    
    for attempt in range(self.MAX_RETRIES):
        try:
            result = func(*args)
            
            self.metrics.append(StageMetrics(
                stage=stage.value,
                processing_time_ms=int((time.time() - start_time) * 1000),
                success=True,
                retry_count=attempt
            ))
            
            return result
            
        except Exception as e:
            if attempt == self.MAX_RETRIES - 1:
                raise PipelineStageError(f"{stage.value} failed: {e}")
            
            time.sleep(self.RETRY_BACKOFF_BASE ** attempt)
```

---

## Stage Details

### Stage 0: OCR

**File**: `backend/services/ocr.py`

```python
class GOTOCR2:
    """GOT-OCR2 Vision-Language Model."""
    
    OCR_MODES = {
        'plain': 'ocr',        # Plain text
        'format': 'format',    # Preserve formatting
        'fine-grained': 'fine' # Detailed extraction
    }
    
    def extract(self, image: Image) -> ExtractionResult:
        """Extract text from image."""
        ...
```

**Features**:
- 95%+ accuracy on Indian scripts
- Formula recognition (LaTeX)
- Table extraction
- Mixed layout handling

---

### Stage 1: Simplification

**File**: `backend/services/simplify/simplifier.py`

```python
class TextSimplifier:
    """Grade-level text simplification."""
    
    def simplify(
        self,
        text: str,
        grade_level: int,
        subject: str
    ) -> SimplifiedText:
        """
        Simplify text for target grade.
        
        Uses Llama-3.2-3B-Instruct with prompt:
        "Simplify this {subject} content for grade {grade} students..."
        """
        ...
```

**Optimization**:
- Sentence-level batching for long texts
- vLLM serving for production (2-3x faster)
- INT4 quantization (3GB VRAM)

---

### Stage 2: Translation

**File**: `backend/services/translate/engine.py`

```python
class TranslationEngine:
    """IndicTrans2 translation engine."""
    
    LANGUAGE_CODES = {
        'Hindi': 'hin_Deva',
        'Tamil': 'tam_Taml',
        'Telugu': 'tel_Telu',
        'Bengali': 'ben_Beng',
        'Marathi': 'mar_Deva',
        'Gujarati': 'guj_Gujr',
        'Kannada': 'kan_Knda',
        'Malayalam': 'mal_Mlym',
        'Punjabi': 'pan_Guru',
        'Odia': 'ory_Orya'
    }
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """Translate text to target language."""
        ...
```

**Features**:
- Sentence-level translation for accuracy
- Technical term preservation
- Code-mixing support (Hinglish, etc.)

---

### Stage 3: Validation

**File**: `backend/services/validate/validator.py`

```python
class ValidationModule:
    """NCERT curriculum validation."""
    
    QUALITY_THRESHOLD = 0.80
    
    def validate_content(
        self,
        original_text: str,
        translated_text: str,
        grade_level: int,
        subject: str,
        language: str
    ) -> ValidationResult:
        """
        Validate content quality.
        
        Checks:
        1. Semantic accuracy (≥80%)
        2. NCERT alignment (≥80%)
        3. Script accuracy
        4. Age appropriateness
        """
        ...
```

**Validation Checks**:

| Check | Model | Threshold |
|-------|-------|-----------|
| Semantic Accuracy | BERT embeddings | ≥80% |
| NCERT Alignment | Keyword + Objective matching | ≥80% |
| Script Validation | Unicode checks | Pass/Fail |
| Age Appropriate | Content filtering | Pass/Fail |

---

### Stage 4: Text-to-Speech

**File**: `backend/services/speech/generator.py`

```python
class SpeechGenerator:
    """Multilingual TTS generation."""
    
    def generate(
        self,
        text: str,
        language: str,
        voice: str = 'default'
    ) -> AudioFile:
        """
        Generate audio from text.
        
        Output: MP3, 22050Hz sample rate
        """
        ...
```

**Features**:
- Technical term pronunciation handling
- Language-specific prosody
- Audio normalization
- ASR validation (optional)

---

## Celery Task Integration

### Pipeline Task

```python
# backend/tasks/pipeline_tasks.py

@celery_app.task(
    bind=True,
    queue='pipeline',
    max_retries=3,
    time_limit=1800
)
def process_content_task(
    self,
    text: str,
    target_language: str,
    grade_level: int,
    subject: str,
    user_id: str
) -> dict:
    """Async content processing task."""
    
    orchestrator = ContentPipelineOrchestrator()
    
    result = orchestrator.process_content(
        input_data=text,
        target_language=target_language,
        grade_level=grade_level,
        subject=subject
    )
    
    # Save to database
    save_processed_content(result, user_id)
    
    return result.to_dict()
```

### Task Queues

| Queue | Purpose | Concurrency |
|-------|---------|-------------|
| `pipeline` | Full pipeline | 2 |
| `ocr` | OCR extraction | 1 (GPU) |
| `ml_gpu` | GPU inference | 1 |
| `ml_cpu` | CPU inference | 4 |

---

## Resource Optimization

### Memory Management

```python
# Principle T: OOM Alerting (85% threshold)
if memory_usage > 0.85 * total_memory:
    trigger_alert()
    evict_least_used_model()
```

### Model Loading Strategy

```python
# Lazy loading with LRU eviction
class ModelLoader:
    def get_model(self, model_id: str):
        if model_id not in self._cache:
            if self._cache_full():
                self._evict_lru()
            self._cache[model_id] = self._load_model(model_id)
        return self._cache[model_id]
```

### vLLM Configuration

```python
# Production serving settings
VLLM_GPU_MEMORY_UTILIZATION = 0.90
VLLM_SWAP_SPACE_GB = 4
VLLM_ENABLE_PREFIX_CACHING = True
VLLM_BLOCK_SIZE = 16
```

---

## Error Handling

### Pipeline Errors

```python
class PipelineValidationError(Exception):
    """Invalid input parameters."""
    pass

class PipelineStageError(Exception):
    """Stage failed after retries."""
    pass
```

### Retry Strategy

| Attempt | Backoff | Total Wait |
|---------|---------|------------|
| 1 | 0s | 0s |
| 2 | 2s | 2s |
| 3 | 4s | 6s |

### Graceful Degradation

```python
# Circuit breaker for external services
@circuit_breaker(failure_threshold=3, recovery_timeout=60)
async def call_bhashini_fallback():
    """Fallback to Bhashini API if local TTS fails."""
    ...
```

---

## Caching

### Redis Caching

```python
# Cache processed content
cache_key = f"content:{hash(text)}:{language}:{grade}"
cached = redis.get(cache_key)
if cached:
    return cached

result = process_content(...)
redis.setex(cache_key, 3600, result)  # 1 hour TTL
```

### Model KV Cache

```python
# vLLM prefix caching
VLLM_ENABLE_PREFIX_CACHING = True  # Reuse KV cache for similar prompts
```

---

## Metrics & Monitoring

### Prometheus Metrics

```python
# Inference latency
ssetu_inference_latency_seconds{model="llama", stage="simplify"}

# Pipeline throughput
ssetu_pipeline_requests_total{status="success"}

# Stage timing
ssetu_stage_duration_seconds{stage="translation"}
```

### Logging

```python
# Stage execution logging
logger.info(f"Stage {stage.value} completed in {time_ms}ms")
logger.error(f"Stage {stage.value} failed: {error}")
```

---

## Input/Output Formats

### Input

```python
@dataclass
class PipelineInput:
    text: str                    # Raw or OCR-extracted text
    target_language: str         # Target Indian language
    grade_level: int             # 5-12
    subject: str                 # Mathematics, Science, etc.
    output_format: str = 'both'  # text | audio | both
```

### Output

```python
@dataclass
class ProcessedContentResult:
    id: str
    original_text: str
    simplified_text: str
    translated_text: str
    language: str
    grade_level: int
    subject: str
    audio_file_path: Optional[str]
    ncert_alignment_score: float
    audio_accuracy_score: Optional[float]
    validation_status: str       # passed | failed | needs_review
    created_at: datetime
    metadata: Dict[str, Any]
    metrics: List[StageMetrics]
```

---

## Local vs Production

| Aspect | Local (M4) | Production (GPU) |
|--------|------------|------------------|
| Quantization | INT4 | AWQ |
| vLLM | Disabled | Enabled |
| Batch Size | 1 | 8-16 |
| Concurrency | 1 | 4 |
| Memory | 10GB | 16GB+ |
| Throughput | ~10 req/min | ~100 req/min |

---

⸻

Created by: **K Dhiraj**  
Email: kdhiraj152@gmail.com  
GitHub: [github.com/KDhiraj152](https://github.com/KDhiraj152)  
LinkedIn: [linkedin.com/in/kdhiraj152](https://linkedin.com/in/kdhiraj152)
