# AI / ML Pipeline — Shiksha Setu

Complete guide to the AI/ML pipeline architecture, implementation, and optimization.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Pipeline Stages](#pipeline-stages)
3. [Architecture Components](#architecture-components)
4. [Orchestration & Resilience](#orchestration--resilience)
5. [Optimization](#optimization)
6. [Performance Targets](#performance-targets)

---

## Overview

The Shiksha Setu content pipeline processes educational material through multiple AI/ML stages:

```
Upload → OCR → Simplification → Translation → Validation → TTS → Storage
```

**Key Locations**:
- Pipeline: `backend/pipeline/` and `backend/services/ai/`
- Celery tasks: `backend/tasks/`
- Models: `backend/services/ai/orchestrator.py`

---

## Pipeline Stages

### 1. OCR (Optical Character Recognition)

**Tools**: PyMuPDF (PDF), Tesseract (images)

**Purpose**: Extract text from uploaded documents

**Output**: Cleaned UTF-8 text with Indic script support

### 2. Simplification

**Model**: FLAN-T5 (or cloud HuggingFace endpoint)

**Purpose**: Reduce reading complexity to target grade level (5-12)

**Features**:
- Grade-level targeting
- Readability analysis
- Factual accuracy preservation

### 3. Translation

**Model**: IndicTrans2 (AI4Bharat) or NLLB-200

**Purpose**: Translate to Indian regional languages

**Supported Languages**: Hindi, Tamil, Telugu, Bengali, Marathi, Kannada, Malayalam, Gujarati, Odia, Punjabi

### 4. Validation

**Model**: BERT-based semantic similarity

**Purpose**: Ensure NCERT alignment and factual consistency

**Validation Thresholds**:
- NCERT alignment: ≥80%
- Factual consistency: ≥75%

### 5. Text-to-Speech (TTS)

**Engines**: MMS-TTS, Coqui TTS, or Bhashini API

**Purpose**: Generate natural-sounding multilingual audio

---

## Architecture Components

### Enhanced AI Orchestrator

**Purpose**: Unified pipeline coordination with optimization

**Features**:
- Request coalescing (batching)
- Predictive preloading
- Language-aware routing
- Automatic memory management

### Request Coalescing Engine

**Problem Solved**: Inefficient handling of concurrent identical requests

**Throughput Improvement**: 2-5x under concurrent load

### Language-Aware Model Router

**Problem Solved**: Token expansion inconsistency for Indic scripts

**Script Complexity Factors**:
| Script | Multiplier | Languages |
|--------|-----------|-----------|
| English | 1.0x | English |
| Devanagari | 1.4x | Hindi, Sanskrit, Marathi |
| Tamil | 1.6x | Tamil |
| Telugu | 1.5x | Telugu |
| Malayalam | 1.55x | Malayalam |
| Bengali | 1.35x | Bengali, Assamese |
| Kannada | 1.45x | Kannada |
| Gujarati | 1.3x | Gujarati |
| Odia | 1.35x | Odia |

### Predictive Memory Scheduler

**Purpose**: Reduce cold-start latency via intelligent preloading

**Service Memory Map**:
| Service | Memory | Priority |
|---------|--------|----------|
| Translation | 2.5GB | High |
| Simplification | 2.0GB | High |
| Embeddings | 1.2GB | Medium |
| Validation | 0.5GB | Low |
| TTS | 0.1GB | Low |

### NCERT Knowledge Graph

**Purpose**: Curriculum-aware validation

**Features**:
- Concept prerequisite validation
- Readability analysis (Flesch-Kincaid)
- Bloom's taxonomy detection

**Grade Profiles**:
| Grade | FK Score | Avg Sentence | Vocab Level |
|-------|----------|-------------|-------------|
| 5 | 70-85 | 12 words | Basic |
| 8 | 60-75 | 18 words | Intermediate |
| 10 | 50-65 | 22 words | Advanced |
| 12 | 45-60 | 25 words | Scholarly |

---

## Orchestration & Resilience

### Lifecycle Management

**Startup Phases**:
1. ENVIRONMENT (0-1s): Config, logging
2. INFRASTRUCTURE (1-2s): Database, Redis
3. SERVICES (2-3s): AI models, workers
4. WARMUP (3-4s): Caches, health checks

**Total Startup**: 4-6 seconds

### Retry Logic

**Strategy**: Exponential backoff (2^n)
- Max retries: 3
- Base delay: 1 second
- Max delay: 30 seconds

### Task Queue

**Broker**: Redis

**Queue Types**:
- `default` - Standard tasks
- `pipeline` - Heavy ML processing
- `ocr` - Document extraction

### Error Handling

**Soft-Fail Policy**: Return partial results with quality metrics

---

## Optimization

### Model Caching

- Keep models in GPU memory during active processing
- Lazy load on first use
- Cache duration: 5 minutes inactivity

### Batching

- Batch size: 4-16 (configurable)
- Max wait: 100ms
- GPU utilization: 70-90%

### Quantization

- **FP16** for inference (50% memory savings)
- **INT8** for large models
- **INT4** for 8GB RAM devices

See [Dynamic Quantization](dynamic-quantization.md) for details.

---

## Performance Targets

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Cold Start | 15-20s | 4-6s | ≤5s |
| Translation (100 words) | 800ms | 400ms | ≤500ms |
| Concurrent Throughput | 10 req/s | 40 req/s | 30+ req/s |
| Memory Efficiency | 60% | 85% | ≥80% |

---

## 📚 Related Documentation

- **[Dynamic Quantization](dynamic-quantization.md)** - Model optimization
- **[Security](security.md)** - API security
- **[Optimization](optimization.md)** - General performance tuning
- **[Database](database.md)** - Vector storage

---

## 👨‍💻 Author

**K Dhiraj** • [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com) • [@KDhiraj152](https://github.com/KDhiraj152)

*Last updated: November 2025*
