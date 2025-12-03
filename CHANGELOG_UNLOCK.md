# CHANGELOG: UNLOCK - Policy Module Consolidation

**Date:** 2024-12-04  
**Version:** 1.0.0  
**Author:** Autonomous Agent  
**Purpose:** Extract hardcoded alignments into configurable policy module for local-first unrestricted AI engine

---

## Summary

This changelog documents the consolidation of all hardcoded curriculum, safety, and content filtering logic from ShikshaSetu's multimodal RAG system into a single, auditable policy module. The goal is to enable unrestricted local operation while maintaining full auditability.

---

## Changes Made

### 1. New Policy Module Created

**Location:** `backend/policy/`

- `__init__.py` - Package exports
- `policy_module.py` - Core policy engine (250+ lines)
  - `PolicyConfig` dataclass with all configuration options
  - `PolicyModule` class with centralized policy decisions
  - Environment variable integration
  - Startup mode banner

**Configuration:** `policy/config.default.json`
- Comprehensive JSON configuration with all policy flags
- Documented defaults for each setting

### 2. Files Refactored

| File | Changes |
|------|---------|
| `backend/services/ai_core/safety.py` | `check_input()` and `filter_response()` now respect policy module |
| `backend/services/ai_core/engine.py` | `_ensure_initialized()` loads policy, safety checks use policy |
| `backend/services/curriculum_validator.py` | `validate_grade_level()` checks policy before validation |
| `backend/api/main.py` | Startup event prints policy mode banner |

### 3. Backups Created

**Location:** `backups/20251204_025122/`

All original files backed up before modification:
- `safety.py`
- `engine.py`
- `curriculum_validator.py`
- `main.py`
- `grade_adaptation.py`
- `prompts.py`

### 4. Test Scripts Created

| Script | Purpose |
|--------|---------|
| `scripts/test_policy_toggle.sh` | Automated unit tests for policy toggle |
| `scripts/smoke_unrestricted.sh` | Smoke test for unrestricted mode |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOW_UNRESTRICTED_MODE` | `false` | Master toggle for unrestricted mode |
| `ALLOW_EXTERNAL_CALLS` | `false` | Local-only enforcement |
| `POLICY_FILTERS_ENABLED` | `true` | Content filtering toggle |
| `POLICY_CURRICULUM_ENABLED` | `true` | Curriculum validation toggle |
| `POLICY_BLOCK_HARMFUL` | `true` | Harmful content blocking |

---

## Usage

### Enable Unrestricted Mode

```bash
export ALLOW_UNRESTRICTED_MODE=true
./start.sh
```

### Verify Mode

Look for startup banner:
```
╔════════════════════════════════════════════════════════════════╗
║                    ⚠️  UNRESTRICTED MODE ⚠️                      ║
║  All content filters and curriculum enforcement DISABLED       ║
║  External calls: BLOCKED (local-only)                          ║
║  This session is being logged for audit purposes               ║
╚════════════════════════════════════════════════════════════════╝
```

### Run Tests

```bash
./scripts/test_policy_toggle.sh
./scripts/smoke_unrestricted.sh
```

---

## Audit Trail

All policy decisions are logged:
- Startup mode banner (stdout)
- Content that would be filtered (with reason)
- External call attempts (blocked or allowed)
- Policy configuration loaded

---

## Rollback

```bash
./stop.sh
cp backups/20251204_025122/*.py backend/services/ai_core/
rm -rf backend/policy/
./start.sh
```

---

## Verification Checklist

- [ ] Policy module loads without errors
- [ ] Restricted mode blocks harmful content
- [ ] Unrestricted mode bypasses filters
- [ ] External calls blocked by default
- [ ] Startup banner displays correctly
- [ ] All filtering logged for audit

---

## Files Changed

```
backend/policy/__init__.py          (NEW)
backend/policy/policy_module.py     (NEW)
policy/config.default.json          (NEW)
policy/audit_manifest.json          (NEW)
policy/README.txt                   (NEW)
scripts/test_policy_toggle.sh       (NEW)
scripts/smoke_unrestricted.sh       (NEW)
backups/20251204_025122/            (NEW - backups)
backend/services/ai_core/safety.py  (MODIFIED)
backend/services/ai_core/engine.py  (MODIFIED)
backend/services/curriculum_validator.py (MODIFIED)
backend/api/main.py                 (MODIFIED)
```

---

**UNLOCK: Hardcoded alignments extracted. Local-first unrestricted engine ready.**
