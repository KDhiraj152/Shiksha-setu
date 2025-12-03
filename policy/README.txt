================================================================================
                    POLICY MODULE - UNRESTRICTED MODE GUIDE
================================================================================

OVERVIEW
--------
This policy module extracts all hardcoded curriculum, safety, and content 
filtering logic into a single configurable system. It enables unrestricted 
local operation while maintaining full auditability.

QUICK START
-----------
1. Enable unrestricted mode:
   
   export ALLOW_UNRESTRICTED_MODE=true
   ./start.sh

2. Verify mode is active (check startup banner):
   
   ⚠️  UNRESTRICTED MODE ⚠️
   All content filters and curriculum enforcement DISABLED

3. Run tests:
   
   ./scripts/test_policy_toggle.sh

ENVIRONMENT VARIABLES
---------------------
+------------------------------+----------+------------------------------------+
| Variable                     | Default  | Description                        |
+------------------------------+----------+------------------------------------+
| ALLOW_UNRESTRICTED_MODE      | false    | Master toggle for unrestricted     |
| ALLOW_EXTERNAL_CALLS         | false    | Allow external API calls           |
| POLICY_FILTERS_ENABLED       | true     | Enable content filtering           |
| POLICY_CURRICULUM_ENABLED    | true     | Enable curriculum validation       |
| POLICY_GRADE_VALIDATION      | true     | Enable grade level validation      |
| POLICY_BLOCK_HARMFUL         | true     | Block harmful content              |
| POLICY_FILTER_PII            | true     | Filter PII from responses          |
| POLICY_FILTER_SECRETS        | true     | Filter secrets/credentials         |
+------------------------------+----------+------------------------------------+

MODE BEHAVIORS
--------------
RESTRICTED MODE (default):
  - All safety filters active
  - Curriculum validation enforced
  - Grade level checking enabled
  - Harmful content blocked
  - PII/secrets filtered
  - External calls blocked

UNRESTRICTED MODE:
  - All safety filters bypassed
  - No curriculum validation
  - No grade level checking
  - Harmful content passes through
  - PII/secrets not filtered
  - External calls still blocked (unless explicitly enabled)
  
  ⚠️  ALL bypassed content is LOGGED for audit purposes

LOCAL-ONLY ENFORCEMENT
----------------------
By default, ALLOW_EXTERNAL_CALLS=false ensures:
  - No API calls to external services
  - All inference runs on local MLX engine
  - Network isolation maintained
  - External attempts logged

AUDIT TRAIL
-----------
All policy decisions are logged:
  - Startup mode printed to stdout
  - Content that would be filtered (with reason)
  - External call attempts
  - Policy configuration at startup

Log locations:
  - stdout (structured JSON)
  - logs/ directory

BACKUP LOCATIONS
----------------
Original files before modification:
  ./.backup/20251204_025122/safety.py.bak
  ./.backup/20251204_025122/engine.py.bak
  ./.backup/20251204_025122/curriculum_validator.py.bak
  ./.backup/20251204_025122/main.py.bak
  ./.backup/20251204_025122/grade_adaptation.py.bak
  ./.backup/20251204_025122/ncert.py.bak

ROLLBACK PROCEDURE
------------------
1. Stop services:
   ./stop.sh

2. Restore original files (remove .bak extension):
   cp .backup/20251204_025122/safety.py.bak backend/services/ai_core/safety.py
   cp .backup/20251204_025122/engine.py.bak backend/services/ai_core/engine.py
   cp .backup/20251204_025122/curriculum_validator.py.bak backend/services/curriculum_validator.py
   cp .backup/20251204_025122/main.py.bak backend/api/main.py

3. Remove policy module:
   rm -rf backend/policy/

4. Restart:
   ./start.sh

FILES REFERENCE
---------------
Policy Module:
  backend/policy/__init__.py       - Package exports
  backend/policy/policy_module.py  - Core policy engine

Configuration:
  policy/config.default.json       - Default policy config
  policy/audit_manifest.json       - Refactoring manifest
  policy/README.txt                - This file

Test Scripts:
  scripts/test_policy_toggle.sh    - Unit tests for toggle
  scripts/smoke_unrestricted.sh    - Smoke test

IMPLICATIONS
------------
⚠️  UNRESTRICTED MODE WARNING:
  - No content filtering means potentially harmful responses
  - No curriculum validation means off-topic content possible
  - No grade adaptation means content may not be age-appropriate
  - All actions are logged but NOT prevented

RECOMMENDED USE CASES:
  - Local development and testing
  - Research and experimentation
  - Adult-only local deployments
  - Debugging content filter issues

NOT RECOMMENDED FOR:
  - Production educational deployments
  - Student-facing systems
  - Unsupervised access

SUPPORT
-------
Audit Manifest: policy/audit_manifest.json
Changelog: CHANGELOG_UNLOCK.md
Backup Timestamp: 20251204_025122

================================================================================
                         END OF POLICY MODULE GUIDE
================================================================================
