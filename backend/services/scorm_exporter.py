"""
SCORM/xAPI LMS Export Service

Generate SCORM 1.2/2004 packages and xAPI statements for
LMS integration (Moodle, Canvas, Blackboard).
"""
import uuid
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
import json

from backend.models import ProcessedContent, User
from backend.utils.logging import get_logger

logger = get_logger(__name__)


class SCORMExporter:
    """
    SCORM 1.2/2004 package generator.
    
    Generates manifest, content structure, and metadata for LMS deployment.
    """
    
    def __init__(self, version: str = "1.2"):
        """
        Initialize SCORM exporter.
        
        Args:
            version: SCORM version ("1.2" or "2004")
        """
        self.version = version
        self.output_dir = Path("data/scorm_packages")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_content(
        self,
        content: ProcessedContent,
        title: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Export content as SCORM package.
        
        Args:
            content: Content to export
            title: Course title
            description: Course description
            metadata: Additional metadata
        
        Returns:
            Path to generated .zip package
        """
        package_id = str(uuid.uuid4())
        package_dir = self.output_dir / package_id
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate manifest
        manifest = self._generate_manifest(
            package_id, title, description, metadata
        )
        manifest_path = package_dir / "imsmanifest.xml"
        manifest.write(manifest_path, encoding='utf-8', xml_declaration=True)
        
        # Generate content HTML
        html_content = self._generate_content_html(content, title)
        content_path = package_dir / "content.html"
        content_path.write_text(html_content, encoding='utf-8')
        
        # Copy SCORM API wrapper
        self._copy_scorm_api(package_dir)
        
        # Create ZIP package
        zip_path = self.output_dir / f"{package_id}.zip"
        self._create_zip(package_dir, zip_path)
        
        logger.info(f"Generated SCORM package: {zip_path}")
        
        return zip_path
    
    def _generate_manifest(
        self,
        package_id: str,
        title: str,
        description: str,
        metadata: Optional[Dict[str, Any]]
    ) -> ET.ElementTree:
        """Generate imsmanifest.xml."""
        # Root element
        manifest = ET.Element('manifest', {
            'identifier': package_id,
            'version': '1.0',
            'xmlns': 'http://www.imsproject.org/xsd/imscp_rootv1p1p2',
            'xmlns:adlcp': 'http://www.adlnet.org/xsd/adlcp_rootv1p2',
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:schemaLocation': 'http://www.imsproject.org/xsd/imscp_rootv1p1p2 imscp_rootv1p1p2.xsd'
        })
        
        # Metadata
        meta = ET.SubElement(manifest, 'metadata')
        schema = ET.SubElement(meta, 'schema')
        schema.text = 'ADL SCORM'
        schemaversion = ET.SubElement(meta, 'schemaversion')
        schemaversion.text = self.version
        
        # Organizations
        orgs = ET.SubElement(manifest, 'organizations', {'default': 'org1'})
        org = ET.SubElement(orgs, 'organization', {'identifier': 'org1'})
        org_title = ET.SubElement(org, 'title')
        org_title.text = title
        
        # Item (learning object)
        item = ET.SubElement(org, 'item', {
            'identifier': 'item1',
            'identifierref': 'resource1'
        })
        item_title = ET.SubElement(item, 'title')
        item_title.text = title
        
        # Resources
        resources = ET.SubElement(manifest, 'resources')
        resource = ET.SubElement(resources, 'resource', {
            'identifier': 'resource1',
            'type': 'webcontent',
            'adlcp:scormtype': 'sco',
            'href': 'content.html'
        })
        
        # Files
        ET.SubElement(resource, 'file', {'href': 'content.html'})
        ET.SubElement(resource, 'file', {'href': 'scorm_api.js'})
        
        return ET.ElementTree(manifest)
    
    def _generate_content_html(self, content: ProcessedContent, title: str) -> str:
        """Generate content HTML with SCORM API wrapper."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="scorm_api.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #333;
        }}
        .content {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
        }}
        .navigation {{
            margin-top: 30px;
            text-align: center;
        }}
        button {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }}
        button:hover {{
            background: #0056b3;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="content">
        {content.simplified_content or content.original_content}
    </div>
    
    <div class="navigation">
        <button onclick="completeLesson()">Complete Lesson</button>
    </div>
    
    <script>
        // Initialize SCORM
        window.addEventListener('load', function() {{
            scormAPI.initialize();
            scormAPI.setValue('cmi.core.lesson_status', 'incomplete');
        }});
        
        function completeLesson() {{
            scormAPI.setValue('cmi.core.lesson_status', 'completed');
            scormAPI.setValue('cmi.core.score.raw', '100');
            scormAPI.commit();
            alert('Lesson completed! You can now close this window.');
        }}
        
        // Commit on page unload
        window.addEventListener('beforeunload', function() {{
            scormAPI.finish();
        }});
    </script>
</body>
</html>
"""
        return html
    
    def _copy_scorm_api(self, package_dir: Path):
        """Copy SCORM API JavaScript wrapper."""
        api_js = """
// SCORM 1.2 API Wrapper
var scormAPI = (function() {
    var API = null;
    
    function findAPI(win) {
        var attempts = 0;
        while (win && attempts < 10) {
            if (win.API) return win.API;
            if (win.parent && win.parent !== win) {
                win = win.parent;
            } else {
                break;
            }
            attempts++;
        }
        return null;
    }
    
    return {
        initialize: function() {
            API = findAPI(window);
            if (API) {
                API.LMSInitialize('');
                return true;
            }
            console.log('SCORM API not found');
            return false;
        },
        
        getValue: function(key) {
            if (API) {
                return API.LMSGetValue(key);
            }
            return '';
        },
        
        setValue: function(key, value) {
            if (API) {
                return API.LMSSetValue(key, value);
            }
            console.log('Set:', key, '=', value);
            return true;
        },
        
        commit: function() {
            if (API) {
                return API.LMSCommit('');
            }
            return true;
        },
        
        finish: function() {
            if (API) {
                this.commit();
                return API.LMSFinish('');
            }
            return true;
        }
    };
})();
"""
        api_path = package_dir / "scorm_api.js"
        api_path.write_text(api_js, encoding='utf-8')
    
    def _create_zip(self, source_dir: Path, zip_path: Path):
        """Create ZIP archive of package."""
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in source_dir.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(source_dir)
                    zipf.write(file, arcname)


class xAPIStatementGenerator:
    """
    Generate xAPI (Tin Can API) statements for learning analytics.
    
    Tracks learner interactions, progress, and achievements.
    """
    
    def __init__(self, lrs_endpoint: Optional[str] = None):
        """
        Initialize xAPI generator.
        
        Args:
            lrs_endpoint: Learning Record Store endpoint URL
        """
        self.lrs_endpoint = lrs_endpoint
    
    def create_statement(
        self,
        actor: User,
        verb: str,
        object_id: str,
        object_name: str,
        result: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create xAPI statement.
        
        Args:
            actor: User performing action
            verb: xAPI verb (viewed, completed, answered, etc.)
            object_id: Content ID
            object_name: Content name
            result: Optional result data (score, duration)
            context: Optional context data
        
        Returns:
            xAPI statement dict
        """
        statement = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor": {
                "objectType": "Agent",
                "name": actor.username,
                "mbox": f"mailto:{actor.email}"
            },
            "verb": self._get_verb_definition(verb),
            "object": {
                "objectType": "Activity",
                "id": object_id,
                "definition": {
                    "name": {"en-US": object_name},
                    "type": "http://adlnet.gov/expapi/activities/lesson"
                }
            }
        }
        
        if result:
            statement["result"] = result
        
        if context:
            statement["context"] = context
        
        return statement
    
    def _get_verb_definition(self, verb: str) -> Dict[str, Any]:
        """Get xAPI verb definition."""
        verbs = {
            "viewed": {
                "id": "http://id.tincanapi.com/verb/viewed",
                "display": {"en-US": "viewed"}
            },
            "completed": {
                "id": "http://adlnet.gov/expapi/verbs/completed",
                "display": {"en-US": "completed"}
            },
            "answered": {
                "id": "http://adlnet.gov/expapi/verbs/answered",
                "display": {"en-US": "answered"}
            },
            "passed": {
                "id": "http://adlnet.gov/expapi/verbs/passed",
                "display": {"en-US": "passed"}
            },
            "failed": {
                "id": "http://adlnet.gov/expapi/verbs/failed",
                "display": {"en-US": "failed"}
            }
        }
        
        return verbs.get(verb, verbs["viewed"])
    
    def send_statement(self, statement: Dict[str, Any]) -> bool:
        """
        Send statement to LRS.
        
        Args:
            statement: xAPI statement
        
        Returns:
            True if successful
        """
        if not self.lrs_endpoint:
            logger.warning("No LRS endpoint configured, statement not sent")
            logger.debug(f"Statement: {json.dumps(statement, indent=2)}")
            return False
        
        # Send xAPI statement to LRS
        try:
            import httpx
            response = httpx.post(
                f\"{self.lrs_endpoint}/statements\",
                json=statement,
                headers={
                    \"Authorization\": self.lrs_auth,
                    \"Content-Type\": \"application/json\",
                    \"X-Experience-API-Version\": \"1.0.3\"
                },
                timeout=10.0
            )
            if response.status_code == 200:
                logger.info(f\"xAPI statement sent successfully: {statement['verb']['display']['en-US']}\")
                return True
            else:
                logger.warning(f\"LRS responded with status {response.status_code}\")\n                return False
        except Exception as e:
            logger.error(f\"Failed to send xAPI statement to LRS: {e}\")
            return False
