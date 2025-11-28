"""
Cultural Context Service

Issue: CODE-REVIEW-GPT #12 (HIGH)
Problem: No cultural adaptation for regional content

Solution: Region-specific content filtering and cultural sensitivity checks
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from enum import Enum
import logging
import re

from sqlalchemy.orm import Session

from ..models import ProcessedContent
from ..core.config import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class IndianRegion(str, Enum):
    """Indian regions for cultural context."""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    NORTHEAST = "northeast"
    CENTRAL = "central"


class CulturalSensitivityLevel(str, Enum):
    """Sensitivity levels for content."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CulturalContextService:
    """Service for cultural context adaptation and sensitivity checking."""
    
    # Regional preferences mapping
    REGIONAL_PREFERENCES = {
        IndianRegion.NORTH: {
            "languages": ["hi", "pa", "ur"],
            "festivals": ["diwali", "holi", "lohri", "baisakhi"],
            "food_examples": ["roti", "paratha", "paneer", "lassi"],
            "cultural_references": ["punjab", "delhi", "rajasthan", "haryana"],
            "avoid_topics": []
        },
        IndianRegion.SOUTH: {
            "languages": ["ta", "te", "kn", "ml"],
            "festivals": ["pongal", "onam", "ugadi", "vishu"],
            "food_examples": ["idli", "dosa", "sambar", "rasam"],
            "cultural_references": ["tamil nadu", "kerala", "karnataka", "andhra"],
            "avoid_topics": []
        },
        IndianRegion.EAST: {
            "languages": ["bn", "or", "as"],
            "festivals": ["durga puja", "rath yatra", "bihu", "pohela boishakh"],
            "food_examples": ["mishti", "rosogolla", "luchi", "maacher jhol"],
            "cultural_references": ["bengal", "odisha", "assam"],
            "avoid_topics": []
        },
        IndianRegion.WEST: {
            "languages": ["gu", "mr"],
            "festivals": ["ganesh chaturthi", "navratri", "gudi padwa"],
            "food_examples": ["dhokla", "pav bhaji", "vada pav", "modak"],
            "cultural_references": ["gujarat", "maharashtra", "goa"],
            "avoid_topics": []
        },
        IndianRegion.NORTHEAST: {
            "languages": ["as", "mni", "grt"],
            "festivals": ["bihu", "losar", "sangai", "mopin"],
            "food_examples": ["momos", "thukpa", "bamboo shoot", "smoked meat"],
            "cultural_references": ["assam", "manipur", "arunachal", "nagaland"],
            "avoid_topics": []
        },
        IndianRegion.CENTRAL: {
            "languages": ["hi"],
            "festivals": ["diwali", "holi", "teej"],
            "food_examples": ["dal bafla", "poha", "bhutte ka kees"],
            "cultural_references": ["madhya pradesh", "chhattisgarh"],
            "avoid_topics": []
        }
    }
    
    # Culturally sensitive topics to handle carefully
    SENSITIVE_TOPICS = {
        "religion": {
            "keywords": ["hindu", "muslim", "christian", "sikh", "buddhist", "jain", 
                        "temple", "mosque", "church", "gurudwara", "monastery"],
            "sensitivity": CulturalSensitivityLevel.HIGH,
            "guidelines": "Present facts objectively, respect all religions equally"
        },
        "caste": {
            "keywords": ["caste", "brahmin", "dalit", "schedule", "reservation"],
            "sensitivity": CulturalSensitivityLevel.HIGH,
            "guidelines": "Avoid reinforcing stereotypes, focus on equality"
        },
        "gender": {
            "keywords": ["boy", "girl", "man", "woman", "male", "female"],
            "sensitivity": CulturalSensitivityLevel.MEDIUM,
            "guidelines": "Avoid gender stereotypes, show equal representation"
        },
        "food_habits": {
            "keywords": ["vegetarian", "non-vegetarian", "beef", "pork", "meat"],
            "sensitivity": CulturalSensitivityLevel.MEDIUM,
            "guidelines": "Respect dietary preferences, avoid judgment"
        },
        "regional_identity": {
            "keywords": ["south indian", "north indian", "regional", "state"],
            "sensitivity": CulturalSensitivityLevel.MEDIUM,
            "guidelines": "Celebrate diversity, avoid stereotypes"
        }
    }
    
    # Universal Indian cultural values
    UNIVERSAL_VALUES = [
        "respect for elders",
        "family unity",
        "education importance",
        "hard work",
        "honesty",
        "compassion",
        "diversity appreciation"
    ]
    
    def __init__(self, db: Session):
        self.db = db
    
    def adapt_content_for_region(
        self,
        text: str,
        region: IndianRegion,
        language: str
    ) -> Dict[str, Any]:
        """
        Adapt content for specific Indian region.
        
        Args:
            text: Content text
            region: Target region
            language: Content language
            
        Returns:
            Dictionary with adapted content and metadata
        """
        logger.info(f"Adapting content for region: {region}")
        
        regional_prefs = self.REGIONAL_PREFERENCES.get(region, {})
        
        # Check if content includes regional references
        regional_refs = self._find_regional_references(text)
        
        # Suggest region-specific examples
        suggestions = self._generate_regional_suggestions(text, region)
        
        # Check cultural sensitivity
        sensitivity_issues = self.check_cultural_sensitivity(text)
        
        result = {
            "original_text": text,
            "region": region,
            "language": language,
            "regional_references_found": regional_refs,
            "regional_preferences": {
                "languages": regional_prefs.get("languages", []),
                "festivals": regional_prefs.get("festivals", []),
                "food_examples": regional_prefs.get("food_examples", [])
            },
            "adaptation_suggestions": suggestions,
            "cultural_sensitivity": sensitivity_issues,
            "needs_review": len(sensitivity_issues) > 0
        }
        
        logger.info(
            f"Adaptation complete: {len(regional_refs)} refs found, "
            f"{len(suggestions)} suggestions, {len(sensitivity_issues)} sensitivity issues"
        )
        
        return result
    
    def check_cultural_sensitivity(self, text: str) -> List[Dict[str, Any]]:
        """
        Check content for cultural sensitivity issues.
        
        Args:
            text: Content text to check
            
        Returns:
            List of sensitivity issues found
        """
        issues = []
        text_lower = text.lower()
        
        for topic, config in self.SENSITIVE_TOPICS.items():
            # Check if any keywords are present
            found_keywords = [
                kw for kw in config["keywords"] 
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
            ]
            
            if found_keywords:
                issues.append({
                    "topic": topic,
                    "sensitivity_level": config["sensitivity"],
                    "keywords_found": found_keywords,
                    "guidelines": config["guidelines"],
                    "line_numbers": self._find_keyword_lines(text, found_keywords)
                })
        
        return issues
    
    def _find_regional_references(self, text: str) -> List[str]:
        """Find regional references in text."""
        references = []
        text_lower = text.lower()
        
        for region, prefs in self.REGIONAL_PREFERENCES.items():
            # Check festivals
            for festival in prefs.get("festivals", []):
                if festival in text_lower:
                    references.append(f"{festival} ({region})")
            
            # Check food examples
            for food in prefs.get("food_examples", []):
                if food in text_lower:
                    references.append(f"{food} ({region})")
            
            # Check cultural references
            for ref in prefs.get("cultural_references", []):
                if ref in text_lower:
                    references.append(f"{ref} ({region})")
        
        return list(set(references))
    
    def _generate_regional_suggestions(
        self,
        text: str,
        target_region: IndianRegion
    ) -> List[Dict[str, str]]:
        """Generate region-specific content suggestions."""
        suggestions = []
        regional_prefs = self.REGIONAL_PREFERENCES.get(target_region, {})
        
        # Suggest regional festivals
        if "festival" in text.lower():
            suggestions.append({
                "type": "festival_example",
                "suggestion": f"Consider using festivals from {target_region}: "
                             f"{', '.join(regional_prefs.get('festivals', [])[:3])}",
                "reason": "Region-specific festivals increase relatability"
            })
        
        # Suggest regional food examples
        if any(word in text.lower() for word in ["food", "eat", "meal", "dish"]):
            suggestions.append({
                "type": "food_example",
                "suggestion": f"Consider using food examples from {target_region}: "
                             f"{', '.join(regional_prefs.get('food_examples', [])[:3])}",
                "reason": "Familiar food examples enhance understanding"
            })
        
        # Suggest local language integration
        if len(text.split()) > 50:  # Only for substantial content
            suggestions.append({
                "type": "language_integration",
                "suggestion": f"Consider adding key terms in regional languages: "
                             f"{', '.join(regional_prefs.get('languages', []))}",
                "reason": "Bilingual content improves comprehension"
            })
        
        return suggestions
    
    def _find_keyword_lines(self, text: str, keywords: List[str]) -> List[int]:
        """Find line numbers where keywords appear."""
        lines = text.split('\n')
        line_numbers = []
        
        for i, line in enumerate(lines, start=1):
            line_lower = line.lower()
            if any(kw in line_lower for kw in keywords):
                line_numbers.append(i)
        
        return line_numbers
    
    def validate_inclusivity(self, text: str) -> Dict[str, Any]:
        """
        Validate content for inclusivity and diversity representation.
        
        Args:
            text: Content text
            
        Returns:
            Inclusivity analysis
        """
        logger.info("Validating content inclusivity")
        
        issues = []
        recommendations = []
        
        # Check for gender balance
        male_terms = len(re.findall(r'\b(he|him|his|boy|man|male)\b', text, re.IGNORECASE))
        female_terms = len(re.findall(r'\b(she|her|hers|girl|woman|female)\b', text, re.IGNORECASE))
        
        if male_terms > 0 or female_terms > 0:
            ratio = male_terms / (female_terms + 1)  # Avoid division by zero
            if ratio > 3 or ratio < 0.33:
                issues.append({
                    "type": "gender_imbalance",
                    "severity": "medium",
                    "details": f"Male references: {male_terms}, Female references: {female_terms}",
                    "recommendation": "Balance gender representation in examples"
                })
        
        # Check for diverse name representation
        common_names = re.findall(r'\b[A-Z][a-z]+\b', text)
        if len(common_names) > 3:
            recommendations.append({
                "type": "name_diversity",
                "suggestion": "Use names from different regions and communities",
                "examples": ["Priya", "Mohammed", "Fatima", "Arjun", "Lakshmi", "Singh", "Patel"]
            })
        
        # Check for regional diversity
        regional_refs = self._find_regional_references(text)
        if len(regional_refs) < 2 and len(text.split()) > 100:
            recommendations.append({
                "type": "regional_diversity",
                "suggestion": "Include examples from multiple Indian regions",
                "reason": "Promotes national integration and relatability"
            })
        
        # Check for stereotype language
        stereotype_patterns = [
            (r'\bgirls? (?:are|should|must) (?:good at|better at)', "gender_stereotype"),
            (r'\bboys? (?:are|should|must) (?:good at|better at)', "gender_stereotype"),
            (r'\b(?:all|every) [A-Z][a-z]+s? (?:are|do)', "generalization"),
        ]
        
        for pattern, stereotype_type in stereotype_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                issues.append({
                    "type": stereotype_type,
                    "severity": "high",
                    "matches": matches,
                    "recommendation": "Avoid stereotypical generalizations"
                })
        
        inclusivity_score = 1.0 - (len(issues) * 0.1)
        inclusivity_score = max(0.0, min(1.0, inclusivity_score))
        
        return {
            "inclusivity_score": inclusivity_score,
            "issues": issues,
            "recommendations": recommendations,
            "passed": inclusivity_score >= 0.7,
            "analysis_summary": {
                "gender_balance": "balanced" if abs(ratio - 1) < 0.5 else "imbalanced",
                "regional_diversity": "good" if len(regional_refs) >= 2 else "needs_improvement",
                "stereotype_free": len([i for i in issues if "stereotype" in i["type"]]) == 0
            }
        }
    
    def get_universal_examples(self, topic: str) -> List[str]:
        """
        Get culturally universal examples for a topic.
        
        Args:
            topic: Topic to get examples for
            
        Returns:
            List of universal examples
        """
        universal_examples = {
            "mathematics": [
                "counting rupees and paise",
                "measuring ingredients for cooking",
                "calculating train journey time",
                "dividing sweets among friends"
            ],
            "science": [
                "seasons in India (monsoon, summer, winter)",
                "common plants (neem, tulsi, mango)",
                "local animals (cow, peacock, elephant)",
                "everyday phenomena (evaporation of water)"
            ],
            "language": [
                "writing a letter to family",
                "reading a story book",
                "describing your school",
                "talking about your favorite festival"
            ],
            "social_studies": [
                "Indian national symbols",
                "unity in diversity",
                "fundamental rights",
                "Indian constitution"
            ]
        }
        
        return universal_examples.get(topic.lower(), [])
    
    def adapt_for_multilingual_audience(
        self,
        text: str,
        primary_language: str,
        secondary_languages: List[str]
    ) -> Dict[str, Any]:
        """
        Adapt content for multilingual audience.
        
        Args:
            text: Content text
            primary_language: Primary language code
            secondary_languages: List of secondary language codes
            
        Returns:
            Adaptation recommendations
        """
        recommendations = []
        
        # Identify technical terms that should be glossed
        technical_terms = self._identify_technical_terms(text)
        
        if technical_terms:
            recommendations.append({
                "type": "glossary",
                "terms": technical_terms[:10],
                "suggestion": "Provide translations/explanations for technical terms",
                "languages": secondary_languages
            })
        
        # Suggest code-switching opportunities
        if len(text.split()) > 50:
            recommendations.append({
                "type": "code_switching",
                "suggestion": "Use familiar words from local languages for common concepts",
                "example": "Use 'गुरु' (guru) for teacher, 'पानी' (paani) for water"
            })
        
        # Check if idioms are used
        if any(phrase in text.lower() for phrase in ["piece of cake", "break the ice", "hit the nail"]):
            recommendations.append({
                "type": "idiom_localization",
                "suggestion": "Replace English idioms with Indian equivalents",
                "examples": [
                    "English: 'piece of cake' → Hindi: 'बाएं हाथ का खेल' (left hand's play)",
                    "English: 'early bird' → Hindi: 'जल्दी उठने वाला' (early riser)"
                ]
            })
        
        return {
            "primary_language": primary_language,
            "secondary_languages": secondary_languages,
            "recommendations": recommendations,
            "multilingual_friendly": len(recommendations) > 0
        }
    
    def _identify_technical_terms(self, text: str) -> List[str]:
        """Identify technical terms that may need glossary."""
        # Simple heuristic: words longer than 10 characters or in title case in middle of sentence
        words = text.split()
        technical_terms = []
        
        for i, word in enumerate(words):
            # Clean word
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Check if technical (long, capitalized mid-sentence, etc.)
            if (len(clean_word) > 10 or 
                (i > 0 and clean_word[0].isupper() and words[i-1][-1] not in '.!?')):
                if clean_word.lower() not in ['mathematics', 'science', 'english', 'hindi']:
                    technical_terms.append(clean_word)
        
        return list(set(technical_terms))[:20]  # Limit to 20 unique terms


# Integration function for pipeline
async def apply_cultural_context(
    db: Session,
    content: ProcessedContent,
    text: str,
    region: Optional[IndianRegion] = None
) -> Dict[str, Any]:
    """
    Apply cultural context checks to content in pipeline.
    
    Args:
        db: Database session
        content: ProcessedContent object
        text: Content text
        region: Optional target region
        
    Returns:
        Cultural context analysis results
    """
    service = CulturalContextService(db)
    
    results = {}
    
    # Check cultural sensitivity
    results["sensitivity"] = service.check_cultural_sensitivity(text)
    
    # Validate inclusivity
    results["inclusivity"] = service.validate_inclusivity(text)
    
    # Regional adaptation (if region specified)
    if region:
        results["regional_adaptation"] = service.adapt_content_for_region(
            text, region, content.language
        )
    
    # Universal examples
    results["universal_examples"] = service.get_universal_examples(content.subject)
    
    # Overall cultural appropriateness score
    inclusivity_score = results["inclusivity"]["inclusivity_score"]
    sensitivity_penalty = len(results["sensitivity"]) * 0.1
    cultural_score = max(0.0, inclusivity_score - sensitivity_penalty)
    
    results["cultural_appropriateness_score"] = cultural_score
    results["passed"] = cultural_score >= 0.7
    
    return results
