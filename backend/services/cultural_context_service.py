"""
Cultural Context Service for Indian Education.

This service injects region-specific examples, festivals, local stories, and
culturally relevant content to make learning more relatable for Indian students.
"""
import json
import random
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class Region(Enum):
    """Indian geographical regions."""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    GENERAL = "general"


class Subject(Enum):
    """Educational subjects."""
    MATHEMATICS = "mathematics"
    SCIENCE = "science"
    SOCIAL_STUDIES = "social_studies"
    HISTORY = "history"
    GEOGRAPHY = "geography"
    ENGLISH = "english"


@dataclass
class CulturalExample:
    """Represents a culturally relevant example."""
    topic: str
    example: str
    region: Region
    subject: Subject
    grade_level: Optional[int] = None
    keywords: Optional[List[str]] = None


@dataclass
class ContextualContent:
    """Content enhanced with cultural context."""
    original_text: str
    enhanced_text: str
    examples_added: List[CulturalExample]
    festivals_mentioned: List[str]
    local_references: List[str]
    metadata: Optional[Dict[str, Any]] = None


class CulturalContextDatabase:
    """
    Database of cultural context for different Indian regions.
    """
    
    def __init__(self, data_dir: str = "data/cultural_context"):
        """
        Initialize cultural context database.
        
        Args:
            data_dir: Directory containing cultural context JSON files
        """
        self.data_dir = Path(data_dir)
        self.context_data: Dict[str, Any] = {}
        self._load_context_data()
        logger.info(f"CulturalContextDatabase loaded from {data_dir}")
    
    def _load_context_data(self):
        """Load cultural context data from JSON files."""
        # Load main Indian context file
        indian_context_file = self.data_dir / "indian_context.json"
        if indian_context_file.exists():
            with open(indian_context_file, 'r', encoding='utf-8') as f:
                self.context_data = json.load(f)
                logger.debug(f"Loaded context for {len(self.context_data.get('regions', {}))} regions")
        else:
            logger.warning(f"Cultural context file not found: {indian_context_file}")
            self.context_data = self._get_default_context()
    
    def _get_default_context(self) -> Dict[str, Any]:
        """Get minimal default context if files are missing."""
        return {
            "regions": {},
            "general": {
                "national_symbols": {},
                "freedom_fighters": [],
                "scientists": [],
                "traditional_stories": []
            },
            "examples_by_subject": {}
        }
    
    def get_regional_info(self, region: Region) -> Dict[str, Any]:
        """
        Get regional information.
        
        Args:
            region: Region enum
        
        Returns:
            Dictionary with regional data
        """
        return self.context_data.get("regions", {}).get(region.value, {})
    
    def get_festivals(self, region: Optional[Region] = None) -> List[Dict[str, Any]]:
        """
        Get festivals for a region or all festivals.
        
        Args:
            region: Optional region filter
        
        Returns:
            List of festival dictionaries
        """
        if region and region != Region.GENERAL:
            regional_data = self.get_regional_info(region)
            return regional_data.get("festivals", [])
        
        # Get all festivals from all regions
        all_festivals = []
        for region_data in self.context_data.get("regions", {}).values():
            all_festivals.extend(region_data.get("festivals", []))
        
        return all_festivals
    
    def get_examples_by_subject(
        self,
        subject: Subject,
        topic: Optional[str] = None
    ) -> List[str]:
        """
        Get culturally relevant examples for a subject.
        
        Args:
            subject: Subject area
            topic: Optional specific topic
        
        Returns:
            List of example strings
        """
        examples_data = self.context_data.get("examples_by_subject", {})
        subject_examples = examples_data.get(subject.value, {})
        
        if topic:
            return subject_examples.get(topic, [])
        
        # Return all examples for the subject
        all_examples = []
        for examples_list in subject_examples.values():
            all_examples.extend(examples_list)
        
        return all_examples
    
    def get_famous_places(self, region: Optional[Region] = None) -> List[str]:
        """Get famous places for a region."""
        if region and region != Region.GENERAL:
            regional_data = self.get_regional_info(region)
            return regional_data.get("famous_places", [])
        
        # Get all famous places
        all_places = []
        for region_data in self.context_data.get("regions", {}).values():
            all_places.extend(region_data.get("famous_places", []))
        
        return all_places
    
    def get_traditional_foods(self, region: Optional[Region] = None) -> List[str]:
        """Get traditional foods for a region."""
        if region and region != Region.GENERAL:
            regional_data = self.get_regional_info(region)
            return regional_data.get("foods", [])
        
        # Get all foods
        all_foods = []
        for region_data in self.context_data.get("regions", {}).values():
            all_foods.extend(region_data.get("foods", []))
        
        return all_foods
    
    def get_traditional_stories(self) -> List[Dict[str, Any]]:
        """Get traditional Indian stories."""
        return self.context_data.get("general", {}).get("traditional_stories", [])
    
    def get_national_symbols(self) -> Dict[str, str]:
        """Get Indian national symbols."""
        return self.context_data.get("general", {}).get("national_symbols", {})
    
    def get_freedom_fighters(self) -> List[Dict[str, Any]]:
        """Get information about Indian freedom fighters."""
        return self.context_data.get("general", {}).get("freedom_fighters", [])
    
    def get_indian_scientists(self) -> List[Dict[str, Any]]:
        """Get information about Indian scientists."""
        return self.context_data.get("general", {}).get("scientists", [])


class CulturalContextService:
    """
    Service to inject cultural context into educational content.
    
    Features:
    - Add region-specific examples
    - Reference local festivals and traditions
    - Use familiar places and foods in examples
    - Incorporate Indian scientists and freedom fighters
    - Make abstract concepts concrete with local context
    """
    
    def __init__(
        self,
        context_db: Optional[CulturalContextDatabase] = None,
        default_region: Region = Region.GENERAL
    ):
        """
        Initialize the cultural context service.
        
        Args:
            context_db: Optional context database instance
            default_region: Default region for context
        """
        self.context_db = context_db or CulturalContextDatabase()
        self.default_region = default_region
        logger.info(f"CulturalContextService initialized with region: {default_region.value}")
    
    def _enhance_with_examples(
        self,
        content: str,
        subject: Subject,
        topic: Optional[str],
        region: Region,
        grade_level: Optional[int],
        max_examples: int
    ) -> tuple[str, list]:
        """Add culturally relevant examples to content."""
        enhanced_text = content
        examples_added = []
        
        relevant_examples = self._get_relevant_examples(subject, topic)
        
        for example_data in relevant_examples[:max_examples]:
            example_obj = CulturalExample(
                topic=example_data.get('topic', topic or 'general'),
                example=example_data['example'],
                region=region,
                subject=subject,
                grade_level=grade_level
            )
            examples_added.append(example_obj)
            enhanced_text = self._inject_example(enhanced_text, example_data['example'])
        
        return enhanced_text, examples_added
    
    def _enhance_with_festival(
        self,
        content: str,
        region: Region
    ) -> tuple[str, list, list]:
        """Add festival references when relevant."""
        festivals_mentioned = []
        local_references = []
        
        if self._should_add_festival_reference(content):
            festival = self._get_relevant_festival(region, content)
            if festival:
                festivals_mentioned.append(festival['name'])
                content = self._add_festival_reference(content, festival)
                local_references.append(f"Festival: {festival['name']}")
        
        return content, festivals_mentioned, local_references
    
    def _enhance_with_place(
        self,
        content: str,
        region: Region,
        local_references: list
    ) -> tuple[str, list]:
        """Add place references when relevant."""
        if self._should_add_place_reference(content):
            place = self._get_relevant_place(region)
            if place:
                content = self._add_place_reference(content, place)
                local_references.append(f"Place: {place}")
        
        return content, local_references
    
    def _enhance_with_person(
        self,
        content: str,
        subject: Subject,
        local_references: list
    ) -> tuple[str, list]:
        """Add references to famous persons when appropriate."""
        if subject in [Subject.SCIENCE, Subject.HISTORY, Subject.SOCIAL_STUDIES]:
            person = self._get_relevant_person(subject, content)
            if person:
                content = self._add_person_reference(content, person)
                local_references.append(f"Person: {person['name']}")
        
        return content, local_references
    
    def _enhance_with_story(
        self,
        content: str,
        grade_level: Optional[int],
        local_references: list
    ) -> tuple[str, list]:
        """Add traditional stories for younger grades."""
        if grade_level and grade_level <= 8:
            story = self._get_relevant_story(grade_level)
            if story:
                content = self._add_story_reference(content, story)
                local_references.append(f"Story: {story['name']}")
        
        return content, local_references
    
    def enhance_content(
        self,
        content: str,
        subject: Subject,
        topic: Optional[str] = None,
        region: Optional[Region] = None,
        grade_level: Optional[int] = None,
        max_examples: int = 3
    ) -> ContextualContent:
        """
        Enhance content with cultural context.
        
        Args:
            content: Original educational content
            subject: Subject area
            topic: Specific topic (if known)
            region: Region for localized context
            grade_level: Grade level (1-12)
            max_examples: Maximum number of examples to add
        
        Returns:
            ContextualContent with enhanced text and metadata
        """
        if not content or len(content.strip()) == 0:
            raise ValueError("Content cannot be empty")
        
        region = region or self.default_region
        
        logger.info(
            f"Enhancing content for {subject.value}, region: {region.value}, "
            f"grade: {grade_level or 'N/A'}"
        )
        
        # Add subject-specific examples
        enhanced_text, examples_added = self._enhance_with_examples(
            content, subject, topic, region, grade_level, max_examples
        )
        
        # Add festival references
        enhanced_text, festivals_mentioned, local_references = self._enhance_with_festival(
            enhanced_text, region
        )
        
        # Add place references
        enhanced_text, local_references = self._enhance_with_place(
            enhanced_text, region, local_references
        )
        
        # Add person references
        enhanced_text, local_references = self._enhance_with_person(
            enhanced_text, subject, local_references
        )
        
        # Add story references for younger grades
        enhanced_text, local_references = self._enhance_with_story(
            enhanced_text, grade_level, local_references
        )
        
        logger.info(
            f"Enhanced content with {len(examples_added)} examples, "
            f"{len(festivals_mentioned)} festivals, {len(local_references)} references"
        )
        
        return ContextualContent(
            original_text=content,
            enhanced_text=enhanced_text,
            examples_added=examples_added,
            festivals_mentioned=festivals_mentioned,
            local_references=local_references,
            metadata={
                'subject': subject.value,
                'region': region.value,
                'grade_level': grade_level,
                'topic': topic,
                'enhancement_count': len(examples_added) + len(local_references)
            }
        )
    
    def _get_relevant_examples(
        self,
        subject: Subject,
        topic: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Get relevant examples for the subject and topic."""
        # Get examples from database
        examples = self.context_db.get_examples_by_subject(subject, topic)
        
        # Convert strings to dicts
        example_dicts = [
            {'example': ex, 'topic': topic or 'general'}
            for ex in examples
        ]
        
        # Shuffle for variety
        random.shuffle(example_dicts)
        
        return example_dicts
    
    def _inject_example(self, content: str, example: str) -> str:
        """Inject an example into the content."""
        # Add example after the first paragraph or at the end
        paragraphs = content.split('\n\n')
        
        if len(paragraphs) > 1:
            # Insert after first paragraph
            paragraphs.insert(1, f"\n**Example:** {example}\n")
        else:
            # Append at the end
            paragraphs.append(f"\n**Example:** {example}")
        
        return '\n\n'.join(paragraphs)
    
    def _should_add_festival_reference(self, content: str) -> bool:
        """Determine if a festival reference would be appropriate."""
        # Add festivals for certain keywords or subjects
        festival_keywords = [
            'celebration', 'tradition', 'culture', 'festival',
            'harvest', 'season', 'time', 'occasion', 'event'
        ]
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in festival_keywords)
    
    def _get_relevant_festival(
        self,
        region: Region,
        content: str
    ) -> Optional[Dict[str, Any]]:
        """Get a relevant festival based on content."""
        festivals = self.context_db.get_festivals(region)
        
        if not festivals:
            return None
        
        # Try to find a relevant festival based on keywords in content
        content_lower = content.lower()
        
        for festival in festivals:
            festival_keywords = festival.get('description', '').lower().split()
            if any(keyword in content_lower for keyword in festival_keywords):
                return festival
        
        # Return a random festival if no specific match
        return random.choice(festivals) if festivals else None
    
    def _add_festival_reference(
        self,
        content: str,
        festival: Dict[str, Any]
    ) -> str:
        """Add a festival reference to the content."""
        reference = (
            f"\n\n**Cultural Connection:** In India, during {festival['name']} "
            f"({festival['month']}), {festival['description'].lower()}. "
            f"This festival {festival['significance'].lower()}."
        )
        
        return content + reference
    
    def _should_add_place_reference(self, content: str) -> bool:
        """Determine if a place reference would be appropriate."""
        place_keywords = ['location', 'place', 'where', 'geography', 'visit', 'see']
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in place_keywords)
    
    def _get_relevant_place(self, region: Region) -> Optional[str]:
        """Get a relevant famous place."""
        places = self.context_db.get_famous_places(region)
        return random.choice(places) if places else None
    
    def _add_place_reference(self, content: str, place: str) -> str:
        """Add a place reference to the content."""
        reference = f"\n\n**Did you know?** {place} is a famous place in India that showcases our rich heritage."
        return content + reference
    
    def _get_relevant_person(
        self,
        subject: Subject,
        content: str
    ) -> Optional[Dict[str, Any]]:
        """Get a relevant Indian scientist or freedom fighter."""
        if subject == Subject.SCIENCE:
            people = self.context_db.get_indian_scientists()
        else:  # History or Social Studies
            people = self.context_db.get_freedom_fighters()
        
        if not people:
            return None
        
        # Try to match based on content keywords
        content_lower = content.lower()
        
        for person in people:
            person_keywords = person.get('contribution', '').lower().split()
            if any(keyword in content_lower for keyword in person_keywords):
                return person
        
        # Return random person if no specific match
        return random.choice(people) if people else None
    
    def _add_person_reference(self, content: str, person: Dict[str, Any]) -> str:
        """Add a reference to an Indian scientist or freedom fighter."""
        if 'field' in person:  # Scientist
            reference = (
                f"\n\n**Indian Achievement:** {person['name']}, an Indian {person['field']} expert, "
                f"{person['achievement'].lower()}. This shows India's contribution to science."
            )
        else:  # Freedom fighter
            reference = (
                f"\n\n**Indian History:** {person['name']} {person['contribution'].lower()}. "
                f"They are known for {person['known_for'].lower()}."
            )
        
        return content + reference
    
    def _get_relevant_story(
        self,
        grade_level: int
    ) -> Optional[Dict[str, Any]]:
        """Get a relevant traditional story."""
        stories = self.context_db.get_traditional_stories()
        
        # Filter by age group
        age_min, age_max = grade_level + 5, grade_level + 7  # Approximate age
        
        relevant_stories = [
            story for story in stories
            if self._matches_age_group(story.get('age_group', ''), age_min, age_max)
        ]
        
        return random.choice(relevant_stories) if relevant_stories else None
    
    def _matches_age_group(self, age_group_str: str, age_min: int, age_max: int) -> bool:
        """Check if story age group matches student age."""
        if not age_group_str:
            return False
        
        # Parse "5-12" format
        try:
            parts = age_group_str.split('-')
            story_min = int(parts[0])
            story_max = int(parts[1])
            
            # Check for overlap
            return not (age_max < story_min or age_min > story_max)
        except (ValueError, IndexError):
            return True  # Include if can't parse
    
    def _add_story_reference(self, content: str, story: Dict[str, Any]) -> str:
        """Add a reference to a traditional story."""
        reference = (
            f"\n\n**Story Connection:** This reminds us of {story['name']}, "
            f"where the moral teaches us about {story['moral'].lower()}."
        )
        
        return content + reference
    
    def get_localized_vocabulary(self, region: Region) -> Dict[str, str]:
        """
        Get region-specific vocabulary and terms.
        
        Args:
            region: Region for localization
        
        Returns:
            Dictionary mapping general terms to regional terms
        """
        # This could be expanded with regional language terms
        regional_data = self.context_db.get_regional_info(region)
        
        vocabulary = {
            'food': regional_data.get('foods', []),
            'festivals': [f['name'] for f in regional_data.get('festivals', [])],
            'places': regional_data.get('famous_places', []),
            'crops': regional_data.get('crops', [])
        }
        
        return vocabulary
    
    def batch_enhance(
        self,
        contents: List[Dict[str, Any]],
        subject: Subject,
        region: Optional[Region] = None
    ) -> List[ContextualContent]:
        """
        Enhance multiple content items in batch.
        
        Args:
            contents: List of content dictionaries with 'text' and optional 'topic', 'grade_level'
            subject: Subject area
            region: Optional region for localization
        
        Returns:
            List of ContextualContent objects
        """
        results = []
        
        for i, content_item in enumerate(contents):
            try:
                result = self.enhance_content(
                    content=content_item['text'],
                    subject=subject,
                    topic=content_item.get('topic'),
                    region=region or self.default_region,
                    grade_level=content_item.get('grade_level')
                )
                results.append(result)
                
                logger.info(f"Batch enhancement {i+1}/{len(contents)}: success")
                
            except Exception as e:
                logger.error(f"Batch enhancement failed for item {i+1}: {e}")
                # Add original content as fallback
                results.append(ContextualContent(
                    original_text=content_item['text'],
                    enhanced_text=content_item['text'],
                    examples_added=[],
                    festivals_mentioned=[],
                    local_references=[],
                    metadata={'error': str(e)}
                ))
        
        return results


# Convenience function
def add_cultural_context(
    content: str,
    subject: str,
    region: str = "general",
    grade_level: Optional[int] = None
) -> str:
    """
    Quick utility to add cultural context to content.
    
    Args:
        content: Original content
        subject: Subject area (mathematics, science, etc.)
        region: Region (north, south, east, west, general)
        grade_level: Optional grade level
    
    Returns:
        Enhanced content text
    """
    service = CulturalContextService()
    
    # Convert string inputs to enums
    subject_enum = Subject[subject.upper()]
    region_enum = Region[region.upper()]
    
    result = service.enhance_content(
        content=content,
        subject=subject_enum,
        region=region_enum,
        grade_level=grade_level
    )
    
    return result.enhanced_text


if __name__ == "__main__":
    # Example usage
    sample_content = """
    Photosynthesis is the process by which plants make their food using sunlight,
    water, and carbon dioxide. This process is essential for life on Earth as it
    produces oxygen that we breathe.
    """
    
    service = CulturalContextService(default_region=Region.SOUTH)
    
    result = service.enhance_content(
        content=sample_content,
        subject=Subject.SCIENCE,
        topic="photosynthesis",
        region=Region.SOUTH,
        grade_level=6
    )
    
    logger.info("Original Content:")
    logger.info("=" * 60)
    logger.info(result.original_text)
    
    logger.info("\n\nEnhanced Content:")
    logger.info("=" * 60)
    logger.info(result.enhanced_text)
    
    logger.info("\n\nMetadata:")
    logger.info("=" * 60)
    logger.info(f"Examples added: {len(result.examples_added)}")
    logger.info(f"Festivals mentioned: {result.festivals_mentioned}")
    logger.info(f"Local references: {result.local_references}")
    logger.info(f"Enhancement count: {result.metadata['enhancement_count']}")
