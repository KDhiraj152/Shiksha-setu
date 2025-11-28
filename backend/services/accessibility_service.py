"""
Accessibility Service for Enhanced Learning Support.

This module provides accessibility features including dyslexia-friendly modes,
high-contrast themes, screen reader optimization, and enhanced TTS controls.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AccessibilityMode(Enum):
    """Accessibility mode types."""
    DYSLEXIA_FRIENDLY = "dyslexia_friendly"
    HIGH_CONTRAST = "high_contrast"
    SCREEN_READER = "screen_reader"
    LARGE_TEXT = "large_text"
    SIMPLIFIED_LAYOUT = "simplified_layout"
    COLOR_BLIND_SAFE = "color_blind_safe"


class TextToSpeechSpeed(Enum):
    """TTS playback speeds."""
    VERY_SLOW = 0.5
    SLOW = 0.75
    NORMAL = 1.0
    FAST = 1.25
    VERY_FAST = 1.5


class FontFamily(Enum):
    """Dyslexia-friendly font families."""
    OPEN_DYSLEXIC = "OpenDyslexic"
    COMIC_SANS = "Comic Sans MS"
    VERDANA = "Verdana"
    ARIAL = "Arial"
    LEXEND = "Lexend"


@dataclass
class AccessibilitySettings:
    """User accessibility settings."""
    modes: List[AccessibilityMode]
    font_family: FontFamily
    font_size: int  # pixels
    line_spacing: float  # multiplier
    word_spacing: float  # multiplier
    letter_spacing: float  # em units
    tts_speed: TextToSpeechSpeed
    tts_voice: Optional[str]
    high_contrast_enabled: bool
    reduce_animations: bool
    keyboard_navigation: bool
    focus_indicators: bool


@dataclass
class AccessibleContent:
    """Content formatted for accessibility."""
    text: str
    html: str
    css: Dict[str, str]
    aria_labels: Dict[str, str]
    alt_text: List[str]
    tts_optimized_text: str
    metadata: Dict[str, Any]


class DyslexiaFriendlyFormatter:
    """
    Formats text to be dyslexia-friendly.
    
    Features:
    - Increased letter spacing
    - Larger line height
    - Special fonts (OpenDyslexic, Comic Sans)
    - Highlighted key words
    - Reduced visual clutter
    """
    
    RECOMMENDED_SETTINGS = {
        'font_size': 16,  # px, minimum
        'line_height': 1.5,  # minimum
        'letter_spacing': 0.12,  # em
        'word_spacing': 0.16,  # em
        'max_line_length': 70,  # characters
        'paragraph_spacing': 2.0,  # em
    }
    
    @classmethod
    def format_text(
        cls,
        text: str,
        font_family: FontFamily = FontFamily.OPEN_DYSLEXIC,
        font_size: int = 16
    ) -> Tuple[str, Dict[str, str]]:
        """
        Format text for dyslexia-friendly reading.
        
        Args:
            text: Original text
            font_family: Font to use
            font_size: Font size in pixels
        
        Returns:
            Tuple of (formatted_html, css_dict)
        """
        # Break long lines
        formatted_text = cls._break_long_lines(text)
        
        # Add paragraph breaks
        formatted_text = cls._add_paragraph_spacing(formatted_text)
        
        # Generate CSS
        css = cls._generate_dyslexia_css(font_family, font_size)
        
        # Wrap in HTML
        html = f'<div class="dyslexia-friendly">{formatted_text}</div>'
        
        return html, css
    
    @classmethod
    def _break_long_lines(cls, text: str) -> str:
        """Break lines that are too long."""
        max_length = cls.RECOMMENDED_SETTINGS['max_line_length']
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > max_length:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
            else:
                current_line.append(word)
                current_length += word_length
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '<br>'.join(lines)
    
    @classmethod
    def _add_paragraph_spacing(cls, text: str) -> str:
        """Add spacing between paragraphs."""
        paragraphs = text.split('\n\n')
        return '</p><p class="dyslexia-paragraph">'.join(paragraphs)
    
    @classmethod
    def _generate_dyslexia_css(
        cls,
        font_family: FontFamily,
        font_size: int
    ) -> Dict[str, str]:
        """Generate CSS for dyslexia-friendly formatting."""
        settings = cls.RECOMMENDED_SETTINGS
        
        return {
            '.dyslexia-friendly': f'''
                font-family: '{font_family.value}', sans-serif;
                font-size: {max(font_size, settings['font_size'])}px;
                line-height: {settings['line_height']};
                letter-spacing: {settings['letter_spacing']}em;
                word-spacing: {settings['word_spacing']}em;
                color: #000000;
                background-color: #FAFAFA;
                padding: 20px;
                max-width: 800px;
            ''',
            '.dyslexia-paragraph': f'''
                margin-bottom: {settings['paragraph_spacing']}em;
                text-align: left;
            '''
        }


class HighContrastTheme:
    """
    Provides high-contrast color schemes for better visibility.
    """
    
    THEMES = {
        'black_on_white': {
            'background': '#FFFFFF',
            'text': '#000000',
            'headings': '#000000',
            'links': '#0000EE',
            'borders': '#000000',
        },
        'white_on_black': {
            'background': '#000000',
            'text': '#FFFFFF',
            'headings': '#FFFFFF',
            'links': '#4D94FF',
            'borders': '#FFFFFF',
        },
        'yellow_on_black': {
            'background': '#000000',
            'text': '#FFFF00',
            'headings': '#FFFF00',
            'links': '#00FFFF',
            'borders': '#FFFF00',
        },
        'black_on_yellow': {
            'background': '#FFFF00',
            'text': '#000000',
            'headings': '#000000',
            'links': '#0000FF',
            'borders': '#000000',
        },
    }
    
    @classmethod
    def get_theme_css(cls, theme_name: str = 'black_on_white') -> Dict[str, str]:
        """
        Get CSS for high-contrast theme.
        
        Args:
            theme_name: Name of theme
        
        Returns:
            Dictionary of CSS rules
        """
        theme = cls.THEMES.get(theme_name, cls.THEMES['black_on_white'])
        
        return {
            'body': f'''
                background-color: {theme['background']} !important;
                color: {theme['text']} !important;
            ''',
            'h1, h2, h3, h4, h5, h6': f'''
                color: {theme['headings']} !important;
                border-bottom: 2px solid {theme['borders']};
            ''',
            'a': f'''
                color: {theme['links']} !important;
                text-decoration: underline;
            ''',
            '.card, .panel, .box': f'''
                background-color: {theme['background']} !important;
                border: 2px solid {theme['borders']} !important;
            ''',
            'button': f'''
                background-color: {theme['text']} !important;
                color: {theme['background']} !important;
                border: 2px solid {theme['borders']} !important;
            ''',
        }


class ScreenReaderOptimizer:
    """
    Optimizes content for screen readers.
    """
    
    @classmethod
    def add_aria_labels(cls, html: str, labels: Dict[str, str]) -> str:
        """
        Add ARIA labels to HTML elements.
        
        Args:
            html: Original HTML
            labels: Dictionary of {element_id: label}
        
        Returns:
            HTML with ARIA labels
        """
        # This is a simplified implementation
        # In production, use a proper HTML parser
        
        for element_id, label in labels.items():
            # Add aria-label attribute
            html = html.replace(
                f'id="{element_id}"',
                f'id="{element_id}" aria-label="{label}"'
            )
        
        return html
    
    @classmethod
    def generate_content_structure(cls, text: str) -> Dict[str, Any]:
        """
        Generate content structure for screen readers.
        
        Args:
            text: Content text
        
        Returns:
            Dictionary with structured content
        """
        # Detect headings, lists, etc.
        structure = {
            'type': 'document',
            'sections': [],
            'has_headings': False,
            'has_lists': False,
            'has_tables': False,
        }
        
        # Simple detection
        if any(marker in text for marker in ['#', 'Chapter', 'Section']):
            structure['has_headings'] = True
        
        if any(marker in text for marker in ['1.', '2.', '•', '-']):
            structure['has_lists'] = True
        
        return structure
    
    @classmethod
    def optimize_for_tts(cls, text: str) -> str:
        """
        Optimize text for text-to-speech.
        
        - Expand abbreviations
        - Add pronunciation hints
        - Format numbers properly
        - Handle special characters
        
        Args:
            text: Original text
        
        Returns:
            TTS-optimized text
        """
        optimized = text
        
        # Expand common abbreviations
        abbreviations = {
            'Dr.': 'Doctor',
            'Mr.': 'Mister',
            'Mrs.': 'Missus',
            'Ms.': 'Miss',
            'St.': 'Saint',
            'Ave.': 'Avenue',
            'vs.': 'versus',
            'etc.': 'etcetera',
            'i.e.': 'that is',
            'e.g.': 'for example',
        }
        
        for abbr, full in abbreviations.items():
            optimized = optimized.replace(abbr, full)
        
        # Add pauses after punctuation for natural speech
        optimized = optimized.replace('.', '. ')
        optimized = optimized.replace(',', ', ')
        optimized = optimized.replace(';', '; ')
        
        # Remove extra whitespace
        optimized = ' '.join(optimized.split())
        
        return optimized


class AccessibilityService:
    """
    Main accessibility service providing comprehensive accessibility features.
    """
    
    def __init__(self):
        """Initialize the accessibility service."""
        self.dyslexia_formatter = DyslexiaFriendlyFormatter()
        self.high_contrast = HighContrastTheme()
        self.screen_reader_optimizer = ScreenReaderOptimizer()
        logger.info("AccessibilityService initialized")
    
    def apply_accessibility_settings(
        self,
        content: str,
        settings: AccessibilitySettings
    ) -> AccessibleContent:
        """
        Apply accessibility settings to content.
        
        Args:
            content: Original content
            settings: User accessibility settings
        
        Returns:
            AccessibleContent with all accessibility features applied
        """
        if not content or len(content.strip()) == 0:
            raise ValueError("Content cannot be empty")
        
        logger.info(f"Applying accessibility settings with {len(settings.modes)} modes")
        
        # Start with original content
        html = content
        css = {}
        aria_labels = {}
        alt_text = []
        
        # Apply dyslexia-friendly formatting
        if AccessibilityMode.DYSLEXIA_FRIENDLY in settings.modes:
            html, dyslexia_css = self.dyslexia_formatter.format_text(
                content,
                settings.font_family,
                settings.font_size
            )
            css.update(dyslexia_css)
            logger.debug("Applied dyslexia-friendly formatting")
        
        # Apply high-contrast theme
        if settings.high_contrast_enabled:
            contrast_css = self.high_contrast.get_theme_css('black_on_white')
            css.update(contrast_css)
            logger.debug("Applied high-contrast theme")
        
        # Apply large text mode
        if AccessibilityMode.LARGE_TEXT in settings.modes:
            css['.content'] = f'''
                font-size: {settings.font_size}px;
                line-height: {settings.line_spacing};
            '''
            logger.debug(f"Applied large text mode: {settings.font_size}px")
        
        # Optimize for screen readers
        if AccessibilityMode.SCREEN_READER in settings.modes:
            aria_labels = {
                'main': 'Main educational content',
                'navigation': 'Course navigation',
                'sidebar': 'Additional resources'
            }
            html = self.screen_reader_optimizer.add_aria_labels(html, aria_labels)
            logger.debug("Optimized for screen readers")
        
        # Optimize for TTS
        tts_optimized = self.screen_reader_optimizer.optimize_for_tts(content)
        
        # Apply additional CSS for animations
        if settings.reduce_animations:
            css['*'] = '''
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            '''
            logger.debug("Reduced animations")
        
        # Apply keyboard navigation enhancements
        if settings.keyboard_navigation:
            css['*:focus'] = '''
                outline: 3px solid #4A90E2 !important;
                outline-offset: 2px !important;
            '''
            logger.debug("Enhanced keyboard navigation")
        
        logger.info("Accessibility settings applied successfully")
        
        return AccessibleContent(
            text=content,
            html=html,
            css=css,
            aria_labels=aria_labels,
            alt_text=alt_text,
            tts_optimized_text=tts_optimized,
            metadata={
                'modes': [mode.value for mode in settings.modes],
                'font_size': settings.font_size,
                'tts_speed': settings.tts_speed.value,
                'high_contrast': settings.high_contrast_enabled
            }
        )
    
    def get_default_settings(self) -> AccessibilitySettings:
        """Get default accessibility settings."""
        return AccessibilitySettings(
            modes=[],
            font_family=FontFamily.ARIAL,
            font_size=16,
            line_spacing=1.5,
            word_spacing=1.0,
            letter_spacing=0.0,
            tts_speed=TextToSpeechSpeed.NORMAL,
            tts_voice=None,
            high_contrast_enabled=False,
            reduce_animations=False,
            keyboard_navigation=True,
            focus_indicators=True
        )
    
    def get_dyslexia_preset(self) -> AccessibilitySettings:
        """Get preset for dyslexia support."""
        return AccessibilitySettings(
            modes=[AccessibilityMode.DYSLEXIA_FRIENDLY],
            font_family=FontFamily.OPEN_DYSLEXIC,
            font_size=18,
            line_spacing=1.8,
            word_spacing=1.2,
            letter_spacing=0.12,
            tts_speed=TextToSpeechSpeed.SLOW,
            tts_voice=None,
            high_contrast_enabled=False,
            reduce_animations=True,
            keyboard_navigation=True,
            focus_indicators=True
        )
    
    def get_visual_impairment_preset(self) -> AccessibilitySettings:
        """Get preset for visual impairment support."""
        return AccessibilitySettings(
            modes=[
                AccessibilityMode.HIGH_CONTRAST,
                AccessibilityMode.LARGE_TEXT,
                AccessibilityMode.SCREEN_READER
            ],
            font_family=FontFamily.ARIAL,
            font_size=24,
            line_spacing=2.0,
            word_spacing=1.0,
            letter_spacing=0.05,
            tts_speed=TextToSpeechSpeed.NORMAL,
            tts_voice=None,
            high_contrast_enabled=True,
            reduce_animations=True,
            keyboard_navigation=True,
            focus_indicators=True
        )
    
    def export_settings(self, settings: AccessibilitySettings) -> Dict[str, Any]:
        """
        Export settings to JSON-serializable format.
        
        Args:
            settings: Accessibility settings
        
        Returns:
            Dictionary of settings
        """
        return {
            'modes': [mode.value for mode in settings.modes],
            'font_family': settings.font_family.value,
            'font_size': settings.font_size,
            'line_spacing': settings.line_spacing,
            'word_spacing': settings.word_spacing,
            'letter_spacing': settings.letter_spacing,
            'tts_speed': settings.tts_speed.value,
            'tts_voice': settings.tts_voice,
            'high_contrast_enabled': settings.high_contrast_enabled,
            'reduce_animations': settings.reduce_animations,
            'keyboard_navigation': settings.keyboard_navigation,
            'focus_indicators': settings.focus_indicators
        }
    
    def import_settings(self, settings_dict: Dict[str, Any]) -> AccessibilitySettings:
        """
        Import settings from dictionary.
        
        Args:
            settings_dict: Settings dictionary
        
        Returns:
            AccessibilitySettings object
        """
        return AccessibilitySettings(
            modes=[AccessibilityMode(mode) for mode in settings_dict.get('modes', [])],
            font_family=FontFamily(settings_dict.get('font_family', 'Arial')),
            font_size=settings_dict.get('font_size', 16),
            line_spacing=settings_dict.get('line_spacing', 1.5),
            word_spacing=settings_dict.get('word_spacing', 1.0),
            letter_spacing=settings_dict.get('letter_spacing', 0.0),
            tts_speed=TextToSpeechSpeed(settings_dict.get('tts_speed', 1.0)),
            tts_voice=settings_dict.get('tts_voice'),
            high_contrast_enabled=settings_dict.get('high_contrast_enabled', False),
            reduce_animations=settings_dict.get('reduce_animations', False),
            keyboard_navigation=settings_dict.get('keyboard_navigation', True),
            focus_indicators=settings_dict.get('focus_indicators', True)
        )


# Convenience function
def make_accessible(
    content: str,
    preset: str = "default"
) -> AccessibleContent:
    """
    Quick utility to make content accessible.
    
    Args:
        content: Original content
        preset: Preset name (default, dyslexia, visual_impairment)
    
    Returns:
        AccessibleContent
    """
    service = AccessibilityService()
    
    if preset == "dyslexia":
        settings = service.get_dyslexia_preset()
    elif preset == "visual_impairment":
        settings = service.get_visual_impairment_preset()
    else:
        settings = service.get_default_settings()
    
    return service.apply_accessibility_settings(content, settings)


if __name__ == "__main__":
    # Example usage
    sample_content = """
    Photosynthesis is the process by which plants convert light energy into chemical energy.
    This process takes place in the chloroplasts of plant cells and requires sunlight, water,
    and carbon dioxide. The products of photosynthesis are glucose and oxygen.
    
    The equation for photosynthesis is:
    6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂
    """
    
    service = AccessibilityService()
    
    # Test different presets
    presets = ['default', 'dyslexia', 'visual_impairment']
    
    for preset_name in presets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Preset: {preset_name}")
        logger.info('='*60)
        
        result = make_accessible(sample_content, preset_name)
        
        logger.info(f"\nAccessibility Modes: {result.metadata['modes']}")
        logger.info(f"Font Size: {result.metadata['font_size']}px")
        logger.info(f"TTS Speed: {result.metadata['tts_speed']}x")
        logger.info(f"High Contrast: {result.metadata['high_contrast']}")
        
        logger.info("\nTTS-Optimized Text:")
        logger.info(result.tts_optimized_text[:200] + "...")
        
        logger.info(f"\nCSS Rules Applied: {len(result.css)}")
        logger.info(f"ARIA Labels: {len(result.aria_labels)}")
