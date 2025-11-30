"""
Script Accuracy Validator for Indian Languages and Mathematical/Scientific Notation.

This module validates complex characters, mathematical symbols, and scientific notation
across all Indian language scripts to ensure accurate rendering and display.
"""
import re
import unicodedata
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ScriptType(Enum):
    """Indian language script types."""
    DEVANAGARI = "devanagari"  # Hindi, Marathi, Sanskrit
    BENGALI = "bengali"
    TAMIL = "tamil"
    TELUGU = "telugu"
    GUJARATI = "gujarati"
    KANNADA = "kannada"
    MALAYALAM = "malayalam"
    GURMUKHI = "gurmukhi"  # Punjabi
    ODIA = "odia"
    URDU = "urdu"  # Arabic script
    LATIN = "latin"  # English
    MIXED = "mixed"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"  # Critical issue, must fix
    WARNING = "warning"  # Should fix, but not critical
    INFO = "info"  # Informational, optional fix


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    category: str  # math, science, script, encoding, etc.
    message: str
    position: int  # Character position in text
    context: str  # Surrounding text
    suggestion: Optional[str] = None


@dataclass
class ScriptValidationResult:
    """Result of script validation."""
    is_valid: bool
    detected_scripts: List[ScriptType]
    issues: List[ValidationIssue]
    stats: Dict[str, int]
    recommendations: List[str]


class UnicodeScriptDetector:
    """
    Detects and validates Unicode scripts in text.
    """
    
    # Unicode ranges for Indian scripts
    SCRIPT_RANGES = {
        ScriptType.DEVANAGARI: [(0x0900, 0x097F)],
        ScriptType.BENGALI: [(0x0980, 0x09FF)],
        ScriptType.GUJARATI: [(0x0A80, 0x0AFF)],
        ScriptType.GURMUKHI: [(0x0A00, 0x0A7F)],
        ScriptType.KANNADA: [(0x0C80, 0x0CFF)],
        ScriptType.MALAYALAM: [(0x0D00, 0x0D7F)],
        ScriptType.ODIA: [(0x0B00, 0x0B7F)],
        ScriptType.TAMIL: [(0x0B80, 0x0BFF)],
        ScriptType.TELUGU: [(0x0C00, 0x0C7F)],
        ScriptType.URDU: [(0x0600, 0x06FF), (0x0750, 0x077F)],  # Arabic
        ScriptType.LATIN: [(0x0041, 0x005A), (0x0061, 0x007A)],
    }
    
    @classmethod
    def detect_scripts(cls, text: str) -> List[ScriptType]:
        """
        Detect which scripts are used in the text.
        
        Args:
            text: Text to analyze
        
        Returns:
            List of detected script types
        """
        detected = set()
        
        for char in text:
            char_code = ord(char)
            
            for script_type, ranges in cls.SCRIPT_RANGES.items():
                for start, end in ranges:
                    if start <= char_code <= end:
                        detected.add(script_type)
                        break
        
        scripts = list(detected)
        
        # If multiple scripts detected, add MIXED
        if len(scripts) > 1:
            scripts.append(ScriptType.MIXED)
        
        return scripts


class MathematicalSymbolValidator:
    """
    Validates mathematical symbols and notation.
    """
    
    # Common mathematical symbols and their Unicode
    MATH_SYMBOLS = {
        # Basic operations
        '+': '\u002B',
        '−': '\u2212',  # Minus (not hyphen)
        '×': '\u00D7',  # Multiplication
        '÷': '\u00F7',  # Division
        '=': '\u003D',
        '≠': '\u2260',
        '≈': '\u2248',
        '≡': '\u2261',
        
        # Comparison
        '<': '\u003C',
        '>': '\u003E',
        '≤': '\u2264',
        '≥': '\u2265',
        
        # Advanced operations
        '√': '\u221A',  # Square root
        '∛': '\u221B',  # Cube root
        '∫': '\u222B',  # Integral
        '∑': '\u2211',  # Sum
        '∏': '\u220F',  # Product
        '∂': '\u2202',  # Partial derivative
        '∇': '\u2207',  # Nabla/Del
        
        # Greek letters (commonly used in math)
        'α': '\u03B1',  # alpha
        'β': '\u03B2',  # beta
        'γ': '\u03B3',  # gamma
        'δ': '\u03B4',  # delta
        'π': '\u03C0',  # pi
        'θ': '\u03B8',  # theta
        'λ': '\u03BB',  # lambda
        'μ': '\u03BC',  # mu
        'σ': '\u03C3',  # sigma
        'Σ': '\u03A3',  # Sigma (capital)
        'Δ': '\u0394',  # Delta (capital)
        'Ω': '\u03A9',  # Omega
        
        # Special
        '∞': '\u221E',  # Infinity
        '∅': '\u2205',  # Empty set
        '∈': '\u2208',  # Element of
        '∉': '\u2209',  # Not element of
        '⊂': '\u2282',  # Subset
        '⊃': '\u2283',  # Superset
        '∪': '\u222A',  # Union
        '∩': '\u2229',  # Intersection
        '°': '\u00B0',  # Degree
        '′': '\u2032',  # Prime
        '″': '\u2033',  # Double prime
    }
    
    # Common errors (incorrect character → correct character)
    COMMON_ERRORS = {
        '-': '−',  # Hyphen vs minus
        'x': '×',  # Letter x vs multiplication
        '/': '÷',  # Slash vs division symbol (context-dependent)
        '~': '≈',  # Tilde vs approximately equal
        '!=': '≠',  # Programming notation
        '<=': '≤',
        '>=': '≥',
        'sqrt': '√',
        'sum': '∑',
        'pi': 'π',
        'theta': 'θ',
        'alpha': 'α',
        'beta': 'β',
        'gamma': 'γ',
        'delta': 'δ',
        'infinity': '∞',
    }
    
    @classmethod
    def _check_hyphen_as_minus(cls, text: str, pos: int, char: str) -> Optional[ValidationIssue]:
        """Check if hyphen is used as minus in math context."""
        if char == '-' and cls._is_math_context(text, pos):
            context = cls._get_context(text, pos)
            return ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="math",
                message="Hyphen (-) used instead of minus sign (−)",
                position=pos,
                context=context,
                suggestion=text[:pos] + '−' + text[pos+1:]
            )
        return None
    
    @classmethod
    def _check_x_as_multiplication(cls, text: str, pos: int, char: str) -> Optional[ValidationIssue]:
        """Check if letter 'x' is used as multiplication."""
        if char.lower() == 'x' and cls._is_multiplication_context(text, pos):
            context = cls._get_context(text, pos)
            return ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="math",
                message="Letter 'x' used instead of multiplication symbol (×)",
                position=pos,
                context=context,
                suggestion=text[:pos] + '×' + text[pos+1:]
            )
        return None
    
    @classmethod
    def _check_text_based_symbols(cls, text: str) -> List[ValidationIssue]:
        """Check for text-based math symbols that could be replaced."""
        issues = []
        for text_symbol, unicode_symbol in cls.COMMON_ERRORS.items():
            if text_symbol in text.lower():
                for match in re.finditer(re.escape(text_symbol), text, re.IGNORECASE):
                    pos = match.start()
                    context = cls._get_context(text, pos)
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        category="math",
                        message=f"Text '{text_symbol}' could be replaced with symbol '{unicode_symbol}'",
                        position=pos,
                        context=context,
                        suggestion=text[:pos] + unicode_symbol + text[pos+len(text_symbol):]
                    ))
        return issues
    
    @classmethod
    def validate_math_symbols(cls, text: str) -> List[ValidationIssue]:
        """
        Validate mathematical symbols in text.
        
        Args:
            text: Text to validate
        
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check for common character errors
        for pos, char in enumerate(text):
            issue = cls._check_hyphen_as_minus(text, pos, char)
            if issue:
                issues.append(issue)
            
            issue = cls._check_x_as_multiplication(text, pos, char)
            if issue:
                issues.append(issue)
        
        # Check for text-based math symbols
        issues.extend(cls._check_text_based_symbols(text))
        
        return issues
    
    @classmethod
    def _is_math_context(cls, text: str, pos: int) -> bool:
        """Check if position is in a mathematical context."""
        # Look at surrounding characters
        before = text[max(0, pos-5):pos]
        after = text[pos+1:min(len(text), pos+6)]
        
        # Check if surrounded by digits or math operators
        math_chars = '0123456789+−×÷=≠<>≤≥()[]{}.'
        
        has_math_before = any(c in math_chars for c in before)
        has_math_after = any(c in math_chars for c in after)
        
        return has_math_before and has_math_after
    
    @classmethod
    def _is_multiplication_context(cls, text: str, pos: int) -> bool:
        """Check if 'x' is likely a multiplication symbol."""
        # Check if surrounded by numbers or variables
        before = text[max(0, pos-2):pos].strip()
        after = text[pos+1:min(len(text), pos+3)].strip()
        
        # 'x' as multiplication: "2 x 3" or "a x b"
        is_number_before = before and (before[-1].isdigit() or before[-1].isalpha())
        is_number_after = after and (after[0].isdigit() or after[0].isalpha())
        
        return is_number_before and is_number_after
    
    @classmethod
    def _get_context(cls, text: str, pos: int, window: int = 20) -> str:
        """Get context around a position."""
        start = max(0, pos - window)
        end = min(len(text), pos + window + 1)
        context = text[start:end]
        
        # Add markers
        marker_pos = pos - start
        context = context[:marker_pos] + '►' + context[marker_pos] + '◄' + context[marker_pos+1:]
        
        return context


class ScientificNotationValidator:
    """
    Validates scientific notation and symbols.
    """
    
    # Scientific symbols
    SCIENTIFIC_SYMBOLS = {
        # Chemistry
        'H₂O': 'Water',
        'CO₂': 'Carbon dioxide',
        'O₂': 'Oxygen',
        'N₂': 'Nitrogen',
        'H⁺': 'Hydrogen ion',
        'OH⁻': 'Hydroxide ion',
        
        # Physics units
        'm/s': 'meters per second',
        'm/s²': 'meters per second squared',
        'kg·m/s²': 'Newton',
        'J': 'Joule',
        'W': 'Watt',
        'N': 'Newton',
        'Pa': 'Pascal',
        
        # Math/Science
        '×10': 'Scientific notation',
        'e': 'Euler\'s number',
    }
    
    # Subscript and superscript mappings
    SUBSCRIPTS = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'
    }
    
    SUPERSCRIPTS = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '+': '⁺', '−': '⁻', '=': '⁼', '(': '⁽', ')': '⁾'
    }
    
    @classmethod
    def validate_scientific_notation(cls, text: str) -> List[ValidationIssue]:
        """
        Validate scientific notation in text.
        
        Args:
            text: Text to validate
        
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check for improper subscript notation (H2O instead of H₂O)
        chemical_pattern = r'([A-Z][a-z]?)(\d+)'
        for match in re.finditer(chemical_pattern, text):
            element = match.group(1)
            number = match.group(2)
            pos = match.start()
            
            # Convert to proper subscript
            subscript_number = ''.join(cls.SUBSCRIPTS[d] for d in number)
            proper_notation = element + subscript_number
            
            context = MathematicalSymbolValidator._get_context(text, pos)
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="science",
                message=f"Chemical formula should use subscripts: {element}{number} → {proper_notation}",
                position=pos,
                context=context,
                suggestion=text[:pos] + proper_notation + text[pos+len(match.group(0)):]
            ))
        
        # Check for improper exponent notation (10^6 instead of 10⁶)
        exponent_pattern = r'(\d+)\^([+\-]?\d+)'
        for match in re.finditer(exponent_pattern, text):
            base = match.group(1)
            exponent = match.group(2)
            pos = match.start()
            
            # Convert to proper superscript
            superscript_exp = ''.join(
                cls.SUPERSCRIPTS.get(c, c) for c in exponent
            )
            proper_notation = base + superscript_exp
            
            context = MathematicalSymbolValidator._get_context(text, pos)
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="science",
                message=f"Exponent should use superscripts: {base}^{exponent} → {proper_notation}",
                position=pos,
                context=context,
                suggestion=text[:pos] + proper_notation + text[pos+len(match.group(0)):]
            ))
        
        return issues


class ScriptValidator:
    """
    Main script validation class that coordinates all validators.
    """
    
    def __init__(self):
        """Initialize the script validator."""
        self.script_detector = UnicodeScriptDetector()
        self.math_validator = MathematicalSymbolValidator()
        self.science_validator = ScientificNotationValidator()
        logger.info("ScriptValidator initialized")
    
    def validate(
        self,
        text: str,
        check_math: bool = True,
        check_science: bool = True,
        check_encoding: bool = True
    ) -> ScriptValidationResult:
        """
        Perform comprehensive script validation.
        
        Args:
            text: Text to validate
            check_math: Check mathematical symbols
            check_science: Check scientific notation
            check_encoding: Check Unicode encoding issues
        
        Returns:
            ScriptValidationResult with all findings
        """
        if not text:
            return ScriptValidationResult(
                is_valid=True,
                detected_scripts=[],
                issues=[],
                stats={},
                recommendations=[]
            )
        
        logger.info(f"Validating text of length {len(text)}")
        
        # Detect scripts
        detected_scripts = self.script_detector.detect_scripts(text)
        logger.debug(f"Detected scripts: {[s.value for s in detected_scripts]}")
        
        issues = []
        
        # Mathematical symbol validation
        if check_math:
            math_issues = self.math_validator.validate_math_symbols(text)
            issues.extend(math_issues)
            logger.debug(f"Found {len(math_issues)} math-related issues")
        
        # Scientific notation validation
        if check_science:
            science_issues = self.science_validator.validate_scientific_notation(text)
            issues.extend(science_issues)
            logger.debug(f"Found {len(science_issues)} science-related issues")
        
        # Encoding validation
        if check_encoding:
            encoding_issues = self._check_encoding_issues(text)
            issues.extend(encoding_issues)
            logger.debug(f"Found {len(encoding_issues)} encoding-related issues")
        
        # Calculate statistics
        stats = self._calculate_stats(text, issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, detected_scripts)
        
        # Determine if valid (no ERROR severity issues)
        is_valid = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        
        logger.info(
            f"Validation complete: {len(issues)} issues found, "
            f"valid={is_valid}"
        )
        
        return ScriptValidationResult(
            is_valid=is_valid,
            detected_scripts=detected_scripts,
            issues=issues,
            stats=stats,
            recommendations=recommendations
        )
    
    def _check_encoding_issues(self, text: str) -> List[ValidationIssue]:
        """Check for Unicode encoding issues."""
        issues = []
        
        for pos, char in enumerate(text):
            try:
                # Check if character is properly encoded
                char.encode('utf-8')
                
                # Check for problematic characters
                category = unicodedata.category(char)
                
                # Detect replacement characters (�)
                if char == '\ufffd':
                    context = MathematicalSymbolValidator._get_context(text, pos)
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="encoding",
                        message="Replacement character detected (encoding error)",
                        position=pos,
                        context=context,
                        suggestion="Fix encoding or replace with correct character"
                    ))
                
                # Detect zero-width characters
                elif category in ['Cf', 'Cc']:  # Format or control characters
                    if char not in ['\n', '\t', '\r']:  # Allow common whitespace
                        context = MathematicalSymbolValidator._get_context(text, pos)
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category="encoding",
                            message=f"Unusual control character detected (U+{ord(char):04X})",
                            position=pos,
                            context=context,
                            suggestion="Remove or replace control character"
                        ))
            
            except UnicodeEncodeError:
                context = MathematicalSymbolValidator._get_context(text, pos)
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="encoding",
                    message="Character cannot be encoded in UTF-8",
                    position=pos,
                    context=context,
                    suggestion="Replace with compatible character"
                ))
        
        return issues
    
    def _calculate_stats(
        self,
        text: str,
        issues: List[ValidationIssue]
    ) -> Dict[str, int]:
        """Calculate statistics about the text and validation."""
        stats = {
            'total_characters': len(text),
            'total_issues': len(issues),
            'error_count': sum(1 for i in issues if i.severity == ValidationSeverity.ERROR),
            'warning_count': sum(1 for i in issues if i.severity == ValidationSeverity.WARNING),
            'info_count': sum(1 for i in issues if i.severity == ValidationSeverity.INFO),
        }
        
        # Count issues by category
        categories = {issue.category for issue in issues}
        for category in categories:
            stats[f'{category}_issues'] = sum(
                1 for i in issues if i.category == category
            )
        
        # Count mathematical symbols
        math_symbols = MathematicalSymbolValidator.MATH_SYMBOLS
        stats['math_symbols_used'] = sum(
            1 for char in text if char in math_symbols.values()
        )
        
        return stats
    
    def _generate_recommendations(
        self,
        issues: List[ValidationIssue],
        scripts: List[ScriptType]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Group issues by severity
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        warnings = [i for i in issues if i.severity == ValidationSeverity.WARNING]
        
        if errors:
            recommendations.append(
                f"Fix {len(errors)} critical error(s) to ensure proper text rendering."
            )
        
        if warnings:
            recommendations.append(
                f"Address {len(warnings)} warning(s) to improve text accuracy."
            )
        
        # Math-specific recommendations
        math_issues = [i for i in issues if i.category == "math"]
        if math_issues:
            recommendations.append(
                f"Replace {len(math_issues)} text-based math notation with proper symbols."
            )
        
        # Science-specific recommendations
        science_issues = [i for i in issues if i.category == "science"]
        if science_issues:
            recommendations.append(
                f"Use proper subscripts/superscripts for {len(science_issues)} scientific notation(s)."
            )
        
        # Script-mixing recommendation
        if ScriptType.MIXED in scripts:
            recommendations.append(
                "Multiple scripts detected. Ensure consistent formatting across scripts."
            )
        
        # Positive feedback
        if not issues:
            recommendations.append(
                "Excellent! All symbols and notation are correctly formatted."
            )
        
        return recommendations
    
    def auto_fix(self, text: str) -> str:
        """
        Automatically fix common issues in text.
        
        Args:
            text: Text to fix
        
        Returns:
            Fixed text
        """
        result = text
        
        # Get validation results
        validation = self.validate(result)
        
        # Apply suggestions for INFO and WARNING issues
        for issue in sorted(validation.issues, key=lambda x: x.position, reverse=True):
            if issue.suggestion and issue.severity != ValidationSeverity.ERROR:
                # Apply the suggestion
                result = issue.suggestion
        
        logger.info(f"Auto-fix applied {len(validation.issues)} corrections")
        
        return result


# Convenience function
def validate_text(text: str) -> ScriptValidationResult:
    """
    Quick validation of text.
    
    Args:
        text: Text to validate
    
    Returns:
        ScriptValidationResult
    """
    validator = ScriptValidator()
    return validator.validate(text)


if __name__ == "__main__":
    # Example usage
    test_texts = [
        # Mathematical notation
        "The equation is 2 + 2 = 4, and sqrt(16) = 4. Also, pi ≈ 3.14.",
        
        # Scientific notation
        "Water (H2O) reacts with carbon dioxide (CO2) to form carbonic acid.",
        
        # Mixed script (Hindi-English)
        "गणित में √16 = 4 होता है। This is called square root.",
        
        # Errors
        "The formula is 2 x 3 - 5 = 1 (using wrong x for multiplication)",
    ]
    
    validator = ScriptValidator()
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {text[:50]}...")
        print('='*60)
        
        result = validator.validate(text)
        
        print(f"\nDetected Scripts: {', '.join(s.value for s in result.detected_scripts)}")
        print(f"Valid: {result.is_valid}")
        print(f"Issues Found: {len(result.issues)}")
        
        if result.issues:
            print("\nIssues:")
            for issue in result.issues[:5]:  # Show first 5
                print(f"  [{issue.severity.value.upper()}] {issue.message}")
                print(f"    Context: {issue.context}")
                if issue.suggestion:
                    print(f"    Suggestion: ...{issue.suggestion[max(0, issue.position-20):issue.position+30]}...")
        
        print("\nStatistics:")
        for key, value in result.stats.items():
            print(f"  {key}: {value}")
        
        if result.recommendations:
            print("\nRecommendations:")
            for rec in result.recommendations:
                print(f"  • {rec}")
