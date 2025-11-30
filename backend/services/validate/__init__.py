"""Validation module for educational content quality assurance."""

from .validator import ValidationModule, ValidationResult, QualityReport
from .standards import NCERTStandardsLoader, NCERTStandardData, initialize_ncert_standards

__all__ = [
    'ValidationModule',
    'ValidationResult', 
    'QualityReport',
    'NCERTStandardsLoader',
    'NCERTStandardData',
    'initialize_ncert_standards'
]
