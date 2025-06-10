"""
Output mapping strategy definitions.
"""

from enum import Enum


class OutputMappingStrategy(Enum):
    """Strategy for mapping quantum probability distributions to classical outputs."""
    LINEAR = 'linear'
    GROUPING = 'grouping'
    LEXGROUPING = 'lexgrouping'
    MODGROUPING = 'modgrouping'
    NONE = 'none'