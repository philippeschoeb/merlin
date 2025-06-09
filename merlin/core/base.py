"""
Abstract base classes for computation processes.
"""

from abc import ABC, abstractmethod


class AbstractComputationProcess(ABC):
    """Abstract base class for quantum computation processes."""

    @abstractmethod
    def compute(self, *args, **kwargs):
        """Perform the computation."""
        pass