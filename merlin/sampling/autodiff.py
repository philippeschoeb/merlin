"""
Automatic differentiation handling for sampling.
"""

import warnings
from typing import Tuple
from .process import SamplingProcess


class AutoDiffProcess:
    """Handles automatic differentiation backend and sampling noise integration."""

    def __init__(self):
        self.sampling_noise = SamplingProcess()

    def autodiff_backend(self, needs_gradient: bool, apply_sampling: bool,
                         shots: int) -> Tuple[bool, int]:
        """Determine sampling configuration based on gradient requirements."""
        if needs_gradient and (apply_sampling or shots > 0):
            warnings.warn(
                "Sampling was requested but is disabled because gradients are being computed. "
                "Sampling during gradient computation would lead to incorrect gradients.",
                UserWarning
            )
            return False, 0
        return apply_sampling, shots