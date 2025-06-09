"""
Merlin - Photonic Quantum Neural Networks for PyTorch

A comprehensive framework for integrating photonic quantum circuits
into PyTorch neural networks with automatic differentiation support.
"""

# Core API - Most users will only need these
from .core.layer import QuantumLayer
from .core.photonicbackend import PhotonicBackend
from .core.ansatz import Ansatz, AnsatzFactory

# Essential enums
from .core.generators import CircuitType, StatePattern
from .sampling.strategies import OutputMappingStrategy

# Advanced components (for power users)
from .core.generators import CircuitGenerator
from .core.generators import StateGenerator
from .torch_utils.torch_codes import FeatureEncoder
from .torch_utils.torch_codes import SamplingProcess
from .sampling.autodiff import AutoDiffProcess
from .sampling.mappers import OutputMapper, LexGroupingMapper, ModGroupingMapper
from .pcvl_pytorch import CircuitConverter, build_slos_distribution_computegraph

# Version and metadata
__version__ = "0.1.0"
__author__ = "Merlin Team"
__description__ = "Photonic Quantum Machine Learning Framework"

# Public API - what users see with `import merlin as ML`
__all__ = [
    # Core classes (most common usage)
    "QuantumLayer",
    "PhotonicBackend",
    "Ansatz",
    "AnsatzFactory",

    # Configuration enums
    "CircuitType",
    "StatePattern",
    "OutputMappingStrategy",

    # Advanced components
    "CircuitGenerator",
    "StateGenerator",
    "FeatureEncoder",
    "SamplingProcess",
    "AutoDiffProcess",
    "OutputMapper",
    "LexGroupingMapper",
    "ModGroupingMapper",

    "CircuitConverter",
    "build_slos_distribution_computegraph"
]