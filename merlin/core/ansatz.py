"""
Ansatz configuration and factory for quantum layers.
"""

from typing import Optional
import torch

from .photonicbackend import PhotonicBackend
from ..sampling.strategies import OutputMappingStrategy
from ..torch_utils.torch_codes import FeatureEncoder
from ..core.generators import CircuitGenerator
from ..core.generators import StateGenerator
from ..core.process import ComputationProcessFactory


class Ansatz:
    """Complete configuration for a quantum neural network layer."""

    def __init__(self, PhotonicBackend: PhotonicBackend, input_size: int, output_size: Optional[int] = None,
                 output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.LINEAR,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None):
        self.experiment = PhotonicBackend
        self.input_size = input_size
        self.output_size = output_size
        self.output_mapping_strategy = output_mapping_strategy
        self.device = device
        self.dtype = dtype or torch.float32

        # Create feature encoder
        self.feature_encoder = FeatureEncoder(input_size)

        # Generate circuit and state
        self.circuit, self.total_shifters = CircuitGenerator.generate_circuit(
            PhotonicBackend.circuit_type, PhotonicBackend.n_modes, input_size
        )


        self.input_state = StateGenerator.generate_state(
            PhotonicBackend.n_modes, PhotonicBackend.n_photons, PhotonicBackend.state_pattern
        )

        # Setup parameter patterns
        self.input_parameters = ["pl"]
        self.trainable_parameters = [] if PhotonicBackend.reservoir_mode else ["phi_"]
        #self.trainable_parameters= ["phi"]

        # Create computation process with proper dtype
        self.computation_process = ComputationProcessFactory.create(
            circuit=self.circuit,
            input_state=self.input_state,
            trainable_parameters=self.trainable_parameters,
            input_parameters=self.input_parameters,
            reservoir_mode=PhotonicBackend.reservoir_mode,
            dtype=self.dtype,
            device=self.device
        )


class AnsatzFactory:
    """Factory for creating quantum layer ansatzes (complete configurations)."""

    @staticmethod
    def create(PhotonicBackend: PhotonicBackend, input_size: int, output_size: Optional[int] = None,
               output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.LINEAR,
               device: Optional[torch.device] = None,
               dtype: Optional[torch.dtype] = None) -> Ansatz:
        """Create a complete ansatz configuration."""
        return Ansatz(
            PhotonicBackend=PhotonicBackend,
            input_size=input_size,
            output_size=output_size,
            output_mapping_strategy=output_mapping_strategy,
            device=device,
            dtype=dtype
        )
