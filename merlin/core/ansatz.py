# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Ansatz configuration and factory for quantum layers.
"""

import torch

from ..core.generators import CircuitGenerator, StateGenerator
from ..core.process import ComputationProcessFactory
from ..sampling.strategies import OutputMappingStrategy
from ..torch_utils.torch_codes import FeatureEncoder
from .photonicbackend import PhotonicBackend


class Ansatz:
    """Complete configuration for a quantum neural network layer."""

    def __init__(
        self,
        PhotonicBackend: PhotonicBackend,
        input_size: int,
        output_size: int | None = None,
        output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.LINEAR,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        r"""Initialize the Ansatz with the given configuration.

        Args:
            PhotonicBackend (PhotonicBackend): The backend configuration to use.
            input_size (int): Size of the input feature vector.
            output_size (int | None): Size of the output vector. If None, it is defined by the backend.
            output_mapping_strategy (OutputMappingStrategy): Strategy for mapping outputs.
            device (torch.device | None): Device to run computations on.
            dtype (torch.dtype | None): Data type for computations.
        """
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
            PhotonicBackend.n_modes,
            PhotonicBackend.n_photons,
            PhotonicBackend.state_pattern,
        )

        # Setup parameter patterns
        self.input_parameters = ["pl"]
        self.trainable_parameters = [] if PhotonicBackend.reservoir_mode else ["phi_"]
        # self.trainable_parameters= ["phi"]

        # Create computation process with proper dtype
        self.computation_process = ComputationProcessFactory.create(
            circuit=self.circuit,
            input_state=self.input_state,
            trainable_parameters=self.trainable_parameters,
            input_parameters=self.input_parameters,
            reservoir_mode=PhotonicBackend.reservoir_mode,
            dtype=self.dtype,
            device=self.device,
        )


class AnsatzFactory:
    """Factory for creating quantum layer ansatzes (complete configurations)."""

    @staticmethod
    def create(
        PhotonicBackend: PhotonicBackend,
        input_size: int,
        output_size: int | None = None,
        output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.LINEAR,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Ansatz:
        r"""Create a complete ansatz configuration.

        Args:
            PhotonicBackend (PhotonicBackend): The backend configuration to use.
            input_size (int): Size of the input feature vector.
            output_size (int | None): Size of the output vector. If None, it is defined by the backend.
            output_mapping_strategy (OutputMappingStrategy): Strategy for mapping outputs.
            device (torch.device | None): Device to run computations on.
            dtype (torch.dtype | None): Data type for computations.

        Returns:
            Ansatz: A complete ansatz configuration for the quantum layer.
        """
        return Ansatz(
            PhotonicBackend=PhotonicBackend,
            input_size=input_size,
            output_size=output_size,
            output_mapping_strategy=output_mapping_strategy,
            device=device,
            dtype=dtype,
        )
