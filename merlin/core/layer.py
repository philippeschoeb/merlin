"""
Main QuantumLayer implementation with bug fixes, index_photons support, and simple API.
"""

import math
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import perceval as pcvl

from .ansatz import Ansatz, AnsatzFactory
from ..core.photonicbackend import PhotonicBackend as Experiment
from ..core.generators import CircuitType, StatePattern
from ..sampling.strategies import OutputMappingStrategy
from ..sampling.mappers import OutputMapper
from ..sampling.autodiff import AutoDiffProcess
from ..core.process import ComputationProcessFactory


class QuantumLayer(nn.Module):
    """
    Enhanced Quantum Neural Network Layer with factory-based architecture.

    This layer can be created in multiple ways:
    1. Simple mode: QuantumLayer(input_size=3, n_params=100) - NEW!
    2. Ansatz mode: QuantumLayer(input_size=3, ansatz=ansatz_object)
    3. Custom mode: QuantumLayer(input_size=3, circuit=circuit_object, ...)

    Args:
        input_size (int): Input dimension
        n_params (int, optional): Target number of parameters for simple mode
        ansatz (Ansatz, optional): Pre-created ansatz object
        circuit (pcvl.Circuit, optional): Custom circuit
        index_photons (List[Tuple[int, int]], optional): Photon placement constraints

    Simple mode specific args:
        circuit_type (CircuitType): Type of circuit (default: SERIES)
        reservoir_mode (bool): Whether to use reservoir computing (default: False)
        use_bandwidth_tuning (bool): Enable bandwidth tuning (default: False)
        state_pattern (StatePattern): Pattern for input state (default: PERIODIC)
    """

    def __init__(self, input_size: int, output_size: Optional[int] = None,
                 # Simple mode parameters (NEW!)
                 n_params: Optional[int] = None,
                 # Ansatz-based construction
                 ansatz: Optional[Ansatz] = None,
                 # Custom circuit construction (backward compatible)
                 circuit: Optional[pcvl.Circuit] = None,
                 input_state: Optional[List[int]] = None,
                 n_photons: Optional[int] = None,
                 trainable_parameters: List[str] = [],
                 input_parameters: List[str] = [],
                 # Common parameters
                 output_mapping_strategy: Optional[OutputMappingStrategy] = None,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 shots: int = 0,
                 sampling_method: str = 'multinomial',
                 no_bunching: bool = True,
                 # New parameter for constrained photon placement
                 index_photons: Optional[List[Tuple[int, int]]] = None,
                 # Simple mode specific defaults (NEW!)
                 circuit_type: CircuitType = CircuitType.SERIES,
                 reservoir_mode: bool = False,
                 use_bandwidth_tuning: bool = False,
                 state_pattern: StatePattern = StatePattern.PERIODIC):

        super().__init__()

        self.device = device
        self.dtype = dtype or torch.float32
        self.input_size = input_size
        self.no_bunching = no_bunching
        self.index_photons = index_photons
        

        # Determine construction mode
        if n_params is not None and ansatz is None and circuit is None:
            # NEW: Simple mode
            if self.index_photons is not None:
                raise ValueError(
                    "index_photons should not be used with simple mode (n_params). "
                    "Simple mode abstracts away quantum details. Use ansatz or circuit mode instead."
                )

            # Default to NONE strategy for simple mode
            if output_mapping_strategy is None:
                output_mapping_strategy = OutputMappingStrategy.NONE

            ansatz = self._create_default_ansatz(
                input_size, n_params, circuit_type, reservoir_mode,
                use_bandwidth_tuning, state_pattern, output_mapping_strategy,
                n_photons
            )
            self._init_from_ansatz(ansatz, output_size, output_mapping_strategy)
        elif ansatz is not None:
            # Ansatz mode - use LINEAR as default if not specified
            if output_mapping_strategy is None:
                output_mapping_strategy = OutputMappingStrategy.LINEAR
            self._init_from_ansatz(ansatz, output_size, output_mapping_strategy)
        elif circuit is not None:
            # Custom mode - use LINEAR as default if not specified
            if output_mapping_strategy is None:
                output_mapping_strategy = OutputMappingStrategy.LINEAR
            self._init_from_custom_circuit(
                circuit, input_state, n_photons, trainable_parameters,
                input_parameters, output_size, output_mapping_strategy
            )
        else:
            raise ValueError("Must provide either 'n_params' (simple mode), 'ansatz', or 'circuit'")

        # Setup sampling
        self.autodiff_process = AutoDiffProcess()
        self.shots = shots
        self.sampling_method = sampling_method

    # NEW: Helper methods for simple mode
    @staticmethod
    def _infer_modes_from_params(input_size: int, n_params: int, circuit_type: CircuitType) -> int:
        """
        Infer the number of modes needed based on parameter count.
        For SERIES circuit: n_params = 2 * (n_modes + 1)^2
        """
        if circuit_type == CircuitType.SERIES:
            # Solve: n_params = 2 * (n_modes + 1)^2
            ideal_modes = math.sqrt(n_params / 2) - 1
        else:
            # For other circuit types, use a heuristic
            ideal_modes = math.sqrt(n_params / 4)  # Rough estimate

        # Round to nearest integer, ensure minimum modes
        n_modes = max(int(round(ideal_modes)), input_size + 1)

        # Verify the parameter count is reasonable
        if circuit_type == CircuitType.SERIES:
            actual_params = 2 * (n_modes + 1) ** 2
            if abs(actual_params - n_params) > n_params * 0.1:  # Allow 10% tolerance
                print(f"Warning: Requested {n_params} parameters, but {n_modes} modes gives {actual_params} parameters")

        return n_modes



    @staticmethod
    def _create_periodic_state(n_modes: int, n_photons: Optional[int] = None) -> List[int]:
        """
        Create a periodic state pattern like 10100 for 5 modes.
        If n_photons not specified, uses n_modes // 2.
        """
        if n_photons is None:
            n_photons = n_modes // 2

        state = [0] * n_modes
        for i in range(n_photons):
            state[i * 2 % n_modes] = 1

        return state

    def _create_default_ansatz(self, input_size: int, n_params: int,
                               circuit_type: CircuitType,
                               reservoir_mode: bool,
                               use_bandwidth_tuning: bool,
                               state_pattern: StatePattern,
                               output_mapping_strategy: Optional[OutputMappingStrategy],
                               n_photons: Optional[int] = None) -> Ansatz:
        """Create a default ansatz based on simple parameters."""

        # Infer number of modes
        n_modes = self._infer_modes_from_params(input_size, n_params, circuit_type)

        # Determine number of photons
        if n_photons is not None:
            # Explicit photon count provided
            actual_n_photons = n_photons
        else:
            # Default: use half the modes
            actual_n_photons = n_modes // 2

        # Calculate actual parameter count for the circuit
        if circuit_type == CircuitType.SERIES:
            actual_params = 2 * (n_modes + 1) ** 2
            # Additional parameters from input encoding
            encoding_params = input_size * n_modes  # Phase shifters for input encoding
            total_params = actual_params + encoding_params

            print(f"\nCreating quantum layer:")
            print(f"  Modes: {n_modes}, Photons: {actual_n_photons}")
            print(f"  Circuit parameters: {actual_params}")
            print(f"  Input encoding parameters: {encoding_params}")
            print(f"  Total trainable parameters: {total_params}")

            if abs(total_params - n_params) > n_params * 0.1:
                print(f"  WARNING: Requested {n_params} parameters, but actual count is {total_params}")
                print(f"  For SERIES circuits: parameters = 2*(n_modes+1)² + input_size*n_modes")
        else:
            print(f"Creating quantum layer: {n_modes} modes, {actual_n_photons} photons")

        # Create experiment - let it handle the state creation
        experiment = Experiment(
            circuit_type=circuit_type,
            n_modes=n_modes,
            n_photons=actual_n_photons,
            reservoir_mode=reservoir_mode,
            use_bandwidth_tuning=use_bandwidth_tuning,
            state_pattern=state_pattern
        )

        # Create ansatz
        ansatz = AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=input_size,
            output_mapping_strategy=output_mapping_strategy or OutputMappingStrategy.NONE
        )

        return ansatz

    # NEW: Convenience class methods
    @classmethod
    def simple(cls, input_size: int, n_params: int = None, **kwargs):
        """
        Convenience constructor for simple quantum layer creation.

        Examples:
            >>> # Create with default parameters
            >>> qlayer = QuantumLayer.simple(input_size=3)

            >>> # Create with specific parameter count
            >>> qlayer = QuantumLayer.simple(input_size=3, n_params=100)

            >>> # Create with custom options
            >>> qlayer = QuantumLayer.simple(
            ...     input_size=3,
            ...     n_params=100,
            ...     reservoir_mode=True,
            ...     shots=10000
            ... )
        """
        if n_params is None:
            # Default: create a reasonably sized circuit
            # For input_size=3, this gives 5 modes and 50 parameters
            n_params = 2 * (input_size + 2) ** 2

        return cls(input_size=input_size, n_params=n_params, **kwargs)

    # All existing methods remain unchanged below this point
    def _init_from_ansatz(self, ansatz: Ansatz, output_size: Optional[int],
                          output_mapping_strategy: OutputMappingStrategy):
        """Initialize from ansatz (auto-generated mode)."""
        self.ansatz = ansatz
        self.auto_generation_mode = True

        # Compute output size if not specified
        if output_size is None and hasattr(ansatz, 'experiment'):
            from math import comb
            n_modes = ansatz.experiment.n_modes
            n_photons = ansatz.experiment.n_photons

            if self.no_bunching:
                # No bunching: C(n_modes, n_photons)
                output_size = comb(n_modes, n_photons)
            else:
                # With bunching: C(n_modes + n_photons - 1, n_photons)
                output_size = comb(n_modes + n_photons - 1, n_photons)

        # For ansatz mode, we need to create a new computation process with index_photons
        if self.index_photons is not None:
            # Create a new computation process with index_photons support
            self.computation_process = ComputationProcessFactory.create(
                circuit=ansatz.circuit,
                input_state=ansatz.input_state,
                trainable_parameters=ansatz.trainable_parameters,
                input_parameters=ansatz.input_parameters,
                reservoir_mode=ansatz.experiment.reservoir_mode,
                device=self.device,
                dtype=self.dtype,
                no_bunching=self.no_bunching,
                index_photons=self.index_photons
            )
        else:
            # Use the ansatz's computation process as before
            self.computation_process = ComputationProcessFactory.create(
                circuit=ansatz.circuit,
                input_state=ansatz.input_state,
                trainable_parameters=ansatz.trainable_parameters,
                input_parameters=ansatz.input_parameters,
                reservoir_mode=ansatz.experiment.reservoir_mode,
                device=self.device,
                dtype=self.dtype,
                no_bunching=self.no_bunching,
                index_photons=self.index_photons
            )

        self.feature_encoder = ansatz.feature_encoder

        # Use the ansatz's output mapping strategy - it should take precedence!
        actual_strategy = ansatz.output_mapping_strategy
        actual_output_size = output_size or ansatz.output_size

        # Setup bandwidth tuning if enabled
        if ansatz.experiment.use_bandwidth_tuning:
            self.bandwidth_coeffs = nn.ParameterDict()
            for d in range(self.input_size):
                init = torch.linspace(0.0, 2.0, steps=ansatz.experiment.n_modes,
                                      dtype=self.dtype, device=self.device)
                self.bandwidth_coeffs[f"dim_{d}"] = nn.Parameter(init.clone(), requires_grad=True)
        else:
            self.bandwidth_coeffs = None

        # Setup parameters
        self._setup_parameters_from_ansatz(ansatz)

        # Setup output mapping using ansatz configuration
        self._setup_output_mapping(ansatz, actual_output_size, actual_strategy)

    def _init_from_custom_circuit(self, circuit: pcvl.Circuit, input_state: Optional[List[int]],
                                  n_photons: Optional[int], trainable_parameters: List[str],
                                  input_parameters: List[str], output_size: Optional[int],
                                  output_mapping_strategy: OutputMappingStrategy):
        """Initialize from custom circuit (backward compatible mode)."""
        self.auto_generation_mode = False
        self.bandwidth_coeffs = None

        # Handle state - with index_photons consideration
        if input_state is not None:
            self.input_state = input_state
            # Validate input_state against index_photons constraints if provided
            if self.index_photons is not None:
                self._validate_input_state_with_index_photons(input_state)
        elif n_photons is not None:
            if self.index_photons is not None:
                # Create input state respecting index_photons constraints
                self.input_state = self._create_input_state_from_index_photons(n_photons, circuit.m)
            else:
                # Default behavior: place photons in first n_photons modes
                self.input_state = [1] * n_photons + [0] * (circuit.m - n_photons)
        else:
            raise ValueError("Either input_state or n_photons must be provided")

        # Create computation process with index_photons support
        self.computation_process = ComputationProcessFactory.create(
            circuit=circuit,
            input_state=self.input_state,
            trainable_parameters=trainable_parameters,
            input_parameters=input_parameters,
            device=self.device,
            dtype=self.dtype,
            no_bunching=self.no_bunching,
            index_photons=self.index_photons
        )

        # Setup parameters
        self._setup_parameters_from_custom(trainable_parameters)

        # Setup output mapping
        self._setup_output_mapping_from_custom(output_size, output_mapping_strategy)

    def _validate_input_state_with_index_photons(self, input_state: List[int]):
        """Validate that input_state respects index_photons constraints."""
        if self.index_photons is None:
            return  # No constraints to validate

        photon_idx = 0
        for mode_idx, photon_count in enumerate(input_state):
            for _ in range(photon_count):
                if photon_idx >= len(self.index_photons):
                    raise ValueError(
                        f"Input state has more photons than index_photons constraints. "
                        f"Found {sum(input_state)} photons but only {len(self.index_photons)} constraints."
                    )

                min_mode, max_mode = self.index_photons[photon_idx]
                if not (min_mode <= mode_idx <= max_mode):
                    raise ValueError(
                        f"Photon {photon_idx} is in mode {mode_idx} but index_photons constrains it to "
                        f"modes [{min_mode}, {max_mode}]"
                    )
                photon_idx += 1

    def _create_input_state_from_index_photons(self, n_photons: int, n_modes: int) -> List[int]:
        """Create input state respecting index_photons constraints."""
        if self.index_photons is None or len(self.index_photons) != n_photons:
            raise ValueError(
                f"index_photons must specify constraints for exactly {n_photons} photons. "
                f"Got {len(self.index_photons) if self.index_photons else 0} constraints."
            )

        input_state = [0] * n_modes

        for photon_idx, (min_mode, max_mode) in enumerate(self.index_photons):
            # Validate constraint bounds
            if not (0 <= min_mode <= max_mode < n_modes):
                raise ValueError(
                    f"Invalid index_photons constraint for photon {photon_idx}: "
                    f"[{min_mode}, {max_mode}] must be within [0, {n_modes - 1}]"
                )

            # Place photon in the minimum allowed mode (simplest strategy)
            # Users can override by providing explicit input_state
            input_state[min_mode] += 1

        return input_state

    def _setup_parameters_from_ansatz(self, ansatz: Ansatz):
        """Setup parameters from ansatz configuration."""
        spec_mappings = self.computation_process.converter.spec_mappings
        self.thetas = []
        self.theta_names = []

        # Setup trainable parameters - FIXED: Only add if not in reservoir mode
        if not ansatz.experiment.reservoir_mode:
            for tp in ansatz.trainable_parameters:
                if tp in spec_mappings:
                    theta_list = spec_mappings[tp]
                    self.theta_names += theta_list
                    parameter = nn.Parameter(
                        torch.randn((len(theta_list),), dtype=self.dtype, device=self.device) * torch.pi
                    )
                    self.register_parameter(tp, parameter)
                    self.thetas.append(parameter)

        # Setup reservoir parameters if needed
        if ansatz.experiment.reservoir_mode and "phi_" in spec_mappings:
            phi_list = spec_mappings["phi_"]
            if phi_list:
                phi_values = []
                for param_name in phi_list:
                    # For reservoir mode, just use random values
                    phi_values.append(2 * math.pi * np.random.rand())

                phi_tensor = torch.tensor(phi_values, dtype=self.dtype, device=self.device)
                self.register_buffer("phi_static", phi_tensor)

    def _setup_parameters_from_custom(self, trainable_parameters: List[str]):
        """Setup parameters from custom circuit configuration."""
        spec_mappings = self.computation_process.converter.spec_mappings
        self.thetas = []
        self.theta_names = []

        for tp in trainable_parameters:
            if tp in spec_mappings:
                theta_list = spec_mappings[tp]
                self.theta_names += theta_list
                parameter = nn.Parameter(
                    torch.randn((len(theta_list),), dtype=self.dtype, device=self.device) * torch.pi
                )
                self.register_parameter(tp, parameter)
                self.thetas.append(parameter)

    def _setup_output_mapping(self, ansatz: Ansatz, output_size: Optional[int],
                              output_mapping_strategy: OutputMappingStrategy):
        """Setup output mapping for ansatz-based construction."""
        # Get distribution size
        dummy_params = self._create_dummy_parameters()
        distribution = self.computation_process.compute(dummy_params)
        dist_size = distribution.shape[-1]

        # Determine output size
        if output_size is None:
            if output_mapping_strategy == OutputMappingStrategy.NONE:
                self.output_size = dist_size
            else:
                raise ValueError("output_size must be specified for non-NONE strategies")
        else:
            self.output_size = output_size

        # Validate NONE strategy
        if output_mapping_strategy == OutputMappingStrategy.NONE and self.output_size != dist_size:
            raise ValueError(
                f"Distribution size ({dist_size}) must equal output size ({self.output_size}) "
                f"when using 'none' strategy"
            )

        # Create output mapping
        self.output_mapping = OutputMapper.create_mapping(
            output_mapping_strategy, dist_size, self.output_size
        )

        # Ensure output mapping has correct dtype and device
        if hasattr(self.output_mapping, 'weight'):
            self.output_mapping = self.output_mapping.to(dtype=self.dtype, device=self.device)

    def _setup_output_mapping_from_custom(self, output_size: Optional[int],
                                          output_mapping_strategy: OutputMappingStrategy):
        """Setup output mapping for custom circuit construction."""
        # Get distribution size
        dummy_params = self._create_dummy_parameters()
        distribution = self.computation_process.compute(dummy_params)
        dist_size = distribution.shape[-1]

        # Determine output size
        if output_size is None:
            if output_mapping_strategy == OutputMappingStrategy.NONE:
                self.output_size = dist_size
            else:
                raise ValueError("output_size must be specified for non-NONE strategies")
        else:
            self.output_size = output_size

        # Create output mapping
        self.output_mapping = OutputMapper.create_mapping(
            output_mapping_strategy, dist_size, self.output_size
        )

        # Ensure output mapping has correct dtype and device
        if hasattr(self.output_mapping, 'weight'):
            self.output_mapping = self.output_mapping.to(dtype=self.dtype, device=self.device)

    def _create_dummy_parameters(self) -> List[torch.Tensor]:
        """Create dummy parameters for initialization."""
        params = [theta for theta in self.thetas]

        # Add dummy input parameters - FIXED: Use correct parameter count
        if self.auto_generation_mode:
            dummy_input = torch.zeros(self.ansatz.total_shifters, dtype=self.dtype, device=self.device)
            params.append(dummy_input)
        else:
            # For custom circuits, create dummy based on input parameter count
            spec_mappings = self.computation_process.converter.spec_mappings
            input_params = self.computation_process.input_parameters
            for ip in input_params:
                if ip in spec_mappings:
                    param_count = len(spec_mappings[ip])
                    dummy_input = torch.zeros(param_count, dtype=self.dtype, device=self.device)
                    params.append(dummy_input)

        # Add static phi parameters if in reservoir mode
        if hasattr(self, "phi_static"):
            params.append(self.phi_static)

        return params

    def _prepare_input_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare input encoding based on mode."""
        if self.auto_generation_mode:
            # Use FeatureEncoder for auto-generated circuits
            x_norm = torch.clamp(x, 0, 1)  # Ensure inputs are in valid range

            return self.feature_encoder.encode(
                x_norm,
                self.ansatz.experiment.circuit_type,
                self.ansatz.experiment.n_modes,
                self.bandwidth_coeffs
            )
        else:
            # For custom circuits
            return x

    def prepare_parameters(self, input_parameters: List[torch.Tensor]) -> List[torch.Tensor]:
        """Prepare parameter list for circuit evaluation."""
        # Handle batching
        if input_parameters and input_parameters[0].dim() > 1:
            batch_size = input_parameters[0].shape[0]
            params = [theta.expand(batch_size, -1) for theta in self.thetas]
        else:
            params = [theta for theta in self.thetas]

        # Apply input encoding
        if self.auto_generation_mode and len(input_parameters) == 1:
            encoded = self._prepare_input_encoding(input_parameters[0])
            params.append(encoded)
        else:
            # Custom mode or multiple parameters
            for x in input_parameters:
                encoded = self._prepare_input_encoding(x)
                params.append(encoded)

        # Add static phi parameters if in reservoir mode
        if hasattr(self, "phi_static"):
            if input_parameters and input_parameters[0].dim() > 1:
                batch_size = input_parameters[0].shape[0]
                params.append(self.phi_static.expand(batch_size, -1))
            else:
                params.append(self.phi_static)

        return params

    def forward(self, *input_parameters: torch.Tensor,
                apply_sampling: Optional[bool] = None,
                shots: Optional[int] = None) -> torch.Tensor:
        """Forward pass through the quantum layer."""
        # Prepare parameters
        params = self.prepare_parameters(list(input_parameters))

        # Get quantum output
        distribution = self.computation_process.compute(params)

        # Handle sampling
        needs_gradient = self.training and torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters())
        apply_sampling, shots = self.autodiff_process.autodiff_backend(
            needs_gradient, apply_sampling or False, shots or self.shots
        )

        if apply_sampling and shots > 0:
            distribution = self.autodiff_process.sampling_noise.pcvl_sampler(
                distribution, shots, self.sampling_method
            )

        # Apply output mapping
        return self.output_mapping(distribution)

    def set_sampling_config(self, shots: Optional[int] = None, method: Optional[str] = None):
        """Update sampling configuration."""
        if shots is not None:
            if not isinstance(shots, int) or shots < 0:
                raise ValueError(f"shots must be a non-negative integer, got {shots}")
            self.shots = shots
        if method is not None:
            valid_methods = ['multinomial', 'binomial', 'gaussian']
            if method not in valid_methods:
                raise ValueError(f"Invalid sampling method: {method}. Valid options are: {valid_methods}")
            self.sampling_method = method

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Manually move any additional tensors
        device = kwargs.get('device', None)
        if device is None and len(args) > 0:
            device = args[0]
        if device is not None:
            self.device = device
            self.computation_process.simulation_graph = self.computation_process.simulation_graph.to(self.dtype, device)
            self.computation_process.converter = self.computation_process.converter.to(self.dtype, device)
        return self

    def get_index_photons_info(self) -> dict:
        """
        Get information about index_photons constraints.

        Returns:
            dict: Information about photon placement constraints
        """
        if self.index_photons is None:
            return {
                'constrained': False,
                'message': 'No photon placement constraints (photons can be placed in any mode)'
            }

        info = {
            'constrained': True,
            'n_photons': len(self.index_photons),
            'constraints': []
        }

        for i, (min_mode, max_mode) in enumerate(self.index_photons):
            info['constraints'].append({
                'photon_index': i,
                'allowed_modes': f'[{min_mode}, {max_mode}]',
                'n_allowed_modes': max_mode - min_mode + 1
            })

        return info

    # NEW: Enhanced info method
    def get_circuit_info(self) -> dict:
        """Get detailed information about the quantum circuit configuration."""
        info = {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'construction_mode': 'auto' if self.auto_generation_mode else 'custom',
            'shots': self.shots,
            'sampling_method': self.sampling_method,
            'no_bunching': self.no_bunching
        }

        if self.auto_generation_mode:
            info.update({
                'circuit_type': self.ansatz.experiment.circuit_type.value,
                'n_modes': self.ansatz.experiment.n_modes,
                'n_photons': self.ansatz.experiment.n_photons,
                'input_state': self.ansatz.input_state,
                'reservoir_mode': self.ansatz.experiment.reservoir_mode,
                'use_bandwidth_tuning': self.ansatz.experiment.use_bandwidth_tuning,
                'n_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'output_mapping_strategy': self.ansatz.output_mapping_strategy.value
            })

        if self.index_photons is not None:
            info['index_photons'] = self.index_photons

        return info

    def __str__(self) -> str:
        """String representation of the quantum layer."""
        base_str = ""
        if self.auto_generation_mode:
            base_str = f"QuantumLayer(ansatz={self.ansatz.experiment.circuit_type.value}, " \
                       f"modes={self.ansatz.experiment.n_modes}, " \
                       f"input_size={self.input_size}, output_size={self.output_size}"
        else:
            base_str = f"QuantumLayer(custom_circuit, input_size={self.input_size}, " \
                       f"output_size={self.output_size}"

        # Add index_photons info if present
        if self.index_photons is not None:
            base_str += f", index_photons={self.index_photons}"

        return base_str + ")"
