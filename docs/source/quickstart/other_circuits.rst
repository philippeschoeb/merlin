:github_url: https://github.com/merlinquantum/merlin


=======================
Different Circuit Types
=======================

Simple Parallel Circuit
-----------------------

For basic feature processing:

.. code-block:: python

    def create_simple_circuit(m):
        """Simple parallel processing circuit"""
        circuit = pcvl.Circuit(m)

        # Input encoding only
        for i in range(4):
            px = pcvl.P(f"px{i + 1}")
            circuit.add(i, pcvl.PS(px))

        # Single interferometer
        interferometer = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta{i}")),
            shape=pcvl.InterferometerShape.RECTANGLE
        )
        circuit.add(0, interferometer, merge=True)

        return circuit

Series Circuit with Interactions
--------------------------------

For feature interaction learning:

.. code-block:: python

    def create_series_circuit(m):
        """Series circuit with multiple stages"""
        circuit = pcvl.Circuit(m)

        # Stage 1: Input encoding
        for i in range(4):
            px = pcvl.P(f"px{i + 1}")
            circuit.add(i, pcvl.PS(px))

        # Stage 2: First transformation
        int1 = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta1_{i}")),
            shape=pcvl.InterferometerShape.RECTANGLE
        )
        circuit.add(0, int1, merge=True)

        # Stage 3: Second transformation
        int2 = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta2_{i}")),
            shape=pcvl.InterferometerShape.RECTANGLE
        )
        circuit.add(0, int2, merge=True)

        return circuit

Key Concepts for ML Researchers
================================

Input Normalization
-------------------

Quantum layers require normalized inputs in [0,1]:

.. code-block:: python

    # Always normalize before quantum layers
    x = self.classical_layer(x)
    x = torch.sigmoid(x)  # or torch.clamp(x, 0, 1)
    x = self.quantum_layer(x)

Initial States and Photon Configuration
---------------------------------------

Different initial states affect quantum behavior:

.. code-block:: python

    # Single photons in alternating modes
    input_state_1 = [1, 0, 1, 0, 0, 0]  # Photons in modes 0 and 2

    # Paired photons
    input_state_2 = [2, 0, 2, 0, 0, 0]  # 2 photons each in modes 0 and 2

    # All photons in first mode
    input_state_3 = [4, 0, 0, 0, 0, 0]  # 4 photons in mode 0

    quantum_layer = ML.QuantumLayer(
        input_size=4,
        output_size=3,
        circuit=circuit,
        trainable_parameters=["theta"],
        input_parameters=["px"],
        input_state=input_state_1,  # Choose based on your problem
        output_mapping_strategy=ML.OutputMappingStrategy.LINEAR
    )

Output Mapping Strategies
-------------------------

Different strategies for converting quantum probabilities to classical outputs:

.. code-block:: python

    # Linear transformation (most flexible)
    layer_linear = ML.QuantumLayer(
        circuit=circuit,
        output_mapping_strategy=ML.OutputMappingStrategy.LINEAR
    )

    # Lexicographical grouping (preserves quantum structure)
    layer_lex = ML.QuantumLayer(
        circuit=circuit,
        output_mapping_strategy=ML.OutputMappingStrategy.LEXGROUPING
    )

    # Modular grouping (for periodic patterns)
    layer_mod = ML.QuantumLayer(
        circuit=circuit,
        output_mapping_strategy=ML.OutputMappingStrategy.MODGROUPING
    )

    # Identity mapping (direct quantum probabilities)
    layer_none = ML.QuantumLayer(
        circuit=circuit,
        output_mapping_strategy=ML.OutputMappingStrategy.NONE
    )

Model Creation Helper
--------------------

Streamlined model creation function:

.. code-block:: python

    def create_model(model_type, variant):
        """Create model instance based on type and variant"""
        if model_type == 'MLP':
            return MLP(input_size=4, output_size=3, config=variant['config'])
        else:
            m = variant['config']['m']
            no_bunching = variant['config'].get('no_bunching', False)
            c = create_quantum_circuit(m)

            return ML.QuantumLayer(
                input_size=4,
                output_size=3,
                circuit=c,
                trainable_parameters=["theta"],
                input_parameters=["px"],
                input_state=[1, 0] * (m // 2) + [0] * (m % 2),
                no_bunching=no_bunching,
                output_mapping_strategy=variant['config']['output_mapping_strategy']
            )

    # Usage examples
    quantum_config = {
        'config': {
            'm': 6,
            'no_bunching': False,
            'output_mapping_strategy': ML.OutputMappingStrategy.LINEAR
        }
    }

    classical_config = {
        'config': {
            'hidden_layers': [64, 32],
            'activation': 'relu'
        }
    }

    quantum_model = create_model('Quantum', quantum_config)
    classical_model = create_model('MLP', classical_config)

Common Patterns
===============

Multi-Layer Quantum Networks
----------------------------

.. code-block:: python

    class DeepQuantumNetwork(nn.Module):
        def __init__(self):
            super().__init__()

            # First quantum layer
            circuit1 = create_quantum_circuit(6)
            self.quantum1 = ML.QuantumLayer(
                input_size=6,
                output_size=8,
                circuit=circuit1,
                trainable_parameters=["theta"],
                input_parameters=["px"],
                input_state=[1, 0, 1, 0, 1, 0],
                output_mapping_strategy=ML.OutputMappingStrategy.LINEAR
            )

            # Classical processing
            self.classical = nn.Sequential(
                nn.Linear(8, 6),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

            # Second quantum layer (different circuit)
            circuit2 = create_simple_circuit(4)
            self.quantum2 = ML.QuantumLayer(
                input_size=6,
                output_size=4,
                circuit=circuit2,
                trainable_parameters=["theta"],
                input_parameters=["px"],
                input_state=[1, 0, 1, 0],
                output_mapping_strategy=ML.OutputMappingStrategy.LEXGROUPING
            )

            self.output = nn.Linear(4, 2)

        def forward(self, x):
            x = torch.sigmoid(x)
            x = self.quantum1(x)
            x = self.classical(x)
            x = torch.sigmoid(x)  # Normalize again
            x = self.quantum2(x)
            return self.output(x)

Reservoir Computing
-------------------

For efficient training with fixed quantum parameters:

.. code-block:: python

    class QuantumReservoir(nn.Module):
        def __init__(self, reservoir_size=8):
            super().__init__()

            # Large quantum reservoir with fixed parameters
            circuit = create_quantum_circuit(reservoir_size)
            self.reservoir = ML.QuantumLayer(
                input_size=4,
                output_size=16,
                circuit=circuit,
                trainable_parameters=[],  # No trainable quantum parameters
                input_parameters=["px"],
                input_state=[1, 0] * (reservoir_size // 2),
                output_mapping_strategy=ML.OutputMappingStrategy.LINEAR
            )

            # Only train the readout layer
            self.readout = nn.Linear(16, 3)

        def forward(self, x):
            x = torch.sigmoid(x)
            x = self.reservoir(x)  # Fixed quantum transformation
            return self.readout(x)    # Trainable linear readout

    # Check trainable parameters
    model = QuantumReservoir()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params}, Total: {total_params}")

Performance Tips
================

Circuit Size Guidelines
-----------------------

.. code-block:: python

    # Small problems: 4-6 modes
    small_circuit = create_quantum_circuit(4)

    # Medium problems: 6-8 modes
    medium_circuit = create_quantum_circuit(6)

    # Large problems: 8+ modes (computational cost increases rapidly)
    large_circuit = create_quantum_circuit(8)

Batch Size Optimization
-----------------------

.. code-block:: python

    # Quantum layers are computationally intensive
    # Start with smaller batch sizes

    batch_size_guidelines = {
        4: 128,   # 4 modes
        6: 64,    # 6 modes
        8: 32,    # 8 modes
        10: 16    # 10+ modes
    }

Debugging and Monitoring
========================

Parameter Tracking
------------------

.. code-block:: python

    # Monitor gradient flow
    for name, param in model.named_parameters():
        if param.grad is not None and 'theta' in name:
            grad_norm = param.grad.norm()
            print(f"Quantum parameter {name}: gradient norm = {grad_norm:.6f}")

Circuit Validation
------------------

.. code-block:: python

    # Test circuit before training
    quantum_layer = ML.QuantumLayer(...)

    # Check input/output shapes
    test_input = torch.rand(5, 4)
    test_output = quantum_layer(test_input)
    print(f"Input: {test_input.shape} -> Output: {test_output.shape}")

    # Check parameter counts
    theta_params = [p for name, p in quantum_layer.named_parameters() if 'theta' in name]
    print(f"Trainable quantum parameters: {sum(p.numel() for p in theta_params)}")

Next Steps
==========

Now that you understand the basics:

1. **Experiment with different circuits**: Try the various circuit creation functions
2. **Explore output mappings**: Test different mapping strategies for your use case
3. **Scale up**: Build deeper hybrid networks with multiple quantum layers
4. **Optimize performance**: Tune batch sizes and learning rates for your hardware

For more advanced topics, see:

- :doc:`../user_guide/circuit_types` - Different quantum architectures
- :doc:`../user_guide/output_mappings` - Output mapping strategies
- :doc:`../user_guide/hybrid_models` - Advanced hybrid architectures
- :doc:`../../examples/` - Complete examples and benchmarks