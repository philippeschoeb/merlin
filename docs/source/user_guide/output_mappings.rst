:github_url: https://github.com/merlinquantum/merlin

=========================
Output Mapping Strategies
=========================

Overview
========

Output mapping strategies transform quantum probability distributions into classical neural network outputs. The choice of mapping strategy significantly impacts model performance, interpretability, and computational efficiency.

Mapping Strategies
==================

LINEAR
------

**Description**: Standard neural network linear transformation with learnable weights and biases.

**Mathematical Form**:

.. code-block:: text

    y = Wx + b

where W is a matrix of size (output_size x input_size), b is a vector of size output_size

**Characteristics**:

- Fully learnable transformation
- Can produce any real-valued output
- Standard backpropagation applies
- Most flexible but computationally intensive

**Use Cases**:

- Classification tasks requiring logits
- Regression with arbitrary output ranges
- When maximum learning flexibility is needed
- Standard deep learning integration

.. code-block:: python

    ansatz = ML.AnsatzFactory.create(
        experiment=experiment,
        input_size=4,
        output_size=10,
        output_mapping_strategy=ML.OutputMappingStrategy.LINEAR
    )

**Advantages**:

- Universal approximation capability
- Familiar optimization landscape
- Easy integration with existing architectures
- Good gradient flow properties

**Disadvantages**:

- Adds learnable parameters
- May lose quantum structure information
- Computationally more expensive

LEXGROUPING
-----------

**Description**: Lexicographical grouping of probability amplitudes into equal-sized buckets.

**Mathematical Form**:

For probability distribution p of size n mapping to output y of size m:

.. code-block:: text

    y_i = sum(p_j) for j from i*k to (i+1)*k-1

where k = ceiling(n/m) (bucket size)

**Padding Behavior**:

If n is not divisible by m, zeros are padded to make equal-sized groups.

**Characteristics**:

- No learnable parameters
- Preserves probability mass
- Deterministic grouping scheme
- Output values in [0, 1] range

**Use Cases**:

- Probability-based outputs
- When preserving quantum measurement statistics
- Resource-constrained environments
- Interpretable quantum outputs

.. code-block:: python

    ansatz = ML.AnsatzFactory.create(
        experiment=experiment,
        input_size=4,
        output_size=6,
        output_mapping_strategy=ML.OutputMappingStrategy.LEXGROUPING
    )

**Example**:

.. code-block:: python

    # Input distribution: [0.1, 0.2, 0.3, 0.1, 0.2, 0.1] (6 elements)
    # Output size: 3
    # Grouping: [0.1+0.2, 0.3+0.1, 0.2+0.1] = [0.3, 0.4, 0.3]

**Advantages**:

- No additional parameters
- Preserves quantum measurement structure
- Fast computation
- Interpretable outputs

**Disadvantages**:

- Limited flexibility
- May not capture optimal feature combinations
- Fixed grouping scheme

MODGROUPING
-----------

**Description**: Groups probability amplitudes based on modulo arithmetic.

**Mathematical Form**:

.. code-block:: text

    y_i = sum(p_j) for all j where j mod m = i

**Characteristics**:

- No learnable parameters
- Distributes indices cyclically
- Good for capturing periodic patterns
- Output values in [0, 1] range

**Use Cases**:

- Periodic or cyclic data patterns
- When spatial/temporal locality matters
- Specific problem structures with modular symmetry

.. code-block:: python

    ansatz = ML.AnsatzFactory.create(
        experiment=experiment,
        input_size=4,
        output_size=3,
        output_mapping_strategy=ML.OutputMappingStrategy.MODGROUPING
    )

**Example**:

.. code-block:: python

    # Input distribution: [0.1, 0.2, 0.3, 0.15, 0.1, 0.15] (indices 0-5)
    # Output size: 3
    # Group 0 (indices 0,3): 0.1 + 0.15 = 0.25
    # Group 1 (indices 1,4): 0.2 + 0.1 = 0.3
    # Group 2 (indices 2,5): 0.3 + 0.15 = 0.45
    # Result: [0.25, 0.3, 0.45]

**Advantages**:

- Captures cyclic patterns
- No additional parameters
- Even distribution of information
- Suitable for certain symmetries

**Disadvantages**:

- Limited to specific problem types
- May not suit arbitrary distributions
- Less intuitive than lexicographical grouping

NONE (Identity)
---------------

**Description**: Direct use of quantum probability distribution as output.

**Mathematical Form**:

.. code-block:: text

    y = p (identity mapping)

**Requirements**:

- Distribution size must equal desired output size
- No size transformation possible

**Characteristics**:

- No parameters or computation overhead
- Direct quantum measurement interpretation
- Outputs are valid probability distributions
- Sum to 1 (normalized)

**Use Cases**:

- Probability estimation tasks
- When quantum distribution is the desired output
- Maximum efficiency requirements
- Quantum-native applications

.. code-block:: python

    # Must ensure distribution size matches output size
    temp_ansatz = ML.AnsatzFactory.create(experiment, input_size=4, output_size=10)
    temp_layer = ML.QuantumLayer(input_size=4, ansatz=temp_ansatz)
    # Determine actual distribution size
    dist_size = temp_layer(torch.rand(1, 4)).shape[1]

    # Create NONE mapping with matching size
    ansatz = ML.AnsatzFactory.create(
        experiment=experiment,
        input_size=4,
        output_size=dist_size,  # Must match
        output_mapping_strategy=ML.OutputMappingStrategy.NONE
    )

**Advantages**:

- Zero computational overhead
- Pure quantum information preservation
- No additional parameters
- Maximum computational efficiency

**Disadvantages**:

- Rigid size constraints
- Limited output range [0,1]
- Sum constraint (sum of all y_i = 1)
- Not suitable for arbitrary outputs

Selection Guidelines
====================

Task-Based Recommendations
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Task Type
     - Primary Choice
     - Alternative
     - Reasoning
   * - Classification
     - LINEAR
     - LEXGROUPING
     - Need logits/flexible outputs
   * - Regression
     - LINEAR
     - None
     - Require arbitrary output ranges
   * - Probability Estimation
     - NONE
     - LEXGROUPING
     - Want direct probabilities
   * - Structured Outputs
     - MODGROUPING
     - LEXGROUPING
     - Exploit pattern structure

Performance Considerations
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Strategy
     - Parameter Cost
     - Computation Cost
     - Memory Usage
   * - LINEAR
     - O(input_size × output_size)
     - O(input_size × output_size)
     - High
   * - LEXGROUPING
     - 0
     - O(input_size)
     - Low
   * - MODGROUPING
     - 0
     - O(input_size)
     - Low
   * - NONE
     - 0
     - 0
     - Minimal

Size Compatibility
------------------

.. code-block:: python

    def check_mapping_compatibility(quantum_output_size, desired_output_size):
        """Check which mapping strategies are compatible."""

        compatible = []

        # LINEAR: Always compatible
        compatible.append('LINEAR')

        # LEXGROUPING: Always compatible (uses padding)
        compatible.append('LEXGROUPING')

        # MODGROUPING: Always compatible
        compatible.append('MODGROUPING')

        # NONE: Only if sizes match exactly
        if quantum_output_size == desired_output_size:
            compatible.append('NONE')

        return compatible

Advanced Usage Patterns
=======================

Dynamic Strategy Selection
--------------------------

.. code-block:: python

    class AdaptiveOutputLayer(nn.Module):
        def __init__(self, quantum_layer, strategies=['LINEAR', 'LEXGROUPING']):
            super().__init__()
            self.quantum_layer = quantum_layer
            self.strategies = strategies
            self.current_strategy = 0

            # Create multiple output mappings
            self.mappers = nn.ModuleDict()
            for strategy in strategies:
                if strategy == 'LINEAR':
                    self.mappers[strategy] = nn.Linear(dist_size, output_size)
                elif strategy == 'LEXGROUPING':
                    self.mappers[strategy] = ML.LexGroupingMapper(dist_size, output_size)
                # ... other strategies

        def forward(self, x):
            quantum_out = self.quantum_layer(x)
            strategy = self.strategies[self.current_strategy]
            return self.mappers[strategy](quantum_out)

        def switch_strategy(self, new_strategy_idx):
            self.current_strategy = new_strategy_idx

Ensemble Output Mapping
-----------------------

.. code-block:: python

    class EnsembleOutputMapping(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()

            # Multiple mapping strategies
            self.linear = nn.Linear(input_size, output_size)
            self.lexgroup = ML.LexGroupingMapper(input_size, output_size)
            self.modgroup = ML.ModGroupingMapper(input_size, output_size)

            # Learnable combination weights
            self.combination_weights = nn.Parameter(torch.ones(3) / 3)

        def forward(self, quantum_distribution):
            # Apply all strategies
            linear_out = self.linear(quantum_distribution)
            lex_out = self.lexgroup(quantum_distribution)
            mod_out = self.modgroup(quantum_distribution)

            # Weighted combination
            weights = torch.softmax(self.combination_weights, dim=0)
            combined = (weights[0] * linear_out +
                       weights[1] * lex_out +
                       weights[2] * mod_out)

            return combined

Hierarchical Mapping
--------------------

.. code-block:: python

    class HierarchicalMapping(nn.Module):
        def __init__(self, input_size, intermediate_size, output_size):
            super().__init__()

            # First stage: Reduce dimensionality with grouping
            self.stage1 = ML.LexGroupingMapper(input_size, intermediate_size)

            # Second stage: Learn final transformation
            self.stage2 = nn.Linear(intermediate_size, output_size)

        def forward(self, quantum_distribution):
            intermediate = self.stage1(quantum_distribution)
            return self.stage2(intermediate)

Optimization Strategies
=======================

Gradient Flow Analysis
----------------------

Different mapping strategies affect gradient flow differently:

**LINEAR**:

- Full gradient backpropagation through learned weights
- May benefit from learning rate scheduling
- Standard optimization techniques apply

**LEXGROUPING/MODGROUPING**:

- Direct gradient flow through grouping operation
- Generally stable gradients
- May require careful quantum layer optimization

**NONE**:

- Direct gradients to quantum layer
- No intermediate transformation noise
- Depends entirely on quantum circuit optimization

Performance Tuning
------------------

.. code-block:: python

    def optimize_mapping_choice(model, val_loader, strategies):
        """Empirically determine best mapping strategy."""

        results = {}

        for strategy in strategies:
            # Create model variant with this strategy
            model_variant = create_model_with_strategy(strategy)

            # Evaluate performance
            val_loss = evaluate_model(model_variant, val_loader)
            train_time = measure_training_speed(model_variant)
            memory_usage = measure_memory_usage(model_variant)

            results[strategy] = {
                'val_loss': val_loss,
                'train_time': train_time,
                'memory_usage': memory_usage,
                'score': val_loss + 0.1 * train_time + 0.01 * memory_usage
            }

        # Return best strategy
        best_strategy = min(results.keys(), key=lambda k: results[k]['score'])
        return best_strategy, results

Integration with Classical Networks
===================================

Pre-quantum Processing
----------------------

.. code-block:: python

    class PreQuantumProcessor(nn.Module):
        def __init__(self, classical_size, quantum_size):
            super().__init__()
            self.processor = nn.Sequential(
                nn.Linear(classical_size, quantum_size * 2),
                nn.ReLU(),
                nn.Linear(quantum_size * 2, quantum_size),
                nn.Sigmoid()  # Normalize for quantum layer
            )

        def forward(self, x):
            return self.processor(x)

Post-quantum Processing
-----------------------

.. code-block:: python

    class PostQuantumProcessor(nn.Module):
        def __init__(self, quantum_size, final_size, mapping_strategy):
            super().__init__()

            # Quantum output mapping
            if mapping_strategy == 'LINEAR':
                self.mapper = nn.Linear(quantum_size, quantum_size // 2)
            elif mapping_strategy == 'LEXGROUPING':
                self.mapper = ML.LexGroupingMapper(quantum_size, quantum_size // 2)

            # Classical post-processing
            self.post_processor = nn.Sequential(
                nn.Linear(quantum_size // 2, final_size * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(final_size * 2, final_size)
            )

        def forward(self, quantum_output):
            mapped = self.mapper(quantum_output)
            return self.post_processor(mapped)

Debugging Output Mappings
=========================

Diagnostic Tools
----------------

.. code-block:: python

    def diagnose_output_mapping(layer, test_input):
        """Diagnose output mapping behavior."""

        # Get quantum distribution
        with torch.no_grad():
            quantum_dist = layer.computation_process.compute(
                layer.prepare_parameters([test_input])
            )

        print(f"Quantum distribution shape: {quantum_dist.shape}")
        print(f"Distribution sum: {quantum_dist.sum():.6f}")
        print(f"Distribution range: [{quantum_dist.min():.6f}, {quantum_dist.max():.6f}]")

        # Test mapping
        final_output = layer.output_mapping(quantum_dist)
        print(f"Final output shape: {final_output.shape}")
        print(f"Output range: [{final_output.min():.6f}, {final_output.max():.6f}]")

        # Check gradients
        loss = final_output.sum()
        loss.backward()

        if hasattr(layer.output_mapping, 'weight'):
            if layer.output_mapping.weight.grad is not None:
                grad_norm = layer.output_mapping.weight.grad.norm()
                print(f"Output mapping gradient norm: {grad_norm:.6f}")