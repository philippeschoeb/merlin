:github_url: https://github.com/merlinquantum/merlin

=============
Circuit Types
=============

Overview
========

Merlin provides three fundamental quantum circuit architectures, each optimized for different machine learning scenarios. This guide provides detailed comparisons and recommendations for selecting appropriate circuit types.

Circuit Architecture Comparison
===============================

PARALLEL_COLUMNS
----------------

**Architecture**: Creates a Cartesian product of input features and optical modes.

**Parameter Count**: ``n_features * n_modes``

**Computational Complexity**: O(n_features * n_modes)

**Mathematical Description**:

For input features ``x = [x1, x2, ..., xn]`` and ``m`` modes, the encoding creates parameters:

.. code-block:: text

    theta_ij = pi * xi for i in [1,n], j in [1,m]

**Advantages**:

- Maximum expressivity for feature interactions
- Rich quantum state exploration
- Suitable for complex datasets with non-linear relationships

**Disadvantages**:

- High parameter count scales quadratically
- Increased training time and memory requirements
- May overfit on small datasets

**Recommended Use Cases**:

- High-dimensional datasets (>100 features)
- Complex pattern recognition tasks
- When maximum model expressivity is required
- Sufficient training data available (>1000 samples)

.. code-block:: python

    experiment = ML.Experiment(
        circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
        n_modes=6,
        n_photons=3,
        use_bandwidth_tuning=True  # Recommended for this type
    )

SERIES
------

**Architecture**: Sequential processing with feature interactions including pairwise terms and sum operations.

**Parameter Count**: Variable, typically ``O(min(2^n_features, n_modes))``

**Computational Complexity**: O(n_features + n_modes)

**Mathematical Description**:

The encoding creates a sequence of transformations:

1. One-to-one mapping: ``theta_i = pi * xi``
2. Pairwise terms: ``theta_ij = pi * (xi + xj) / 2``
3. Sum term: ``theta_sum = pi * sum(xi) / n``

**Advantages**:

- Balanced expressivity and efficiency
- Learns feature combinations naturally
- Moderate parameter scaling
- Good gradient flow properties

**Disadvantages**:

- Limited by exponential scaling for many features
- May miss some complex interactions
- Requires careful tuning for optimal performance

**Recommended Use Cases**:

- Medium-complexity datasets (10-50 features)
- Feature interaction learning
- Classification tasks with moderate complexity
- When interpretability is important

.. code-block:: python

    experiment = ML.Experiment(
        circuit_type=ML.CircuitType.SERIES,
        n_modes=5,
        n_photons=2,
        state_pattern=ML.StatePattern.PERIODIC
    )

PARALLEL
--------

**Architecture**: Direct feature-to-parameter mapping with minimal overhead.

**Parameter Count**: ``n_features`` (multi-feature) or ``n_modes-1`` (single feature)

**Computational Complexity**: O(n_features)

**Mathematical Description**:

Simple direct encoding:

.. code-block:: text

    theta_i = pi * xi for i in [1,n_features]

**Advantages**:

- Minimal parameter overhead
- Fast training and inference
- Excellent for resource-constrained environments
- Low risk of overfitting

**Disadvantages**:

- Limited expressivity
- Cannot capture complex feature interactions
- May underfit complex datasets

**Recommended Use Cases**:

- Simple classification/regression tasks
- Low-dimensional datasets (<10 features)
- Resource-constrained environments
- Baseline model development
- Reservoir computing applications

.. code-block:: python

    experiment = ML.Experiment(
        circuit_type=ML.CircuitType.PARALLEL,
        n_modes=4,
        n_photons=2,
        reservoir_mode=True  # Often used with reservoir computing
    )

Selection Guidelines
====================

Task Complexity
---------------

**Simple Tasks** (Linear separability):

- Use PARALLEL for efficiency
- Consider SERIES if interpretability needed

**Medium Tasks** (Non-linear but structured):

- SERIES provides good balance
- Enable bandwidth tuning for better performance

**Complex Tasks** (Highly non-linear):

- PARALLEL_COLUMNS for maximum expressivity
- Consider ensemble methods with multiple circuit types

Circuit-Specific Optimizations
==============================

PARALLEL_COLUMNS Optimization
-----------------------------

.. code-block:: python

    # Enable bandwidth tuning for better gradient flow
    experiment = ML.Experiment(
        circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
        n_modes=6,
        n_photons=3,
        use_bandwidth_tuning=True
    )

    # Use appropriate batch sizes
    layer = ML.QuantumLayer(input_size=10, ansatz=ansatz)
    # Recommended batch size: 16-32 for this circuit type

SERIES Optimization
-------------------

.. code-block:: python

    # Balance modes and features
    n_features = 8
    n_modes = max(6, n_features)  # Ensure sufficient modes

    experiment = ML.Experiment(
        circuit_type=ML.CircuitType.SERIES,
        n_modes=n_modes,
        n_photons=n_modes // 2
    )

PARALLEL Optimization
---------------------

.. code-block:: python

    # Often combined with reservoir computing
    experiment = ML.Experiment(
        circuit_type=ML.CircuitType.PARALLEL,
        n_modes=8,
        n_photons=4,
        reservoir_mode=True  # Fixed random parameters
    )

    # Larger batch sizes acceptable
    # Recommended batch size: 64-128

Performance Characteristics
===========================

Training Speed Comparison
-------------------------

Relative training time for 1000 samples, 4 modes, 2 photons:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Circuit Type
     - Relative Speed
     - Memory Usage
   * - PARALLEL
     - 1.0x (baseline)
     - Low
   * - SERIES
     - 1.5-2.0x
     - Medium
   * - PARALLEL_COLUMNS
     - 2.5-4.0x
     - High


Hybrid Approaches
=================

Circuit Ensembles
-----------------

Combine multiple circuit types for improved performance:

.. code-block:: python

    class CircuitEnsemble(nn.Module):
        def __init__(self):
            super().__init__()

            # PARALLEL for speed
            exp1 = ML.Experiment(ML.CircuitType.PARALLEL, n_modes=4, n_photons=2)
            ansatz1 = ML.AnsatzFactory.create(exp1, input_size=6, output_size=8)
            self.parallel_layer = ML.QuantumLayer(input_size=6, ansatz=ansatz1)

            # SERIES for interactions
            exp2 = ML.Experiment(ML.CircuitType.SERIES, n_modes=5, n_photons=2)
            ansatz2 = ML.AnsatzFactory.create(exp2, input_size=6, output_size=8)
            self.series_layer = ML.QuantumLayer(input_size=6, ansatz=ansatz2)

            self.combiner = nn.Linear(16, 3)

        def forward(self, x):
            x_norm = torch.sigmoid(x)
            parallel_out = self.parallel_layer(x_norm)
            series_out = self.series_layer(x_norm)
            combined = torch.cat([parallel_out, series_out], dim=1)
            return self.combiner(combined)

Progressive Complexity
----------------------

Start with simple circuits and increase complexity:

.. code-block:: python

    # Stage 1: Baseline with PARALLEL
    baseline_model = create_model(ML.CircuitType.PARALLEL)

    # Stage 2: Add complexity with SERIES
    if baseline_accuracy < threshold:
        enhanced_model = create_model(ML.CircuitType.SERIES)

    # Stage 3: Maximum complexity with PARALLEL_COLUMNS
    if enhanced_accuracy < threshold:
        complex_model = create_model(ML.CircuitType.PARALLEL_COLUMNS)

Troubleshooting Circuit Selection
=================================

Debugging Tools
---------------

.. code-block:: python

    def analyze_circuit_performance(layer, data_loader):
        """Analyze circuit performance characteristics."""

        # Measure training speed
        start_time = time.time()
        for batch in data_loader:
            output = layer(batch)
        training_time = time.time() - start_time

        # Analyze gradient norms
        layer.zero_grad()
        output = layer(next(iter(data_loader)))
        loss = output.sum()
        loss.backward()

        grad_norms = {}
        for name, param in layer.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()

        return {
            'training_time': training_time,
            'gradient_norms': grad_norms,
            'parameter_count': sum(p.numel() for p in layer.parameters())
        }

Future Directions
=================

Adaptive Circuit Selection
--------------------------

Research directions for automatic circuit type selection:

1. **Performance-based switching**: Automatically switch circuit types based on validation performance
2. **Resource-aware selection**: Choose circuits based on computational constraints
3. **Dynamic architectures**: Modify circuit complexity during training

