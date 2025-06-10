:github_url: https://github.com/merlinquantum/merlin

===============================
Hybrid Classical-Quantum Models
===============================

Overview
========

Hybrid models combine classical neural networks with quantum layers to leverage the strengths of both paradigms. This guide covers design patterns, integration strategies, and optimization techniques for building effective hybrid architectures.

Architecture Patterns
=====================

Sequential Hybrid Models
------------------------

**Pattern**: Classical → Quantum → Classical processing chain

.. code-block:: python

    class SequentialHybrid(nn.Module):
        def __init__(self, input_dim, quantum_dim, output_dim):
            super().__init__()

            # Classical preprocessing
            self.classical_pre = nn.Sequential(
                nn.Linear(input_dim, quantum_dim * 2),
                nn.BatchNorm1d(quantum_dim * 2),
                nn.ReLU(),
                nn.Linear(quantum_dim * 2, quantum_dim),
                nn.Sigmoid()  # Normalize for quantum layer
            )

            # Quantum processing
            experiment = ML.Experiment(
                circuit_type=ML.CircuitType.SERIES,
                n_modes=6,
                n_photons=3,
                use_bandwidth_tuning=True
            )
            ansatz = ML.AnsatzFactory.create(
                experiment=experiment,
                input_size=quantum_dim,
                output_size=quantum_dim * 2
            )
            self.quantum_layer = ML.QuantumLayer(input_size=quantum_dim, ansatz=ansatz)

            # Classical postprocessing
            self.classical_post = nn.Sequential(
                nn.Linear(quantum_dim * 2, output_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim)
            )

        def forward(self, x):
            x = self.classical_pre(x)
            x = self.quantum_layer(x)
            return self.classical_post(x)

**Advantages**:

- Clear separation of concerns
- Easy to optimize each component separately
- Good for feature extraction and dimensionality reduction

**Use Cases**:

- Feature preprocessing with classical layers
- Quantum feature extraction
- Classical classification/regression heads

Parallel Hybrid Models
----------------------

**Pattern**: Separate classical and quantum pathways combined at output

.. code-block:: python

    class ParallelHybrid(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()

            # Classical pathway
            self.classical_path = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16)
            )

            # Quantum pathway
            experiment = ML.Experiment(
                circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
                n_modes=5,
                n_photons=2,
                use_bandwidth_tuning=True
            )
            ansatz = ML.AnsatzFactory.create(
                experiment=experiment,
                input_size=input_dim,
                output_size=16
            )
            self.quantum_path = ML.QuantumLayer(input_size=input_dim, ansatz=ansatz)

            # Fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(32, output_dim * 2),  # 16 + 16 = 32 inputs
                nn.ReLU(),
                nn.Linear(output_dim * 2, output_dim)
            )

        def forward(self, x):
            # Normalize input for quantum pathway
            x_quantum = torch.sigmoid(x)

            classical_out = self.classical_path(x)
            quantum_out = self.quantum_path(x_quantum)

            # Combine outputs
            combined = torch.cat([classical_out, quantum_out], dim=1)
            return self.fusion(combined)

**Advantages**:

- Exploits both classical and quantum strengths simultaneously
- Robust to individual pathway failures
- Good for ensemble-like behavior

**Use Cases**:

- Complex pattern recognition
- Multi-modal data processing
- Uncertainty quantification

Residual Hybrid Models
----------------------

**Pattern**: Quantum layers as residual connections in classical networks

.. code-block:: python

    class ResidualHybrid(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()

            self.classical_backbone = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )

            # Quantum residual connection
            experiment = ML.Experiment(
                circuit_type=ML.CircuitType.SERIES,
                n_modes=4,
                n_photons=2
            )
            ansatz = ML.AnsatzFactory.create(
                experiment=experiment,
                input_size=64,
                output_size=64
            )
            self.quantum_residual = ML.QuantumLayer(input_size=64, ansatz=ansatz)

            self.output_layer = nn.Linear(64, output_dim)

        def forward(self, x):
            # Classical processing
            classical_out = self.classical_backbone(x)

            # Quantum residual
            quantum_residual = self.quantum_residual(torch.sigmoid(classical_out))

            # Residual connection
            combined = classical_out + quantum_residual

            return self.output_layer(combined)

**Advantages**:

- Maintains gradient flow through classical pathways
- Quantum layers provide non-linear enhancements
- Easy to integrate into existing architectures

**Use Cases**:

- Enhancing existing classical models
- Fine-tuning with quantum improvements
- Gradual quantum integration

Integration Strategies
======================

Data Flow Management
--------------------

**Input Normalization for Quantum Layers**:

.. code-block:: python

    class NormalizationLayer(nn.Module):
        def __init__(self, method='sigmoid'):
            super().__init__()
            self.method = method

        def forward(self, x):
            if self.method == 'sigmoid':
                return torch.sigmoid(x)
            elif self.method == 'tanh':
                return (torch.tanh(x) + 1) / 2  # Map [-1,1] to [0,1]
            elif self.method == 'minmax':
                x_min = x.min(dim=1, keepdim=True)[0]
                x_max = x.max(dim=1, keepdim=True)[0]
                return (x - x_min) / (x_max - x_min + 1e-8)
            else:
                raise ValueError(f"Unknown normalization method: {self.method}")

**Gradient Scaling**:

.. code-block:: python

    class GradientScaledQuantumLayer(nn.Module):
        def __init__(self, quantum_layer, scale_factor=0.1):
            super().__init__()
            self.quantum_layer = quantum_layer
            self.scale_factor = scale_factor

        def forward(self, x):
            # Scale gradients to prevent quantum layer from dominating
            if self.training:
                return self.quantum_layer(x) * self.scale_factor
            else:
                return self.quantum_layer(x)

Parameter Optimization
----------------------

**Learning Rate Scheduling**:

.. code-block:: python

    def create_hybrid_optimizer(model, classical_lr=1e-3, quantum_lr=1e-4):
        """Create separate optimizers for classical and quantum parameters."""

        classical_params = []
        quantum_params = []

        for name, param in model.named_parameters():
            if 'quantum' in name.lower():
                quantum_params.append(param)
            else:
                classical_params.append(param)

        classical_optimizer = torch.optim.Adam(classical_params, lr=classical_lr)
        quantum_optimizer = torch.optim.Adam(quantum_params, lr=quantum_lr)

        return classical_optimizer, quantum_optimizer

**Alternating Training**:

.. code-block:: python

    def train_hybrid_alternating(model, train_loader, classical_opt, quantum_opt,
                                criterion, classical_steps=5, quantum_steps=1):
        """Train classical and quantum components alternately."""

        model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Classical training phase
            if batch_idx % (classical_steps + quantum_steps) < classical_steps:
                classical_opt.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                classical_opt.step()

            # Quantum training phase
            else:
                quantum_opt.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                quantum_opt.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

Performance Optimization
========================

Memory Management
-----------------

**Gradient Checkpointing**:

.. code-block:: python

    from torch.utils.checkpoint import checkpoint

    class MemoryEfficientHybrid(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.classical_pre = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )

            # Quantum layer with checkpointing
            experiment = ML.Experiment(ML.CircuitType.SERIES, n_modes=5, n_photons=2)
            ansatz = ML.AnsatzFactory.create(experiment, input_size=64, output_size=64)
            self.quantum_layer = ML.QuantumLayer(input_size=64, ansatz=ansatz)

            self.classical_post = nn.Linear(64, output_dim)

        def forward(self, x):
            x = self.classical_pre(x)

            # Use gradient checkpointing for quantum layer
            if self.training:
                x = checkpoint(self.quantum_layer, torch.sigmoid(x))
            else:
                x = self.quantum_layer(torch.sigmoid(x))

            return self.classical_post(x)

**Batch Size Optimization**:

.. code-block:: python

    def find_optimal_batch_size(model, sample_input, max_batch_size=256):
        """Find the largest batch size that fits in memory."""

        batch_size = 1
        while batch_size <= max_batch_size:
            try:
                # Create batch
                batch = sample_input.repeat(batch_size, 1)

                # Forward pass
                model.train()
                output = model(batch)
                loss = output.sum()

                # Backward pass
                loss.backward()
                model.zero_grad()

                print(f"Batch size {batch_size}: OK")
                batch_size *= 2

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Batch size {batch_size}: Out of memory")
                    return batch_size // 2
                else:
                    raise e

        return max_batch_size

Computational Efficiency
------------------------

**Mixed Precision Training**:

.. code-block:: python

    from torch.cuda.amp import autocast, GradScaler

    def train_with_mixed_precision(model, train_loader, optimizer, criterion):
        """Train hybrid model with mixed precision."""

        scaler = GradScaler()
        model.train()

        for data, target in train_loader:
            optimizer.zero_grad()

            with autocast():
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

**Parallel Processing**:

.. code-block:: python

    class ParallelProcessingHybrid(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()

            # Multiple quantum layers for parallel processing
            self.quantum_layers = nn.ModuleList([
                self._create_quantum_layer(input_dim, 32) for _ in range(4)
            ])

            self.fusion = nn.Linear(128, output_dim)  # 4 * 32 = 128

        def _create_quantum_layer(self, input_size, output_size):
            experiment = ML.Experiment(ML.CircuitType.PARALLEL, n_modes=4, n_photons=2)
            ansatz = ML.AnsatzFactory.create(experiment, input_size, output_size)
            return ML.QuantumLayer(input_size=input_size, ansatz=ansatz)

        def forward(self, x):
            x_norm = torch.sigmoid(x)

            # Process in parallel
            outputs = []
            for quantum_layer in self.quantum_layers:
                outputs.append(quantum_layer(x_norm))

            # Combine outputs
            combined = torch.cat(outputs, dim=1)
            return self.fusion(combined)

Evaluation and Analysis
=======================

Performance Metrics
-------------------

**Component-wise Analysis**:

.. code-block:: python

    def analyze_hybrid_performance(model, test_loader):
        """Analyze performance of individual components."""

        model.eval()
        classical_times = []
        quantum_times = []
        total_times = []

        with torch.no_grad():
            for data, _ in test_loader:
                # Time classical components
                start_time = time.time()
                if hasattr(model, 'classical_pre'):
                    _ = model.classical_pre(data)
                classical_time = time.time() - start_time
                classical_times.append(classical_time)

                # Time quantum components
                start_time = time.time()
                if hasattr(model, 'quantum_layer'):
                    _ = model.quantum_layer(torch.sigmoid(data))
                quantum_time = time.time() - start_time
                quantum_times.append(quantum_time)

                # Time full forward pass
                start_time = time.time()
                _ = model(data)
                total_time = time.time() - start_time
                total_times.append(total_time)

        return {
            'classical_avg_time': np.mean(classical_times),
            'quantum_avg_time': np.mean(quantum_times),
            'total_avg_time': np.mean(total_times),
            'quantum_overhead': np.mean(quantum_times) / np.mean(total_times)
        }

**Ablation Studies**:

.. code-block:: python

    def ablation_study(base_model, test_loader, criterion):
        """Compare performance with and without quantum components."""

        results = {}

        # Test full model
        base_model.eval()
        full_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = base_model(data)
                full_loss += criterion(output, target).item()
        results['full_model'] = full_loss / len(test_loader)

        # Test without quantum layers (replace with identity)
        original_quantum = base_model.quantum_layer
        base_model.quantum_layer = nn.Identity()

        classical_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = base_model(data)
                classical_loss += criterion(output, target).item()
        results['classical_only'] = classical_loss / len(test_loader)

        # Restore quantum layer
        base_model.quantum_layer = original_quantum

        results['quantum_contribution'] = results['classical_only'] - results['full_model']

        return results

Best Practices
==============

Design Guidelines
-----------------

1. **Start Simple**: Begin with sequential architectures before exploring parallel designs
2. **Normalize Inputs**: Always normalize inputs to quantum layers to [0,1] range
3. **Scale Gradients**: Use gradient scaling to balance classical and quantum learning
4. **Monitor Components**: Track performance of individual components during training
5. **Use Checkpointing**: Implement gradient checkpointing for memory efficiency

Common Pitfalls
---------------

**Gradient Imbalance**:

.. code-block:: python

    def monitor_gradients(model):
        """Monitor gradient magnitudes across components."""

        classical_grads = []
        quantum_grads = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if 'quantum' in name.lower():
                    quantum_grads.append(grad_norm)
                else:
                    classical_grads.append(grad_norm)

        if classical_grads and quantum_grads:
            classical_avg = np.mean(classical_grads)
            quantum_avg = np.mean(quantum_grads)
            ratio = quantum_avg / classical_avg

            if ratio > 10 or ratio < 0.1:
                print(f"Warning: Gradient imbalance detected. Ratio: {ratio:.2f}")

**Overfitting Detection**:

.. code-block:: python

    def detect_component_overfitting(model, train_loader, val_loader, criterion):
        """Detect if specific components are overfitting."""

        model.eval()

        # Evaluate on training set
        train_loss = 0
        with torch.no_grad():
            for data, target in train_loader:
                output = model(data)
                train_loss += criterion(output, target).item()
        train_loss /= len(train_loader)

        # Evaluate on validation set
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
        val_loss /= len(val_loader)

        overfitting_ratio = val_loss / train_loss

        if overfitting_ratio > 1.5:
            print(f"Warning: Potential overfitting detected. Ratio: {overfitting_ratio:.2f}")
            return True

        return False

Future Directions
=================

Adaptive Architectures
----------------------

Research directions for self-optimizing hybrid models:

1. **Neural Architecture Search**: Automatic discovery of optimal hybrid topologies
2. **Dynamic Component Weighting**: Adaptive importance weighting of classical vs quantum components
3. **Resource-Aware Design**: Automatic architecture adaptation based on available computational resources

Advanced Integration Patterns
-----------------------------

Emerging patterns for hybrid model design:

- **Attention-Based Fusion**: Using attention mechanisms to dynamically weight classical and quantum features
- **Multi-Scale Processing**: Quantum processing at multiple resolution levels
- **Federated Hybrid Learning**: Distributed training of hybrid models across quantum and classical devices