# 🧙‍♂️ Merlin - Photonic Quantum Neural Networks



Merlin is a comprehensive framework for integrating **photonic quantum circuits** into PyTorch neural networks with full automatic differentiation support. Build hybrid classical-quantum models with quantum advantage for machine learning.

## 🚀 Key Features

- **🔧 Easy Integration**: Drop-in quantum layers for existing PyTorch models
- **⚡ Auto-Differentiation**: Full gradient support through quantum circuits  
- **🎯 Multiple Architectures**: PARALLEL_COLUMNS, SERIES, and PARALLEL circuit types
- **🔊 Realistic Noise**: Shot noise and sampling simulation with multiple methods
- **🏗️ Factory Pattern**: Auto-generated circuits or custom circuit support
- **🎛️ Bandwidth Tuning**: Learnable feature encoding optimization
- **🔄 Reservoir Computing**: Fixed random quantum layers for efficient training
- **📊 Flexible Outputs**: Multiple mapping strategies (linear, grouping, modulo)

## 📦 Installation

```bash
pip install merlinquantum
```

### Development Installation
```bash
git clone https://github.com/merlinquantum/merlin.git
cd merlin
pip install -e
```

## 🎯 Quick Start

### Basic Quantum Layer

```python
import torch
import merlin as ML

# Create quantum experiment configuration
experiment = ML.PhotonicBackend(
    circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
    n_modes=4,
    n_photons=2,
    use_bandwidth_tuning=True
)

# Generate quantum ansatz
ansatz = ML.AnsatzFactory.create(
    PhotonicBackend=experiment,
    input_size=3,
    output_size=4
)

# Create quantum layer
quantum_layer = ML.QuantumLayer(input_size=3, ansatz=ansatz)

# Use in your model
x = torch.rand(10, 3)
output = quantum_layer(x)  # Shape: [10, 4]
```

### Hybrid Neural Network

```python
import torch.nn as nn


class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical = nn.Linear(8, 3)

        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.SERIES,
            n_modes=5,
            n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=3,
            output_size=6
        )

        self.quantum = ML.QuantumLayer(input_size=3, ansatz=ansatz)
        self.output = nn.Linear(6, 2)

    def forward(self, x):
        x = self.classical(x)
        x = torch.sigmoid(x)  # Normalize for quantum layer
        x = self.quantum(x)
        return self.output(x)


model = HybridModel()
```

## 🏗️ Architecture Overview

### Circuit Types

| Type | Description | Use Case |
|------|-------------|-----------|
| **PARALLEL_COLUMNS** | Cartesian product encoding | Complex feature interactions |
| **SERIES** | Sequential processing with interactions | Feature combination learning |
| **PARALLEL** | Direct feature-to-parameter mapping | Simple quantum transformations |

### State Patterns

- **PERIODIC**: Alternating photon placement (default)
- **SPACED**: Evenly distributed photons  
- **SEQUENTIAL**: Consecutive photon placement
- **DEFAULT**: Alias for PERIODIC

### Output Mapping

- **LINEAR**: Standard neural network linear layer
- **LEXGROUPING**: Equal-sized bucket grouping
- **MODGROUPING**: Modulo-based index grouping  
- **NONE**: Direct probability distribution output

## 🔬 Advanced Features

### Reservoir Computing

```python
experiment = ML.PhotonicBackend(
    circuit_type=ML.CircuitType.PARALLEL,
    n_modes=6,
    n_photons=3,
    reservoir_mode=True  # Fixed random parameters
)

reservoir_layer = ML.QuantumLayer(input_size=4, ansatz=ansatz)
# Reduced trainable parameters for efficient training
```

### Sampling and Noise Simulation

```python
layer = ML.QuantumLayer(input_size=3, ansatz=ansatz, shots=1000)

# Configure sampling
layer.set_sampling_config(shots=500, method='gaussian')

# Manual sampling control
output = layer(x, apply_sampling=True, shots=100)
```

### Custom Circuits (Backward Compatible)

```python
import perceval as pcvl

# Build custom circuit
circuit = pcvl.Circuit(4)
circuit.add(0, pcvl.GenericInterferometer(4, pcvl.InterferometerShape.RECTANGLE))
# ... add more components

layer = ML.QuantumLayer(
    input_size=2,
    circuit=circuit,
    input_state=[1, 1, 0, 0],
    trainable_parameters=["phi_"],
    input_parameters=["pl"]
)
```

## 📊 Performance & Benchmarks

```python
# Run benchmarks
python -m merlin.examples.benchmarks

# Or use the CLI
merlin-benchmark --circuit_types all --n_modes 4,6,8 --batch_sizes 32,64,128
```

## 🧪 Examples

Check out the `examples/` directory for comprehensive tutorials:

- **`basic_usage.py`**: Simple quantum layers and hybrid networks
- **`advanced_usage.py`**: Custom circuits, training loops, multi-layer networks
- **`benchmarks.py`**: Performance evaluation and comparison

## 🔧 Dependencies

**Core Requirements:**
- `torch >= 1.12.0`
- `perceval-quandela >= 0.8.0` 
- `pcvl-pytorch >= 0.1.0`
- `numpy >= 1.21.0`

**Development:**
- `pytest >= 6.0` (testing)
- `black >= 22.0` (code formatting)
- `sphinx >= 4.0` (documentation)

## 📚 API Reference

### Core Classes

- **`QuantumLayer`**: Main quantum neural network layer
- **`Experiment`**: Configuration container for quantum setups
- **`Ansatz`**: Complete circuit configuration with auto-generation
- **`AnsatzFactory`**: Factory for creating ansatz configurations

### Enums

- **`CircuitType`**: Available circuit topologies
- **`StatePattern`**: Input photon distribution patterns  
- **`OutputMappingStrategy`**: Quantum-to-classical output conversion

## 🧑‍💻 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes with tests: `pytest merlin/tests/`
4. Submit a pull request

### Development Setup

```bash
git clone https://github.com/merlin-team/merlin.git
cd merlin
pip install -e ".[dev]"
pre-commit install  # Install git hooks
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [Perceval](https://perceval.quandela.net/) quantum photonic framework
- Inspired by advances in quantum machine learning research
- Thanks to the quantum computing and PyTorch communities

## 📞 Support

- **Documentation**: [https://merlinquantum.ai/](https://merlinquantum.ai/)
- **Issues**: [GitHub Issues](https://github.com/merlin-team/merlin/issues)
- **Discussions**: [GitHub Discussions](https://github.com/merlin-team/merlin/discussions)

---

**⚡ Harness the power of quantum photonics in your neural networks with Merlin!**