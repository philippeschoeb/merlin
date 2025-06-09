:github_url: https://github.com/merlinquantum/merlin

========================
Your First Quantum Layer
========================

Merlin follows a simple workflow for creating quantum neural networks:

1. **Define Circuit**: Create the quantum circuit using Perceval
2. **Build Layer**: Integrate into PyTorch neural networks


Step 1: Basic Example
---------------------

Let's start with a simple classification task using the direct circuit definition - in this case, we will define
a quantum circuit manually using Perceval's API, and then create a quantum layer that can be used in a PyTorch model.

Don't worry if you are not familiar with Perceval or quantum circuits yet! This example will guide you through the
process step by step and we will see in the next sections how to use the high-level API to create quantum layers without
needing to define the circuit manually.

Circuit Definition
^^^^^^^^^^^^^^^^^^

The quantum circuit consists of three main parts:

.. code-block:: python

    def create_quantum_circuit(m):
        # 1. Left interferometer - trainable transformation
        wl = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_li{i}")) //
                     pcvl.BS() // pcvl.PS(pcvl.P(f"theta_lo{i}")),
            shape=pcvl.InterferometerShape.RECTANGLE
        )

        # 2. Input encoding - maps classical data to quantum parameters
        c_var = pcvl.Circuit(m)
        for i in range(4):  # 4 input features
            px = pcvl.P(f"px{i + 1}")
            c_var.add(i + (m - 4) // 2, pcvl.PS(px))

        # 3. Right interferometer - trainable transformation
        wr = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ri{i}")) //
                     pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ro{i}")),
            shape=pcvl.InterferometerShape.RECTANGLE
        )

**Key Components**:

- ``pcvl.BS()``: Beam splitter for quantum interference
- ``pcvl.PS(pcvl.P("name"))``: Phase shifter with trainable parameter
- ``pcvl.GenericInterferometer``: Creates complex interference patterns
- ``pcvl.Circuit``: Container for quantum components

Step 2: Create Quantum Layer
----------------------------

.. code-block:: python

    import torch
    import torch.nn as nn
    import perceval as pcvl
    import merlin as ML

    # Create quantum layer
    m = 6  # 6 optical modes
    circuit = create_quantum_circuit(m)

    quantum_layer = ML.QuantumLayer(
        input_size=4,                                              # 4 input features
        output_size=3,                                             # 3 output classes
        circuit=circuit,                                           # Quantum circuit
        trainable_parameters=["theta"],                            # Parameters to train
        input_parameters=["px"],                                   # Input encoding parameters
        input_state=[1, 0, 1, 0, 1, 0],           # Initial photon state
        output_mapping_strategy=ML.OutputMappingStrategy.LINEAR    # Output mapping
    )

    # Test the layer
    x = torch.rand(10, 4)  # Batch of 10 samples, 4 features each
    output = quantum_layer(x)
    print(f"Input shape: {x.shape}")    # [10, 4]
    print(f"Output shape: {output.shape}")  # [10, 3]

Step 3: Understanding the Components
------------------------------------

Quantum Layer Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    quantum_layer = ML.QuantumLayer(
        input_size=4,                                              # Classical input features
        output_size=3,                                             # Desired output size
        circuit=circuit,                                           # Quantum circuit
        trainable_parameters=["theta"],                            # Which parameters to train
        input_parameters=["px"],                                   # Input encoding parameters
        input_state=[1, 0] * (m // 2) + [0] * (m % 2),           # Initial photon distribution
        no_bunching=False,                                         # Allow photon bunching
        output_mapping_strategy=ML.OutputMappingStrategy.LINEAR    # How to map quantum output
    )

**Parameter Explanation**:

- ``trainable_parameters``: Parameters updated during backpropagation
- ``input_parameters``: Parameters that encode classical input data
- ``input_state``: Initial photon configuration (e.g., [1,0,1,0,0,0] = photons in modes 0,2)
- ``no_bunching``: Whether multiple photons can occupy the same mode
- ``output_mapping_strategy``: How quantum probabilities become classical outputs

Step 3: Hybrid Neural Network
-----------------------------

Integrate quantum layers into classical networks:

.. code-block:: python

    class HybridClassifier(nn.Module):
        def __init__(self, input_dim=8, n_classes=3, n_modes=6):
            super().__init__()

            # Classical preprocessing
            self.classical_input = nn.Sequential(
                nn.Linear(input_dim, 6),
                nn.ReLU(),
                nn.Linear(6, 4)  # Reduce to quantum layer input size
            )

            # Create quantum circuit
            circuit = create_quantum_circuit(n_modes)

            # Quantum processing layer
            self.quantum_layer = ML.QuantumLayer(
                input_size=4,
                output_size=6,
                circuit=circuit,
                trainable_parameters=["theta"],
                input_parameters=["px"],
                input_state=[1, 0] * (n_modes // 2) + [0] * (n_modes % 2),
                output_mapping_strategy=ML.OutputMappingStrategy.LINEAR
            )

            # Classical output layer
            self.classifier = nn.Sequential(
                nn.Linear(6, n_classes),
                nn.Softmax(dim=1)
            )

        def forward(self, x):
            # Classical preprocessing
            x = self.classical_input(x)

            # Normalize for quantum layer (required: inputs must be in [0,1])
            x = torch.sigmoid(x)

            # Quantum transformation
            x = self.quantum_layer(x)

            # Classical output
            return self.classifier(x)

    # Create and test model
    model = HybridClassifier(input_dim=8, n_classes=3, n_modes=6)
    x = torch.rand(16, 8)  # Batch of 16 samples
    output = model(x)
    print(f"Model output shape: {output.shape}")  # [16, 3]

Step 4: Training Loop
---------------------

Train your hybrid model with standard PyTorch workflows:

.. code-block:: python

    import torch.optim as optim
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000, n_features=8, n_classes=3,
        n_informative=6, random_state=42
    )

    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Setup training
    model = HybridClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(50):
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Evaluation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
                test_acc = (test_outputs.argmax(1) == y_test).float().mean()

            print(f"Epoch {epoch}: Loss={loss:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
            model.train()

