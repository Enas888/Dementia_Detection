import pennylane as qml
import pennylane.numpy as np

# =========================================================
# DEVICE
# =========================================================

n_qubits = 10

dev = qml.device(
    "default.qubit",
    wires=n_qubits
)


# =========================================================
# ANGLE ENCODING
# =========================================================

def angle_embedding(x):
    """
    Encode classical features into quantum states using RY rotations.
    x shape: (10,)
    """
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)


# =========================================================
# VARIATIONAL LAYER
# =========================================================

def variational_layer(theta):
    """
    One trainable layer:
    - RY, RZ rotations
    - CNOT entanglement chain
    """

    # Single-qubit trainable rotations
    for i in range(n_qubits):
        qml.RY(theta[i][0], wires=i)
        qml.RZ(theta[i][1], wires=i)

    # Entanglement: CNOT chain
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

# =========================================================
# FULL QUANTUM CIRCUIT
# =========================================================

@qml.qnode(dev, interface="autograd")

def quantum_circuit(x, theta):
    """
    Full VQC model:
    Encoding → Variational layer → Measurement
    """

    # Step 1: Encoding
    angle_embedding(x)

    # Step 2: Variational block (1 layer)
    variational_layer(theta)

    # Step 3: Measurement (Z expectation per qubit)
    return qml.math.stack(
        [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
)


# =========================================================
# MODEL CLASS
# =========================================================

class VQC:
    """
    Variational Quantum Classifier wrapper
    """

    def __init__(self, n_qubits=10):

        self.n_qubits = n_qubits

        # theta shape: (n_qubits, 3)
        # [RY, RZ, CNOT]
        self.theta = np.random.uniform(
            0, 2 * np.pi,
            (n_qubits, 3),
            requires_grad=True
        )

    def forward(self, x):
        return quantum_circuit(x, self.theta)

    def predict(self, X):
        """
        Simple classifier:
        - Aggregate Z expectations
        - Threshold for class decision (can extend to softmax later)
        """ 
        
        outputs = []

        for x in X:

            z_exp = self.forward(x)

            score = float(np.mean(z_exp))

            outputs.append(score)

        return np.array(outputs)

    def get_params(self):
        return self.theta

    def set_params(self, theta):
        self.theta = theta