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
# ARCHITECTURE CONSTANTS
#
# VQC5 — 3-layer CRX chain with data re-uploading
#
# Theta layout per layer:
#   theta[layer, qubit, 0] → RY angle
#   theta[layer, qubit, 1] → RZ angle
#   theta[layer, qubit, 2] → CRX angle (qubit i → i+1)
#                            (unused for last qubit, kept for uniform shape)
#
# theta shape: (N_LAYERS, n_qubits, 3)
# Total params: 3 × 10 × 3 = 90
# =========================================================

N_LAYERS = 3


# =========================================================
# ANGLE ENCODING
# =========================================================

def angle_embedding(x):
    """
    Angle encoding: maps each feature x_i to RY(x_i) on qubit i.
    Features must be in [-pi, pi] — handled by the training script.
    """
    qml.AngleEmbedding(
        features=x,
        wires=range(n_qubits),
        rotation="Y"
    )


# =========================================================
# VARIATIONAL LAYER
# =========================================================

def variational_layer(theta_layer):
    """
    One trainable block:
      - RY + RZ on every qubit  (single-qubit rotations)
      - CRX chain i → i+1       (trainable entanglement)

    Args:
        theta_layer: shape (n_qubits, 3)
            theta_layer[i, 0] → RY on qubit i
            theta_layer[i, 1] → RZ on qubit i
            theta_layer[i, 2] → CRX control=i, target=i+1
    """
    # Single-qubit rotations
    for i in range(n_qubits):
        qml.RY(theta_layer[i][0], wires=i)
        qml.RZ(theta_layer[i][1], wires=i)

    # CRX entanglement chain
    for i in range(n_qubits - 1):
        qml.CRX(theta_layer[i][2], wires=[i, i + 1])


# =========================================================
# FULL QUANTUM CIRCUIT
# =========================================================

@qml.qnode(dev, interface="autograd")
def quantum_circuit(x, theta):
    """
    Full VQC5 circuit:
      1. Angle encode input x
      2. N_LAYERS variational blocks, each preceded by re-encoding
         (data re-uploading for richer expressibility)
      3. Measure PauliZ on qubits 0, 1, 2 → 3 logits for 3 classes

    Args:
        x:     shape (n_qubits,)           — one sample
        theta: shape (N_LAYERS, n_qubits, 3) — all parameters

    Returns:
        stack of 3 expectation values in [-1, 1]
    """
    for layer in range(N_LAYERS):
        angle_embedding(x)          # re-upload data at every layer
        variational_layer(theta[layer])

    return qml.math.stack([
        qml.expval(qml.PauliZ(0)),
        qml.expval(qml.PauliZ(1)),
        qml.expval(qml.PauliZ(2))
    ])


# =========================================================
# MODEL CLASS
# =========================================================

class VQC:
    """
    Variational Quantum Classifier — VQC5

    Architecture: 3-layer CRX chain with data re-uploading
    Parameters:   theta shape (N_LAYERS, n_qubits, 3) = (3, 10, 3) = 90 total
    """

    def __init__(self, n_qubits=10, n_layers=N_LAYERS, seed=42):

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        np.random.seed(seed)

        # theta shape: (n_layers, n_qubits, 3) → [RY, RZ, CRX]
        self.theta = np.random.uniform(
            0, 2 * np.pi,
            (n_layers, n_qubits, 3),
            requires_grad=True
        )

    def forward(self, x):
        return quantum_circuit(x, self.theta)

    def predict(self, X):
        """
        Predict class labels for a batch of samples.
        Maps 3 PauliZ expectations to a class index via argmax.
        """
        predictions = []

        for x in X:
            z_exp = self.forward(x)
            # argmax over the 3 qubit expectations → class {0, 1, 2}
            pred_class = int(np.argmax(z_exp))
            predictions.append(pred_class)

        return np.array(predictions)

    def predict_proba(self, X):
        """
        Returns raw Z-expectation scores (3 values per sample).
        Can be passed to softmax for probability estimates.
        """
        scores = []

        for x in X:
            z_exp = self.forward(x)
            scores.append(z_exp)

        return np.array(scores)

    def get_params(self):
        return self.theta

    def set_params(self, theta):
        self.theta = theta

    def param_count(self):
        return self.n_layers * self.n_qubits * 3