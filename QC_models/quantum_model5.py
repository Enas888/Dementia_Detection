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
# VQC4 — All-to-All CRX (fully connected entanglement)
#
# Theta layout per layer:
#   [0 : n_qubits]          → RY rotation angles
#   [n_qubits : 2*n_qubits] → RZ rotation angles
#   [2*n_qubits : 2*n_qubits + n_pairs] → CRX angles
#
# n_pairs = n_qubits*(n_qubits-1)//2 = 45 for 10 qubits
# Total params per layer = 10 + 10 + 45 = 65
# With N_LAYERS=3: total params = 195
# =========================================================

N_LAYERS = 3
N_PAIRS  = (n_qubits * (n_qubits - 1)) // 2  # 45
PARAMS_PER_LAYER = 2 * n_qubits + N_PAIRS     # 65

# =========================================================
# ENCODING
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
    One variational block:
        - RY + RZ on every qubit  (trainable single-qubit rotations)
        - CRX between all pairs   (trainable entanglement)

    Args:
        theta_layer: shape (PARAMS_PER_LAYER,) = (65,)
            theta_layer[0      :10]  → RY angles
            theta_layer[10     :20]  → RZ angles
            theta_layer[20     :65]  → CRX angles (ordered i<j)
    """
    # --- single-qubit rotations ---
    for i in range(n_qubits):
        qml.RY(theta_layer[i],            wires=i)
        qml.RZ(theta_layer[n_qubits + i], wires=i)

    # --- all-to-all CRX entanglement ---
    crx_offset = 2 * n_qubits
    idx = 0
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            qml.CRX(
                theta_layer[crx_offset + idx],
                wires=[i, j]
            )
            idx += 1

# =========================================================
# CIRCUIT
# =========================================================

@qml.qnode(dev, interface="autograd")
def quantum_circuit(x, theta):
    """
    Full VQC4 circuit:
        1. Angle encode input x
        2. N_LAYERS variational blocks, each preceded by re-encoding
           (data re-uploading for richer expressibility)
        3. Measure PauliZ on qubits 0, 1, 2  → 3 logits for 3 classes

    Args:
        x:     shape (n_qubits,)             — one sample
        theta: shape (N_LAYERS, PARAMS_PER_LAYER)  — all parameters

    Returns:
        stack of 3 expectation values in [-1, 1]
    """
    for layer in range(N_LAYERS):
        # Re-upload data at every layer — key trick for expressibility
        angle_embedding(x)
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
    Variational Quantum Classifier — VQC4 (all-to-all CRX).

    Parameters
    ----------
    n_qubits : int
        Number of qubits (= number of input features). Default 10.
    n_layers : int
        Number of variational + re-encoding blocks. Default N_LAYERS=3.

    Theta shape:  (n_layers, PARAMS_PER_LAYER)
                = (3, 65) = 195 trainable parameters total.

    Initialised from a narrow Gaussian near zero to avoid
    barren plateaus (uniform [0, 2pi] causes vanishing gradients).
    """

    def __init__(self, n_qubits=10, n_layers=N_LAYERS):

        self.n_qubits  = n_qubits
        self.n_layers  = n_layers
        self.n_pairs   = (n_qubits * (n_qubits - 1)) // 2
        self.params_per_layer = 2 * n_qubits + self.n_pairs

        # Small random init — avoids barren plateau
        self.theta = np.random.normal(
            loc=0.0,
            scale=0.1,
            size=(self.n_layers, self.params_per_layer),
            requires_grad=True
        )

    def forward(self, x):
        """
        Run the circuit for a single sample x.

        Returns
        -------
        logits : array of shape (3,), values in [-1, 1]
        """
        return quantum_circuit(x, self.theta)

    def predict_proba(self, X):
        """
        Compute softmax probabilities for a batch of samples.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        probs : array of shape (n_samples, 3)
        """
        results = []
        for x in X:
            logits = self.forward(x)
            exp_l  = np.exp(logits - np.max(logits))
            probs  = exp_l / np.sum(exp_l)
            results.append(probs)
        return np.array(results)

    def predict(self, X):
        """
        Argmax class prediction for a batch.

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def get_params(self):
        return self.theta

    def set_params(self, theta):
        self.theta = theta