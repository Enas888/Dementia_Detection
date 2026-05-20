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
# ENCODING
# =========================================================

def angle_embedding(x):
    qml.AngleEmbedding(
        features=x,
        wires=range(n_qubits),
        rotation="Y"
    )
    
# =========================================================
# VARIATIONAL LAYER
# =========================================================

def variational_layer(theta):

    # trainable rotations
    for i in range(n_qubits):

        qml.RY(theta[i][0], wires=i)
        qml.RZ(theta[i][1], wires=i)

    # all-to-all CRX entanglement
    idx = 0

    for i in range(n_qubits):

        for j in range(i + 1, n_qubits):

            qml.CRX(
                theta[idx][2],
                wires=[i, j]
            )

            idx += 1

# =========================================================
# CIRCUIT
# =========================================================

@qml.qnode(dev, interface="autograd")

def quantum_circuit(x, theta):

    angle_embedding(x)

    variational_layer(theta)

    return qml.math.stack([
        qml.expval(qml.PauliZ(0)),
        qml.expval(qml.PauliZ(1)),
        qml.expval(qml.PauliZ(2))
    ])

# =========================================================
# MODEL
# =========================================================

class VQC:

    def __init__(self, n_qubits=10):

        self.n_qubits = n_qubits

        # 45 entangling pairs for 10 qubits
        n_pairs = (n_qubits * (n_qubits - 1)) // 2

        self.theta = np.random.uniform(
            0,
            2 * np.pi,
            (n_pairs, 3),
            requires_grad=True
        )

    def forward(self, x):

        return quantum_circuit(x, self.theta)

    def predict(self, X):

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