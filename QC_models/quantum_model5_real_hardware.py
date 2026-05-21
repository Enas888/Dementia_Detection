import time
import pennylane as qml
import pennylane.numpy as np

# =========================================================
# HARDWARE CONFIGURATION
#
# To run on real quantum hardware, PennyLane connects via:
#
#   1. IBM Quantum  →  pip install pennylane-qiskit
#      device: "qiskit.ibmq"
#      requires: IBMQ API token from https://quantum.ibm.com
#
#   2. Amazon Braket →  pip install amazon-braket-pennylane-plugin
#      device: "braket.aws.qubit"
#      requires: AWS credentials + S3 bucket
#
#   3. IQM (superconducting) → pip install pennylane-iqm
#
# Set USE_HARDWARE = True and fill in your token/backend below.
# Set USE_HARDWARE = False to benchmark on simulator first.
# =========================================================

USE_HARDWARE = True         # ← flip to True for real QPU

# ------ IBM Quantum settings (edit these) ------
IBMQ_TOKEN   = "crn:v1:bluemix:public:quantum-computing:us-east:a/37b2529ceff642eb94b4df41cc4b550a:659bac74-8ec2-4c3f-b0dc-a02a00494426::"
IBMQ_BACKEND = "ibm_marrakesh"     # or "ibm_kyoto", "ibm_brisbane" etc.
#                                  check available at quantum.ibm.com
# -----------------------------------------------

n_qubits = 10

# =========================================================
# DEVICE FACTORY
# Returns either a simulator or real hardware device,
# and a label string used in timing output.
# =========================================================

def make_device(use_hardware: bool):
    """
    Build the PennyLane device.

    Simulator  : 'default.qubit'   — exact statevector, CPU
    Hardware   : 'qiskit.ibmq'     — real superconducting QPU via IBM
    """
    if not use_hardware:
        dev   = qml.device("default.qubit", wires=n_qubits)
        label = "Simulator (default.qubit)"

    else:
        # Authenticate with IBM Quantum (qiskit-ibm-runtime replaces qiskit-ibm-provider)
        from qiskit_ibm_runtime import QiskitRuntimeService

        QiskitRuntimeService.save_account(
            channel="ibm_quantum",
            token=IBMQ_TOKEN,
            overwrite=True
        )
        service = QiskitRuntimeService(channel="ibm_quantum")

        dev = qml.device(
            "qiskit.remote",
            wires=n_qubits,
            backend=service.backend(IBMQ_BACKEND),
            shots=1024
        )
        label = f"Hardware ({IBMQ_BACKEND}, 1024 shots)"

    return dev, label


dev, DEVICE_LABEL = make_device(USE_HARDWARE)

print(f"✓ Device: {DEVICE_LABEL}")

# =========================================================
# ARCHITECTURE CONSTANTS  (VQC4 — All-to-All CRX)
# =========================================================

N_LAYERS         = 3
N_PAIRS          = (n_qubits * (n_qubits - 1)) // 2   # 45
PARAMS_PER_LAYER = 2 * n_qubits + N_PAIRS              # 65
# Total params = 3 × 65 = 195

# =========================================================
# ENCODING
# =========================================================

def angle_embedding(x):
    """Angle encoding: RY(x_i) on qubit i. Features in [-π, π]."""
    qml.AngleEmbedding(features=x, wires=range(n_qubits), rotation="Y")

# =========================================================
# VARIATIONAL LAYER
# =========================================================

def variational_layer(theta_layer):
    """
    One variational block:
      - RY + RZ on every qubit
      - CRX between all pairs (all-to-all entanglement)

    theta_layer shape: (PARAMS_PER_LAYER,) = (65,)
      [0:10]   → RY
      [10:20]  → RZ
      [20:65]  → CRX (ordered i < j)
    """
    for i in range(n_qubits):
        qml.RY(theta_layer[i],            wires=i)
        qml.RZ(theta_layer[n_qubits + i], wires=i)

    crx_offset = 2 * n_qubits
    idx = 0
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            qml.CRX(theta_layer[crx_offset + idx], wires=[i, j])
            idx += 1

# =========================================================
# CIRCUIT
# =========================================================

@qml.qnode(dev, interface="autograd")
def quantum_circuit(x, theta):
    """
    Full VQC4 circuit with data re-uploading.

    x     : (n_qubits,)
    theta : (N_LAYERS, PARAMS_PER_LAYER)
    returns: stack of 3 PauliZ expectations → 3-class logits
    """
    for layer in range(N_LAYERS):
        angle_embedding(x)
        variational_layer(theta[layer])

    return qml.math.stack([
        qml.expval(qml.PauliZ(0)),
        qml.expval(qml.PauliZ(1)),
        qml.expval(qml.PauliZ(2))
    ])

# =========================================================
# TIMING BENCHMARK
# Measures single-forward-pass latency on current device.
# Call this before and after switching USE_HARDWARE.
# =========================================================

def benchmark(theta, x_sample, n_runs=5, label=None):
    """
    Run the circuit n_runs times and report latency statistics.

    Args:
        theta    : parameter array (N_LAYERS, PARAMS_PER_LAYER)
        x_sample : one input sample (n_qubits,)
        n_runs   : number of repeated calls
        label    : optional description printed in the report
    """
    label = label or DEVICE_LABEL
    times = []

    print(f"\n{'='*60}")
    print(f" Benchmark — {label}")
    print(f"{'='*60}")
    print(f" Circuit : VQC4 | {n_qubits} qubits | {N_LAYERS} layers | {N_LAYERS * PARAMS_PER_LAYER} params")
    print(f" Runs    : {n_runs}")
    print(f"{'-'*60}")

    # Warm-up pass (compilation / queue overhead excluded)
    _ = quantum_circuit(x_sample, theta)

    for run in range(n_runs):
        t0     = time.perf_counter()
        result = quantum_circuit(x_sample, theta)
        t1     = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        print(f"  Run {run+1:02d} | {elapsed:.4f}s | logits: {[round(float(v), 4) for v in result]}")

    mean_t = sum(times) / len(times)
    min_t  = min(times)
    max_t  = max(times)

    print(f"{'-'*60}")
    print(f"  Mean : {mean_t:.4f}s")
    print(f"  Min  : {min_t:.4f}s")
    print(f"  Max  : {max_t:.4f}s")
    print(f"{'='*60}\n")

    return {
        "device": label,
        "mean_s": round(mean_t, 4),
        "min_s":  round(min_t,  4),
        "max_s":  round(max_t,  4),
        "runs":   n_runs,
    }

# =========================================================
# MODEL CLASS
# =========================================================

class VQC:
    """
    Variational Quantum Classifier — VQC4 (All-to-All CRX)

    Parameters
    ----------
    n_qubits : int   — number of qubits = input features (default 10)
    n_layers : int   — variational + re-encoding blocks  (default 3)

    Theta shape : (n_layers, PARAMS_PER_LAYER) = (3, 65) = 195 params
    Init        : narrow Gaussian N(0, 0.1) — avoids barren plateaus
    """

    def __init__(self, n_qubits=10, n_layers=N_LAYERS):

        self.n_qubits         = n_qubits
        self.n_layers         = n_layers
        self.n_pairs          = (n_qubits * (n_qubits - 1)) // 2
        self.params_per_layer = 2 * n_qubits + self.n_pairs

        self.theta = np.random.normal(
            loc=0.0,
            scale=0.1,
            size=(self.n_layers, self.params_per_layer),
            requires_grad=True
        )

    # ----------------------------------------------------------
    def forward(self, x):
        """Single-sample forward pass → logits (3,)."""
        return quantum_circuit(x, self.theta)

    # ----------------------------------------------------------
    def predict_proba(self, X):
        """Softmax probabilities for a batch → (n_samples, 3)."""
        results = []
        for x in X:
            logits = self.forward(x)
            exp_l  = np.exp(logits - np.max(logits))
            results.append(exp_l / np.sum(exp_l))
        return np.array(results)

    # ----------------------------------------------------------
    def predict(self, X):
        """Argmax class labels for a batch → (n_samples,)."""
        return np.argmax(self.predict_proba(X), axis=1)

    # ----------------------------------------------------------
    def benchmark(self, x_sample=None, n_runs=5):
        """
        Convenience wrapper — benchmarks the current device
        using a random sample if none is provided.
        """
        if x_sample is None:
            x_sample = np.random.uniform(-np.pi, np.pi, self.n_qubits)
        return benchmark(self.theta, x_sample, n_runs=n_runs)

    # ----------------------------------------------------------
    def get_params(self):
        return self.theta

    def set_params(self, theta):
        self.theta = theta

    def param_count(self):
        return self.n_layers * self.params_per_layer


# =========================================================
# QUICK TEST — run this file directly to see timing
#
#   python quantum_model4_hardware.py
#
# To compare simulator vs hardware:
#   1. Run with USE_HARDWARE = False  → note mean_s
#   2. Set USE_HARDWARE = True, add token
#   3. Run again → compare mean_s
# =========================================================

if __name__ == "__main__":

    model    = VQC()
    x_sample = np.random.uniform(-np.pi, np.pi, n_qubits)

    timing = model.benchmark(x_sample, n_runs=5)

    print("Timing result dict:")
    print(timing)