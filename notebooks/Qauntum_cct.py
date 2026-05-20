# =========================================================
# QUANTUM CIRCUIT ANALYSIS
# =========================================================

import pennylane as qml
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
# =========================================================
# FIX PROJECT IMPORTS
# =========================================================

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(ROOT_DIR)
from QC_models.quantum_model1 import VQC as VQC1
from QC_models.quantum_model2 import VQC as VQC2
from QC_models.quantum_model3 import VQC as VQC3
from QC_models.quantum_model4 import VQC as VQC4

# =========================================================
# SAVE DIRECTORY
# =========================================================

BASE_DIR = r"C:\Users\emade\Downloads\Dementia_Detection"

SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")

RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)
CCT_INFO_DIR = os.path.join(
    RESULTS_DIR,
    "Quantum_Circuit_Analysis"
)

os.makedirs(CCT_INFO_DIR, exist_ok=True)

# =========================================================
# ANALYSIS FUNCTION
# =========================================================

def analyze_circuit(
    ModelClass,
    model_name,
    n_qubits=10
):

    print("\n" + "="*60)
    print(f"{model_name} ANALYSIS")
    print("="*60)

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    model = ModelClass(n_qubits=n_qubits)

    # dummy input
    x = np.random.uniform(
        0,
        np.pi,
        n_qubits
    )

    theta = model.theta

    # -----------------------------------------------------
    # DRAW CIRCUIT
    # -----------------------------------------------------

    fig, ax = qml.draw_mpl(
        model.forward
    )(x)

    circuit_path = os.path.join(
        CCT_INFO_DIR,
        f"{model_name}_circuit.png"
    )

    fig.savefig(
        circuit_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"✓ Circuit plot saved -> {circuit_path}")

    # -----------------------------------------------------
    # PARAMETER COUNT
    # -----------------------------------------------------

    n_params = theta.size

    print(f"\nTrainable Parameters : {n_params}")

    # -----------------------------------------------------
    # QUBITS
    # -----------------------------------------------------

    print(f"Number of Qubits     : {n_qubits}")

    # -----------------------------------------------------
    # BUILD TAPE MANUALLY
    # -----------------------------------------------------

    qml.capture.disable()

    qnode = model.forward.__globals__["quantum_circuit"]

    tape = qml.workflow.construct_tape(qnode)(
        x,
        theta
    )

    ops = tape.operations

    # -----------------------------------------------------
    # CIRCUIT DEPTH
    # -----------------------------------------------------

    depth = len(ops)

    print(f"Circuit Depth        : {depth}")

    # -----------------------------------------------------
    # GATE COUNT
    # -----------------------------------------------------

    gate_count = len(ops)

    print(f"Total Gates          : {gate_count}")


    # -----------------------------------------------------
    # GATE BREAKDOWN
    # -----------------------------------------------------

    gate_dict = {}

    for op in ops:

        gate_name = op.name

        if gate_name not in gate_dict:
            gate_dict[gate_name] = 0

        gate_dict[gate_name] += 1

    print("\nGate Breakdown:")

    for gate, count in gate_dict.items():
        print(f"  {gate:<10} : {count}")

    # -----------------------------------------------------
    # MEASUREMENTS
    # -----------------------------------------------------

    print(f"\nMeasurements         : {len(tape.measurements)}")

    # -----------------------------------------------------
    # ENTANGLING GATES
    # -----------------------------------------------------

    entangling_gates = [
        op.name for op in ops
        if len(op.wires) > 1
    ]

    print(f"Entangling Gates     : {len(entangling_gates)}")

    # -----------------------------------------------------
    # CONNECTIVITY TYPE
    # -----------------------------------------------------

    if model_name == "VQC1":
        connectivity = "Linear CRX Chain"

    elif model_name == "VQC2":
        connectivity = "Linear CNOT Chain"

    elif model_name == "VQC3":
        connectivity = "Ring Connectivity"

    elif model_name == "VQC4":
        connectivity = "All-to-All Connectivity"

    else:
        connectivity = "Unknown"

    print(f"Connectivity Pattern : {connectivity}")

    # -----------------------------------------------------
    # RESOURCE SUMMARY
    # -----------------------------------------------------

    print("\nRESOURCE SUMMARY")
    print("-"*40)

    print(f"Qubits      : {n_qubits}")
    print(f"Depth       : {depth}")
    print(f"Parameters  : {n_params}")
    print(f"Total Gates : {gate_count}")

    # -----------------------------------------------------
    # SAVE SUMMARY TXT
    # -----------------------------------------------------

    txt_path = os.path.join(
        CCT_INFO_DIR,
        f"{model_name}_summary.txt"
    )

    with open(txt_path, "w") as f:

        f.write(f"{model_name} ANALYSIS\n")
        f.write("="*50 + "\n\n")

        f.write(f"Qubits              : {n_qubits}\n")
        f.write(f"Trainable Params    : {n_params}\n")
        f.write(f"Circuit Depth       : {depth}\n")
        f.write(f"Total Gates         : {gate_count}\n")
        f.write(f"Measurements        : {len(tape.measurements)}\n")
        f.write(f"Entangling Gates    : {len(entangling_gates)}\n")
        f.write(f"Connectivity        : {connectivity}\n\n")

        f.write("Gate Breakdown:\n")

        for gate, count in gate_dict.items():
            f.write(f"{gate:<10} : {count}\n")

    print(f"\n✓ Summary saved -> {txt_path}")

# =========================================================
# RUN ANALYSIS
# =========================================================

analyze_circuit(VQC1, "VQC1")
analyze_circuit(VQC2, "VQC2")
analyze_circuit(VQC3, "VQC3")
analyze_circuit(VQC4, "VQC4")