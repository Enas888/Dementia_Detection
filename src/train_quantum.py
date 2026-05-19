import os
import time
import json
import sys

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from sklearn.model_selection import StratifiedKFold


# =========================================================
# FIX PROJECT IMPORTS
# =========================================================

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(ROOT_DIR)


# =========================================================
# IMPORT QUANTUM MODELS
# =========================================================

from results.QC_models.quantum_model1 import VQC as VQC1
from results.QC_models.quantum_model2 import VQC as VQC2


# =========================================================
# PATHS
# =========================================================

BASE_DIR = r"C:\Users\emade\Downloads\Dementia_Detection"

SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")

RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

X_train = np.load(os.path.join(SPLIT_DIR, "X_train.npy"))
y_train = np.load(os.path.join(SPLIT_DIR, "y_train.npy"))

X_val = np.load(os.path.join(SPLIT_DIR, "X_val.npy"))
y_val = np.load(os.path.join(SPLIT_DIR, "y_val.npy"))

X_test = np.load(os.path.join(SPLIT_DIR, "X_test.npy"))
y_test = np.load(os.path.join(SPLIT_DIR, "y_test.npy"))

print("✓ Data loaded")


# =========================================================
# HELPERS
# =========================================================

def forward_pass(theta, x, model):
    """
    Forward pass for ONE sample.
    Fully differentiable.
    """
    model.set_params(theta)

    output = model.forward(x)

    # convert list -> tensor safely
    output = pnp.stack(output)

    return pnp.mean(output)

def soft_loss(theta, X, y, model):
    """
    MSE loss.
    Fully differentiable.
    """

    preds = pnp.array([
        forward_pass(theta, x, model)
        for x in X
    ])

    return pnp.mean((preds - y) ** 2)


def predict_raw(model, X):
    """
    Raw continuous predictions.
    NO gradients here.
    """
    preds = []

    for x in X:

        output = model.forward(x)

        output = pnp.stack(output)

        preds.append(float(pnp.mean(output)))

    return np.array(preds)

def classify(preds):
    """
    Convert continuous scores -> 3 classes.
    """

    return np.digitize(
        preds,
        bins=[-0.3, 0.3]
    )


def raw_to_proba(raw_scores, n_classes):
    """
    Simple pseudo-probabilities for AUC.
    """

    bins = [-0.3, 0.3]

    proba = np.zeros(
        (len(raw_scores), n_classes)
    )

    for i, score in enumerate(raw_scores):

        cls = min(
            int(np.digitize(score, bins)),
            n_classes - 1
        )

        proba[i, cls] = 1.0

    return proba


# =========================================================
# TRAIN + EVALUATE
# =========================================================

def run_experiment(
    ModelClass,
    model_name,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    n_qubits=10,
    epochs=20,
    cv_epochs=10
):

    # =====================================================
    # MERGE TRAIN + VAL FOR CV
    # =====================================================

    X_all = np.concatenate([X_train, X_val])

    y_all = np.concatenate([y_train, y_val])

    n_classes = len(np.unique(y_all))


    # =====================================================
    # 5-FOLD CROSS VALIDATION
    # =====================================================

    print(f"\n── {model_name} | 5-Fold CV ──")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_all, y_all)):

        print(f"\nFold {fold+1}")

        Xtr = X_all[tr_idx]
        ytr = y_all[tr_idx]

        Xva = X_all[va_idx]
        yva = y_all[va_idx]

        cv_model = ModelClass(
            n_qubits=n_qubits
        )

        optimizer = qml.AdamOptimizer(
            stepsize=0.05
        )

        theta = cv_model.theta

        # ================================================
        # TRAIN CV MODEL
        # ================================================

        for epoch in range(cv_epochs):

            theta, loss = optimizer.step_and_cost(
                lambda t: soft_loss(
                    t,
                    Xtr,
                    ytr,
                    cv_model
                ),
                theta
            )

        cv_model.theta = theta

        # ================================================
        # VALIDATION
        # ================================================

        val_raw = predict_raw(
            cv_model,
            Xva
        )

        val_pred = classify(val_raw)

        fold_acc = accuracy_score(
            yva,
            val_pred
        )

        cv_scores.append(fold_acc)

        print(f"  Accuracy: {fold_acc:.4f}")

    cv_mean = np.mean(cv_scores)

    cv_std = np.std(cv_scores)

    print(f"\nCV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")


    # =====================================================
    # FULL TRAINING
    # =====================================================

    print(f"\n── {model_name} | Full Training ──")

    model = ModelClass(
        n_qubits=n_qubits
    )

    optimizer = qml.AdamOptimizer(
        stepsize=0.05
    )

    theta = model.theta

    start_time = time.time()

    for epoch in range(epochs):

        theta, loss = optimizer.step_and_cost(
            lambda t: soft_loss(
                t,
                X_train,
                y_train,
                model
            ),
            theta
        )

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"| Loss: {loss:.4f}"
        )

    train_time = time.time() - start_time

    model.theta = theta


    # =====================================================
    # EVALUATION
    # =====================================================

    train_raw = predict_raw(
        model,
        X_train
    )

    test_raw = predict_raw(
        model,
        X_test
    )

    train_pred = classify(train_raw)

    test_pred = classify(test_raw)

    test_proba = raw_to_proba(
        test_raw,
        n_classes
    )

    train_acc = accuracy_score(
        y_train,
        train_pred
    )

    test_acc = accuracy_score(
        y_test,
        test_pred
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        average="weighted"
    )

    cm = confusion_matrix(
        y_test,
        test_pred
    )

    # =====================================================
    # AUC
    # =====================================================

    try:

        auc = roc_auc_score(
            y_test,
            test_proba,
            multi_class="ovr",
            average="weighted"
        )

    except Exception:

        auc = None


    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print(f"\n=== {model_name} RESULTS ===")

    print(f"Train Accuracy : {train_acc:.4f}")

    print(f"Test Accuracy  : {test_acc:.4f}")

    print(f"Test F1        : {test_f1:.4f}")

    if auc is not None:
        print(f"AUC-ROC        : {auc:.4f}")
    else:
        print("AUC-ROC        : N/A")

    print(f"CV Accuracy    : {cv_mean:.4f} ± {cv_std:.4f}")

    print(f"Training Time  : {train_time:.2f}s")

    print("\nConfusion Matrix:")

    print(cm)


    # =====================================================
    # RETURN RESULTS
    # =====================================================

    return {
        "Accuracy": round(float(test_acc), 4),

        "F1 (weighted)": round(float(test_f1), 4),

        "AUC-ROC (OvR)": (
            round(float(auc), 4)
            if auc is not None else None
        ),

        "CV Acc (mean)": round(float(cv_mean), 4),

        "CV Acc (std)": round(float(cv_std), 4),

        "Train Time (s)": round(float(train_time), 3)
    }


# =========================================================
# RUN EXPERIMENTS
# =========================================================

# Load existing results if present
if os.path.exists(out_path):

    with open(out_path, "r") as f:
        quantum_results = json.load(f)

else:
    quantum_results = {}
    
# =========================================================
# VQC1
# =========================================================

# quantum_results["VQC1"] = run_experiment(
#     ModelClass=VQC1,
#     model_name="VQC1",
#     X_train=X_train,
#     y_train=y_train,
#     X_val=X_val,
#     y_val=y_val,
#     X_test=X_test,
#     y_test=y_test
# )

# =========================================================
# VQC2
# =========================================================

# quantum_results["VQC2"] = run_experiment(
#     ModelClass=VQC2,
#     model_name="VQC2",
#     X_train=X_train,
#     y_train=y_train,
#     X_val=X_val,
#     y_val=y_val,
#     X_test=X_test,
#     y_test=y_test
# )

quantum_results["VQC3"] = run_experiment(
    ModelClass=VQC3,
    model_name="VQC3",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test
)

# =========================================================
# SAVE RESULTS
# =========================================================

results_path = os.path.join(
    RESULTS_DIR,
    "quantum_results.json"
)

with open(results_path, "w") as f:

    json.dump(
        quantum_results,
        f,
        indent=4
    )

print(f"\n✓ Results saved -> {results_path}")