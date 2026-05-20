import os
import time
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns

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
from utils.early_stopping import EarlyStopping

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

from QC_models.quantum_model1 import VQC as VQC1
from QC_models.quantum_model2 import VQC as VQC2
from QC_models.quantum_model3 import VQC as VQC3
from QC_models.quantum_model4 import VQC as VQC4

# =========================================================
# PATHS
# =========================================================

BASE_DIR = r"C:\Users\emade\Downloads\Dementia_Detection"

SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")

RESULTS_DIR = os.path.join(BASE_DIR, "results")

out_path = os.path.join(
    RESULTS_DIR,
    "quantum_results.json"
)

CM_DIR = os.path.join(
    RESULTS_DIR,
    "Quantum_confusion_matrix"
)

os.makedirs(CM_DIR, exist_ok=True)
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

# =========================================================
# HELPERS
# =========================================================

def softmax(x):

    exp_x = pnp.exp(x - pnp.max(x))

    return exp_x / pnp.sum(exp_x)


def forward_pass(theta, x, model):

    model.set_params(theta)

    logits = model.forward(x)

    probs = softmax(logits)

    return probs


def cross_entropy_loss(theta, X, y, model):

    total_loss = 0.0

    for x_i, y_i in zip(X, y):

        probs = forward_pass(theta, x_i, model)

        one_hot = pnp.eye(3)[int(y_i)]

        loss = -pnp.sum(
            one_hot * pnp.log(probs + 1e-10)
        )

        total_loss += loss

    return total_loss / len(X)


def predict_proba(model, X):

    probs = []

    for x in X:

        logits = model.forward(x)

        probs.append(
            softmax(logits)
        )

    return pnp.stack(probs)


def classify(preds):

    return np.argmax(preds, axis=1)

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
    epochs=100,
    cv_epochs=30
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

        cv_model = ModelClass(n_qubits=n_qubits)
        cv_opt = qml.AdamOptimizer(stepsize=0.05)

        early_stop = EarlyStopping(patience=5)


        # ================================================
        # TRAIN CV MODEL
        # ================================================

        for _ in range(cv_epochs):
            theta, loss = cv_opt.step_and_cost(
                lambda t, Xtr=Xtr, ytr=ytr: cross_entropy_loss(t, Xtr, ytr, cv_model),
                cv_model.theta
            )

            cv_model.theta = theta


            if early_stop.step(loss, theta):
                print("  Early stopping triggered (CV)")
                break

        # restore best parameters
        cv_model.theta = early_stop.best_theta


        # ================================================
        # VALIDATION
        # ================================================

        val_raw = predict_proba(
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

        
    model = ModelClass(n_qubits=n_qubits)
    opt = qml.AdamOptimizer(stepsize=0.05)

    early_stop = EarlyStopping(patience=5)

    t_start = time.time()


    # ================================================
    # TRAIN FULL MODEL
    # ================================================

    for epoch in range(epochs):

        theta, loss = opt.step_and_cost(
            lambda t: cross_entropy_loss(t, X_train, y_train, model),
            model.theta
        )

        model.theta = theta

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

        if early_stop.step(loss, theta):
            print("Early stopping triggered (FULL TRAIN)")
            break

    # restore best weights
    model.theta = early_stop.best_theta

    train_time = time.time() - t_start


    # =====================================================
    # EVALUATION
    # =====================================================

    train_proba = predict_proba(
        model,
        X_train
    )

    test_proba = predict_proba(
        model,
        X_test
    )

    train_pred = classify(train_proba)

    test_pred = classify(test_proba)


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
    # SAVE CONFUSION MATRIX
    # =====================================================

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Demented", "Nondemented", "Converted"],
        yticklabels=["Demented", "Nondemented", "Converted"]
    )

    plt.title(f"{model_name} Confusion Matrix")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_path = os.path.join(
        CM_DIR,
        f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(
        cm_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"✓ Confusion matrix saved -> {cm_path}")

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

# quantum_results["VQC4"] = run_experiment(
#     ModelClass=VQC4,
#     model_name="VQC4",
#     X_train=X_train,
#     y_train=y_train,
#     X_val=X_val,
#     y_val=y_val,
#     X_test=X_test,
#     y_test=y_test
# )

# =========================================================
# SAVE RESULTS
# =========================================================

results_path = os.path.join(
    RESULTS_DIR,
    "quantum_results.json"
)

with open(out_path, "w") as f:

    json.dump(
        quantum_results,
        f,
        indent=4
    )

print(f"\n✓ Results saved -> {out_path}")