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
    roc_auc_score,
    classification_report
)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

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
from QC_models.quantum_model5 import VQC as VQC5  

# tunned VQC2 with 3 layers and CRX entanglement — best so far

# =========================================================
# PATHS
# =========================================================

BASE_DIR = r"C:\Users\emade\Downloads\Dementia_Detection"

SPLIT_DIR  = os.path.join(BASE_DIR, "data", "splits")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CM_DIR     = os.path.join(RESULTS_DIR, "Quantum_confusion_matrix")
CURVE_DIR  = os.path.join(RESULTS_DIR, "Quantum_training_curves")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CM_DIR,      exist_ok=True)
os.makedirs(CURVE_DIR,   exist_ok=True)

RESULTS_JSON = os.path.join(RESULTS_DIR, "quantum_results.json")

# =========================================================
# LOAD DATA
# =========================================================

X_train = np.load(os.path.join(SPLIT_DIR, "X_train.npy"))
y_train = np.load(os.path.join(SPLIT_DIR, "y_train.npy"))

X_val   = np.load(os.path.join(SPLIT_DIR, "X_val.npy"))
y_val   = np.load(os.path.join(SPLIT_DIR, "y_val.npy"))

X_test  = np.load(os.path.join(SPLIT_DIR, "X_test.npy"))
y_test  = np.load(os.path.join(SPLIT_DIR, "y_test.npy"))

print("✓ Data loaded")

# =========================================================
# FEATURE NORMALIZATION
# =========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Clip to stable rotation range for angle embedding
X_train = np.clip(X_train, -np.pi, np.pi)
X_val   = np.clip(X_val,   -np.pi, np.pi)
X_test  = np.clip(X_test,  -np.pi, np.pi)

print("✓ Features normalized")

# =========================================================
# CLASS WEIGHTS
# =========================================================

classes      = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = pnp.array(class_weights)

print("✓ Class weights:", class_weights)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def softmax(x):
    exp_x = pnp.exp(x - pnp.max(x))
    return exp_x / pnp.sum(exp_x)


def forward_pass(theta, x, model):
    model.set_params(theta)
    logits = model.forward(x)
    return softmax(logits)


def cross_entropy_loss(theta, X, y, model):
    total_loss = 0.0
    for x_i, y_i in zip(X, y):
        probs  = forward_pass(theta, x_i, model)
        y_i    = int(y_i)
        one_hot = pnp.eye(3)[y_i]
        ce_loss = -pnp.sum(one_hot * pnp.log(probs + 1e-10))
        total_loss += ce_loss * class_weights[y_i]
    return total_loss / len(X)


def create_batches(X, y, batch_size=16):
    indices = np.random.permutation(len(X))
    for start in range(0, len(X), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


def predict_proba(model, X):
    return np.array([softmax(model.forward(x)) for x in X])


def classify(preds):
    return np.argmax(preds, axis=1)


# =========================================================
# TRAINING FUNCTION
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
    cv_epochs=30,
    batch_size=16,
    learning_rate=0.01
):

    print("\n" + "=" * 70)
    print(f" {model_name}")
    print("=" * 70)

    # =====================================================
    # 5-FOLD CROSS VALIDATION
    # =====================================================

    X_all = np.concatenate([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    print(f"\n── {model_name} | 5-Fold CV ──")

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_all, y_all)):

        print(f"\nFold {fold + 1}")

        Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
        Xva, yva = X_all[va_idx], y_all[va_idx]

        cv_model  = ModelClass(n_qubits=n_qubits)
        cv_opt    = qml.AdamOptimizer(stepsize=learning_rate)
        early_stop = EarlyStopping(patience=10)
        best_loss  = 1e9

        for epoch in range(cv_epochs):

            epoch_loss = 0.0
            n_batches  = 0

            for X_batch, y_batch in create_batches(Xtr, ytr, batch_size):
                theta, batch_loss = cv_opt.step_and_cost(
                    lambda t: cross_entropy_loss(t, X_batch, y_batch, cv_model),
                    cv_model.theta
                )
                cv_model.theta = theta
                epoch_loss    += batch_loss
                n_batches     += 1

            epoch_loss /= n_batches

            print(f"  Epoch {epoch + 1:03d} | Loss: {epoch_loss:.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss

            if early_stop.step(epoch_loss, cv_model.theta):
                print("  ✓ Early stopping (CV)")
                break

        cv_model.theta = early_stop.best_theta

        val_pred = classify(predict_proba(cv_model, Xva))
        val_acc  = accuracy_score(yva, val_pred)
        cv_scores.append(val_acc)

        print(f"  Validation Accuracy: {val_acc:.4f}")

    cv_mean = np.mean(cv_scores)
    cv_std  = np.std(cv_scores)
    print(f"\nCV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

    # =====================================================
    # FULL TRAINING
    # =====================================================

    print(f"\n── {model_name} | Full Training ──")

    model      = ModelClass(n_qubits=n_qubits)
    opt        = qml.AdamOptimizer(stepsize=learning_rate)
    early_stop = EarlyStopping(patience=15)
    train_losses = []
    best_loss    = 1e9

    t_start = time.time()

    for epoch in range(epochs):

        epoch_loss = 0.0
        n_batches  = 0

        for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
            theta, batch_loss = opt.step_and_cost(
                lambda t: cross_entropy_loss(t, X_batch, y_batch, model),
                model.theta
            )
            model.theta  = theta
            epoch_loss  += batch_loss
            n_batches   += 1

        epoch_loss /= n_batches
        train_losses.append(epoch_loss)

        print(f"Epoch {epoch + 1:03d}/{epochs} | Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        if early_stop.step(epoch_loss, model.theta):
            print("✓ Early stopping triggered")
            break

    model.theta  = early_stop.best_theta
    train_time   = time.time() - t_start

    # =====================================================
    # SAVE TRAINING CURVE
    # =====================================================

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Training Loss")

    curve_path = os.path.join(CURVE_DIR, f"{model_name}_loss_curve.png")
    plt.savefig(curve_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Training curve saved -> {curve_path}")

    # =====================================================
    # EVALUATION
    # =====================================================

    test_probs  = predict_proba(model, X_test)
    test_pred   = classify(test_probs)
    train_pred  = classify(predict_proba(model, X_train))

    train_acc = accuracy_score(y_train, train_pred)
    test_acc  = accuracy_score(y_test,  test_pred)
    test_f1   = f1_score(y_test, test_pred, average="weighted")
    cm        = confusion_matrix(y_test, test_pred)

    try:
        auc = roc_auc_score(
            y_test, test_probs,
            multi_class="ovr", average="weighted"
        )
    except Exception:
        auc = None

    # =====================================================
    # SAVE CONFUSION MATRIX
    # =====================================================

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Demented", "Nondemented", "Converted"],
        yticklabels=["Demented", "Nondemented", "Converted"]
    )
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_path = os.path.join(CM_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Confusion matrix saved -> {cm_path}")

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print("\n" + "=" * 70)
    print(f" {model_name} RESULTS")
    print("=" * 70)
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Test F1        : {test_f1:.4f}")
    print(f"AUC-ROC        : {auc:.4f}" if auc else "AUC-ROC        : N/A")
    print(f"CV Accuracy    : {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Training Time  : {train_time:.2f}s")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    # =====================================================
    # RETURN RESULTS
    # =====================================================

    return {
        "Accuracy":       round(float(test_acc),  4),
        "F1 (weighted)":  round(float(test_f1),   4),
        "AUC-ROC (OvR)":  round(float(auc), 4) if auc is not None else None,
        "CV Acc (mean)":  round(float(cv_mean),   4),
        "CV Acc (std)":   round(float(cv_std),    4),
        "Train Time (s)": round(float(train_time), 3),
        "Best Loss":      round(float(best_loss),  6),
    }


# =========================================================
# LOAD PREVIOUS RESULTS
# =========================================================

if os.path.exists(RESULTS_JSON):
    with open(RESULTS_JSON, "r") as f:
        quantum_results = json.load(f)
else:
    quantum_results = {}

# =========================================================
# RUN MODELS
# Only VQC5 is active — comment/uncomment others as needed
# =========================================================

# quantum_results["VQC1"] = run_experiment(
#     ModelClass=VQC1, model_name="VQC1",
#     X_train=X_train, y_train=y_train,
#     X_val=X_val,     y_val=y_val,
#     X_test=X_test,   y_test=y_test,
#     epochs=150, cv_epochs=50, batch_size=16, learning_rate=0.01
# )

# quantum_results["VQC2"] = run_experiment(
#     ModelClass=VQC2, model_name="VQC2",
#     X_train=X_train, y_train=y_train,
#     X_val=X_val,     y_val=y_val,
#     X_test=X_test,   y_test=y_test,
#     epochs=150, cv_epochs=50, batch_size=16, learning_rate=0.01
# )

# quantum_results["VQC3"] = run_experiment(
#     ModelClass=VQC3, model_name="VQC3",
#     X_train=X_train, y_train=y_train,
#     X_val=X_val,     y_val=y_val,
#     X_test=X_test,   y_test=y_test,
#     epochs=150, cv_epochs=50, batch_size=16, learning_rate=0.01
# )

# quantum_results["VQC4"] = run_experiment(
#     ModelClass=VQC4, model_name="VQC4",
#     X_train=X_train, y_train=y_train,
#     X_val=X_val,     y_val=y_val,
#     X_test=X_test,   y_test=y_test,
#     epochs=150, cv_epochs=50, batch_size=16, learning_rate=0.01
# )

quantum_results["VQC2"] = run_experiment(
    ModelClass=VQC2,
    model_name="VQC2",
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    X_test=X_test,   y_test=y_test,
    epochs=200,          # more epochs — 3 layers needs more steps
    cv_epochs=60,
    batch_size=16,
    learning_rate=0.005  # slightly lower lr — more stable for deeper circuit
)

# =========================================================
# SAVE RESULTS
# =========================================================

with open(RESULTS_JSON, "w") as f:
    json.dump(quantum_results, f, indent=4)

print(f"\n✓ Results saved -> {RESULTS_JSON}")