import os
import time
import json
import sys

import matplotlib
matplotlib.use("Agg")
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
from QC_models.quantum_model5 import VQC as VQC4

# =========================================================
# PATHS
# =========================================================

BASE_DIR = r"C:\Users\emade\Downloads\Dementia_Detection"

SPLIT_DIR  = os.path.join(BASE_DIR, "data", "splits")
RESULTS_DIR = os.path.join(BASE_DIR, "results_5")
CM_DIR     = os.path.join(RESULTS_DIR, "Quantum_confusion_matrix_5")
CURVE_DIR  = os.path.join(RESULTS_DIR, "Quantum_training_curves_5")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)
os.makedirs(CURVE_DIR, exist_ok=True)

RESULTS_JSON = os.path.join(RESULTS_DIR, "quantum_results_5.json")

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
print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# =========================================================
# FEATURE NORMALIZATION
#
# Why two steps:
#   1. StandardScaler: zero-mean, unit variance
#   2. Clip to [-pi, pi]: stable rotation angles for RY/RZ gates
#      (quantum gates are periodic — values far outside this range
#       wrap around and cause gradient vanishing)
# =========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)   # fit on train only
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

X_train = np.clip(X_train, -np.pi, np.pi)
X_val   = np.clip(X_val,   -np.pi, np.pi)
X_test  = np.clip(X_test,  -np.pi, np.pi)

print("✓ Features normalized and clipped to [-π, π]")

# =========================================================
# CLASS WEIGHTS
#
# Dataset has severe imbalance: Converted class = only 14 subjects.
# Balanced weights penalize errors on minority classes more heavily.
# =========================================================

classes = np.unique(y_train)

class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = pnp.array(class_weights_arr)

print("✓ Class weights:", dict(zip(classes, class_weights_arr.round(3))))

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def softmax(x):
    """Numerically stable softmax."""
    exp_x = pnp.exp(x - pnp.max(x))
    return exp_x / pnp.sum(exp_x)


def forward_pass(theta, x, model):
    """
    Run circuit for one sample, return class probabilities.
    Sets model parameters before calling forward so PennyLane
    can differentiate through theta correctly.
    """
    model.set_params(theta)
    logits = model.forward(x)
    return softmax(logits)


# =========================================================
# WEIGHTED CROSS-ENTROPY LOSS
#
# L = - (1/N) * sum_i [ w_{y_i} * sum_k [ y_hot_k * log(p_k) ] ]
#
# This is the correct multi-class loss for imbalanced data.
# Using MSE on integer labels (old approach) confuses the model
# because the class labels 0/1/2 have an implied ordering.
# =========================================================

def cross_entropy_loss(theta, X, y, model):

    total_loss = pnp.array(0.0)

    for x_i, y_i in zip(X, y):

        probs   = forward_pass(theta, x_i, model)
        y_i     = int(y_i)
        one_hot = pnp.eye(3)[y_i]

        # Cross-entropy for this sample
        ce = -pnp.sum(one_hot * pnp.log(probs + 1e-10))

        # Weight by class importance
        total_loss = total_loss + ce * class_weights[y_i]

    return total_loss / len(X)


# =========================================================
# LABEL-SMOOTHED CROSS-ENTROPY (optional, helps generalisation)
#
# Instead of hard one-hot [0,0,1], use soft targets [ε/K, ε/K, 1-ε*(K-1)/K]
# This prevents the model from being overconfident and helps on
# tiny datasets like ours (especially the Converted class).
# =========================================================

LABEL_SMOOTHING = 0.1   # set to 0.0 to disable

def smooth_one_hot(y_i, n_classes=3, eps=LABEL_SMOOTHING):
    one_hot = pnp.eye(n_classes)[y_i]
    return one_hot * (1 - eps) + eps / n_classes


def cross_entropy_loss_smooth(theta, X, y, model):

    total_loss = pnp.array(0.0)

    for x_i, y_i in zip(X, y):

        probs   = forward_pass(theta, x_i, model)
        y_i_int = int(y_i)
        target  = smooth_one_hot(y_i_int)

        ce = -pnp.sum(target * pnp.log(probs + 1e-10))
        total_loss = total_loss + ce * class_weights[y_i_int]

    return total_loss / len(X)


# =========================================================
# MINI-BATCH SAMPLING
#
# Batch size 16 is a sweet spot: large enough for stable gradients,
# small enough to see the full training set many times per epoch.
# =========================================================

def create_batches(X, y, batch_size=16):
    indices = np.random.permutation(len(X))
    for start in range(0, len(X), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


# =========================================================
# PREDICTION HELPERS
# =========================================================

def predict_proba(model, X):
    """Softmax probabilities for all samples. Shape: (N, 3)."""
    probs = []
    for x in X:
        logits = model.forward(x)
        probs.append(softmax(logits))
    return np.array(probs)


def classify(probs):
    """Argmax class predictions."""
    return np.argmax(probs, axis=1)


# =========================================================
# LEARNING RATE SCHEDULER
#
# Simple step decay: halve the LR every `step_size` epochs.
# Helps escape plateaus without oscillating late in training.
# =========================================================

def get_lr(initial_lr, epoch, step_size=30, decay=0.5):
    return initial_lr * (decay ** (epoch // step_size))


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
    cv_epochs=40,
    batch_size=16,
    learning_rate=0.01,
    use_label_smoothing=True,
    lr_decay=True,
):
    """
    Full train/eval pipeline for one VQC model:
        1. 5-fold cross-validation (for honest CV score)
        2. Full training on train+val combined
        3. Evaluation on held-out test set
        4. Save confusion matrix + training curve

    Parameters
    ----------
    ModelClass         : VQC class to instantiate
    model_name         : string identifier ("VQC4" etc.)
    epochs             : max full-training epochs
    cv_epochs          : max epochs inside each CV fold
    batch_size         : mini-batch size
    learning_rate      : initial Adam learning rate
    use_label_smoothing: whether to use smoothed CE loss
    lr_decay           : whether to apply step-decay LR schedule
    """

    loss_fn = (
        cross_entropy_loss_smooth
        if use_label_smoothing
        else cross_entropy_loss
    )

    print("\n" + "=" * 70)
    print(f"  {model_name}")
    print("=" * 70)

    # =======================================================
    # 5-FOLD CROSS VALIDATION
    # =======================================================

    X_all = np.concatenate([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    print(f"\n── {model_name} | 5-Fold CV ──")

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_all, y_all)):

        print(f"\n  Fold {fold + 1}/5")

        Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
        Xva, yva = X_all[va_idx], y_all[va_idx]

        cv_model = ModelClass(n_qubits=n_qubits)
        cv_opt   = qml.AdamOptimizer(stepsize=learning_rate)
        early    = EarlyStopping(patience=5)

        for epoch in range(cv_epochs):

            # Optional LR decay
            if lr_decay:
                cv_opt.stepsize = get_lr(learning_rate, epoch)

            epoch_loss = 0.0
            n_batches  = 0

            for X_b, y_b in create_batches(Xtr, ytr, batch_size):

                theta, batch_loss = cv_opt.step_and_cost(
                    lambda t: loss_fn(t, X_b, y_b, cv_model),
                    cv_model.theta
                )
                cv_model.theta = theta
                epoch_loss    += float(batch_loss)
                n_batches     += 1

            epoch_loss /= n_batches

            print(
                f"    Epoch {epoch + 1:03d}/{cv_epochs} | "
                f"Loss: {epoch_loss:.4f} | "
                f"LR: {cv_opt.stepsize:.5f}"
            )

            if early.step(epoch_loss, cv_model.theta):
                print("    ✓ Early stopping (CV)")
                break

        # Restore best weights
        cv_model.theta = early.best_theta

        val_pred = classify(predict_proba(cv_model, Xva))
        val_acc  = accuracy_score(yva, val_pred)
        cv_scores.append(val_acc)
        print(f"  Fold {fold + 1} Val Acc: {val_acc:.4f}")

    cv_mean = np.mean(cv_scores)
    cv_std  = np.std(cv_scores)
    print(f"\n  CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

    # =======================================================
    # FULL TRAINING
    # =======================================================

    print(f"\n── {model_name} | Full Training ──")

    model      = ModelClass(n_qubits=n_qubits)
    opt        = qml.AdamOptimizer(stepsize=learning_rate)
    early      = EarlyStopping(patience=10)
    train_losses  = []
    val_losses    = []
    best_val_acc  = 0.0
    best_theta    = model.theta

    t_start = time.time()

    for epoch in range(epochs):

        if lr_decay:
            opt.stepsize = get_lr(learning_rate, epoch)

        epoch_loss = 0.0
        n_batches  = 0

        for X_b, y_b in create_batches(X_train, y_train, batch_size):

            theta, batch_loss = opt.step_and_cost(
                lambda t: loss_fn(t, X_b, y_b, model),
                model.theta
            )
            model.theta = theta
            epoch_loss += float(batch_loss)
            n_batches  += 1

        epoch_loss /= n_batches
        train_losses.append(epoch_loss)

        # Validation accuracy for monitoring
        val_preds   = classify(predict_proba(model, X_val))
        val_acc_ep  = accuracy_score(y_val, val_preds)

        # Validation loss for early stopping
        val_loss = float(
            loss_fn(model.theta, X_val, y_val, model)
        )
        val_losses.append(val_loss)

        # Track best by val accuracy
        if val_acc_ep > best_val_acc:
            best_val_acc = val_acc_ep
            best_theta   = model.theta

        print(
            f"  Epoch {epoch + 1:03d}/{epochs} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc_ep:.4f} | "
            f"LR: {opt.stepsize:.5f}"
        )

        if early.step(val_loss, model.theta):
            print("  ✓ Early stopping triggered")
            break

    # Restore best checkpoint
    model.theta = best_theta
    train_time  = time.time() - t_start

    # =======================================================
    # SAVE TRAINING CURVES
    # =======================================================

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="Train", color="#0891B2")
    axes[0].plot(val_losses,   label="Val",   color="#B91C1C", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} — Loss Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        [accuracy_score(
            classify(predict_proba(model, X_val)),
            y_val
        )] * len(train_losses),
        color="#6D28D9",
        linestyle=":",
        label="Best Val Acc"
    )
    axes[1].set_title(f"{model_name} — Val Accuracy Ref")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    curve_path = os.path.join(CURVE_DIR, f"{model_name}_loss_curve.png")
    plt.savefig(curve_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Training curve saved → {curve_path}")

    # =======================================================
    # EVALUATION
    # =======================================================

    train_probs = predict_proba(model, X_train)
    test_probs  = predict_proba(model, X_test)

    train_pred  = classify(train_probs)
    test_pred   = classify(test_probs)

    train_acc   = accuracy_score(y_train, train_pred)
    test_acc    = accuracy_score(y_test,  test_pred)
    test_f1     = f1_score(y_test, test_pred, average="weighted")

    cm = confusion_matrix(y_test, test_pred)

    try:
        auc = roc_auc_score(
            y_test,
            test_probs,
            multi_class="ovr",
            average="weighted"
        )
    except Exception:
        auc = None

    # =======================================================
    # SAVE CONFUSION MATRIX
    # =======================================================

    class_labels = ["Demented", "Nondemented", "Converted"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    plt.title(f"{model_name} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    cm_path = os.path.join(CM_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Confusion matrix saved → {cm_path}")

    # =======================================================
    # PRINT SUMMARY
    # =======================================================

    print("\n" + "=" * 70)
    print(f"  {model_name} — FINAL RESULTS")
    print("=" * 70)
    print(f"  Train Accuracy : {train_acc:.4f}")
    print(f"  Test Accuracy  : {test_acc:.4f}")
    print(f"  Test F1        : {test_f1:.4f}")
    print(f"  AUC-ROC        : {auc:.4f}" if auc else "  AUC-ROC        : N/A")
    print(f"  CV Accuracy    : {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  Training Time  : {train_time:.1f}s")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{classification_report(y_test, test_pred, target_names=class_labels)}")

    return {
        "Accuracy":       round(float(test_acc),  4),
        "F1 (weighted)":  round(float(test_f1),   4),
        "AUC-ROC (OvR)":  round(float(auc), 4) if auc else None,
        "CV Acc (mean)":  round(float(cv_mean),   4),
        "CV Acc (std)":   round(float(cv_std),    4),
        "Train Time (s)": round(float(train_time),3),
    }


# =========================================================
# LOAD PREVIOUS RESULTS (so you can run models independently)
# =========================================================

if os.path.exists(RESULTS_JSON):
    with open(RESULTS_JSON, "r") as f:
        quantum_results = json.load(f)
else:
    quantum_results = {}

# =========================================================
# RUN MODELS
#
# Uncomment each block as needed. They can run independently —
# results are merged into the same JSON file each time.
# VQC4 is the slowest (45 CRX gates × N_LAYERS × epochs).
# =========================================================

# --- VQC1: CNOT linear chain ---
# quantum_results["VQC1"] = run_experiment(
#     ModelClass=VQC1,
#     model_name="VQC1",
#     X_train=X_train, y_train=y_train,
#     X_val=X_val,     y_val=y_val,
#     X_test=X_test,   y_test=y_test,
#     epochs=100, cv_epochs=40,
#     batch_size=16, learning_rate=0.01,
# )

# --- VQC2: CRX linear chain (previously best) ---
# quantum_results["VQC2"] = run_experiment(
#     ModelClass=VQC2,
#     model_name="VQC2",
#     X_train=X_train, y_train=y_train,
#     X_val=X_val,     y_val=y_val,
#     X_test=X_test,   y_test=y_test,
#     epochs=100, cv_epochs=40,
#     batch_size=16, learning_rate=0.01,
# )

# --- VQC3: CRX ring ---
# quantum_results["VQC3"] = run_experiment(
#     ModelClass=VQC3,
#     model_name="VQC3",
#     X_train=X_train, y_train=y_train,
#     X_val=X_val,     y_val=y_val,
#     X_test=X_test,   y_test=y_test,
#     epochs=100, cv_epochs=40,
#     batch_size=16, learning_rate=0.01,
# )

# --- VQC4: CRX all-to-all (most expressive, most expensive) ---
quantum_results["VQC4"] = run_experiment(
    ModelClass=VQC4,
    model_name="VQC4",
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    X_test=X_test,   y_test=y_test,
    epochs=100,
    cv_epochs=40,
    batch_size=16,
    learning_rate=0.01,
    use_label_smoothing=True,  # helps with Converted class imbalance
    lr_decay=True,             # halves LR every 30 epochs
)

# =========================================================
# SAVE RESULTS
# =========================================================

with open(RESULTS_JSON, "w") as f:
    json.dump(quantum_results, f, indent=4)

print(f"\n✓ Results saved → {RESULTS_JSON}")
print("\nSummary:")
for model_name, metrics in quantum_results.items():
    print(f"  {model_name}: Acc={metrics['Accuracy']}, F1={metrics['F1 (weighted)']}")