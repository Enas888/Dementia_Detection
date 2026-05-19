import os
import numpy as np
import pennylane as qml
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from QC_models.quantum_model1 import VQC as VQC1



# =========================================================
# PATHS
# =========================================================

BASE_DIR = r"C:\Users\emade\Downloads\Dementia_Detection"

SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")


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
# INIT MODEL
# =========================================================

model = VQC1(n_qubits=10)


# =========================================================
# LOSS FUNCTION
# =========================================================

def soft_loss(X, y, model):
    preds = []

    for x in X:
        out = model.forward(x)
        preds.append(np.mean(out))

    preds = np.array(preds)

    # simple regression-style loss
    return np.mean((preds - y) ** 2)


# =========================================================
# OPTIMIZER
# =========================================================

opt = qml.AdamOptimizer(stepsize=0.05)

epochs = 20


# =========================================================
# TRAINING LOOP
# =========================================================

for epoch in range(epochs):

    model.theta, loss = opt.step_and_cost(
        lambda t: soft_loss(X_train, y_train, model),
        model.theta
    )

    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict(model, X):

    preds = []

    for x in X:
        out = model.forward(x)
        preds.append(np.mean(out))

    return np.array(preds)


# =========================================================
# THRESHOLDING (MULTI-CLASS SIMPLE VERSION)
# =========================================================

def classify(preds):

    # map continuous output → class labels
    return np.digitize(preds, bins=[-0.3, 0.3])


# =========================================================
# EVALUATION
# =========================================================

train_pred = classify(predict(model, X_train))
test_pred = classify(predict(model, X_test))

print("\n=== RESULTS ===")

print("Train Accuracy:", accuracy_score(y_train, train_pred))
print("Test Accuracy:", accuracy_score(y_test, test_pred))

print("Test F1:", f1_score(y_test, test_pred, average='weighted'))

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))