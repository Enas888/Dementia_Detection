# 🧠 Dementia Detection — Classical & Quantum ML Pipeline

A longitudinal brain MRI study for Alzheimer's disease classification using both **classical machine learning** and **Variational Quantum Circuits (VQCs)**, built on the [OASIS Longitudinal Dataset](https://www.oasis-brains.org/).

---

## 📂 Project Structure

```
Dementia_Detection/
│
├── data/
│   ├── raw/                        # Original OASIS CSV/Excel files
│   ├── processed/                  # Cleaned, encoded, scaled features
│   └── splits/
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       └── y_test.npy
│
├── QC_models/
│   ├── __init__.py
│   ├── quantum_model1.py           # VQC1 — Angle encoding + CNOT chain
│   ├── quantum_model2.py           # VQC2 — Angle encoding + CRX linear chain
│   ├── quantum_model3.py           # VQC3 — Angle encoding + CRX ring
│   ├── quantum_model4.py           # VQC4 — Angle encoding + CRX all-to-all
│   └── quantum_model5.py           # VQC5 — Tuned VQC4 (hyperparameter optimized)
│
├── results/
│   ├── metrics/
│   │   └── model_results.xlsx      # Classical model metrics table
│   ├── Quantum_confusion_matrix/   # Per-model confusion matrix PNGs
│   ├── visuals/                    # EDA and preprocessing figures
│   └── quantum_results.json        # Quantum model metrics (JSON)
│
├── src/
│   ├── utils/
│   │   └── early_stopping.py
│   ├── preprocessing.py
│   ├── split_data.py
│   ├── train_classical.py
│   └── train_quantum.py
│
├── notebooks/                      # Exploratory notebooks
├── requirements.txt
└── README.md
```

---

## 📊 Dataset — OASIS Longitudinal

| Property | Value |
|---|---|
| Subjects | 150 (aged 60–96) |
| Total MRI sessions | 373 |
| Visits per subject | ≥ 2 (separated by ≥ 1 year) |
| Scans per session | 3–4 T1-weighted MRI |
| Sex | Male & Female |
| Handedness | All right-handed |
| Nondemented | 72 subjects |
| Demented (from visit 1) | 64 subjects (51 mild–moderate AD) |
| Converted (healthy → dementia) | 14 subjects |

**Target classes:** `Nondemented` · `Demented` · `Converted`

**Features used:** Age, Sex, Education (EDUC), Socioeconomic Status (SES), MMSE, CDR, eTIV, nWBV, ASF, MR Delay

---

## ⚙️ Pipeline

### 1 · Preprocessing (`src/preprocessing.py`)
- Null imputation: SES and MMSE filled with column mean
- Label encoding: Group → {0, 1, 2}, Sex → binary
- StandardScaler normalization
- No duplicate rows found

### 2 · Data Split (`src/split_data.py`)
- Stratified split: Train / Val / Test
- Saved as `.npy` arrays under `data/splits/`

### 3 · Classical Models (`src/train_classical.py`)

8 classifiers evaluated with 5-fold cross-validation:

| Model | Accuracy | F1 (weighted) | AUC-ROC |
|---|---|---|---|
| Gradient Boosting | **0.933** | **0.920** | 0.943 |
| Logistic Regression | 0.920 | 0.898 | **0.961** |
| Random Forest | 0.920 | 0.898 | 0.948 |
| SVM | 0.920 | 0.898 | 0.955 |
| XGBoost | 0.893 | 0.877 | 0.949 |
| Naive Bayes | 0.893 | 0.877 | 0.940 |
| KNN | 0.867 | 0.841 | 0.934 |
| Decision Tree | 0.840 | 0.830 | 0.865 |

Metrics saved to `results/metrics/model_results.xlsx`.

### 4 · Quantum Models (`src/train_quantum.py`)

All models share:
- **Encoding:** Angle Embedding — each feature xᵢ mapped to RY(xᵢ) on qubit i (features scaled to [-π, π])
- **Variational layer:** RY + RZ rotations per qubit, then entanglement block
- **Data re-uploading:** Angle embedding is repeated before each variational layer for richer expressibility
- **Measurement:** PauliZ expectation on qubits 0, 1, 2 → 3 logits for 3-class softmax
- **Optimizer:** PennyLane AdamOptimizer (lr = 0.01)
- **Loss:** Cross-entropy
- **Early stopping:** patience = 10 epochs

| Model | Entanglement | CV Acc | Test Accuracy |
|---|---|---|---|
| VQC1 | CNOT linear chain | 0.4015 ± 0.074 | 0.2034 |
| VQC2 | CRX linear chain | 0.4711 ± 0.063 | 0.3390 |
| VQC3 | CRX ring | 0.4490 ± 0.040 | 0.5254 |
| VQC4 | CRX all-to-all (45 pairs) | 0.4808 ± 0.026 | — |
| **VQC5** | **CRX all-to-all (tuned)** | — | — |

Results saved to `results/quantum_results.json`.

---

## 🔬 Quantum Model Architectures

All models use **Angle Embedding** as the encoding strategy: each of the 10 input features is encoded as a RY rotation angle on its corresponding qubit. Features are pre-scaled to [-π, π] during preprocessing. In multi-layer circuits, the encoding is re-applied before each variational block (data re-uploading).

### VQC1 — CNOT Linear Chain
```
AngleEmbed(x) → RY(θ) RZ(θ) → CNOT(i→i+1) → ⟨Z⟩
```
Entanglement via plain CNOT gates — no trainable entangling parameters.

### VQC2 — CRX Linear Chain
```
AngleEmbed(x) → RY(θ) RZ(θ) → CRX(θ, i→i+1) → ⟨Z⟩
```
Replaces CNOT with trainable CRX, giving the entangling layer its own learnable angles along a linear topology.

### VQC3 — CRX Ring Connectivity
```
AngleEmbed(x) → RY(θ) RZ(θ) → CRX chain + CRX(9→0) → ⟨Z⟩
```
Closes the linear chain into a ring, adding one long-range connection between the last and first qubit.

### VQC4 — CRX All-to-All Connectivity
```
AngleEmbed(x) → RY(θ) RZ(θ) → CRX(i,j) ∀ i<j → ⟨Z⟩
```
Full connectivity: 45 trainable CRX pairs for 10 qubits. 3 layers with data re-uploading. Total parameters: 195.

### VQC5 — Tuned VQC4 *(in progress)*
```
AngleEmbed(x) → RY(θ) RZ(θ) → CRX(i,j) ∀ i<j → ⟨Z⟩  [optimized hyperparameters]
```
VQC5 shares the all-to-all CRX architecture of VQC4 — which achieved the highest cross-validation accuracy (0.4808 ± 0.026) among all quantum models — and extends it with systematic hyperparameter tuning (learning rate, number of layers, optimizer schedule, and batch strategy). VQC5 represents the best-effort quantum configuration in this pipeline.

---

## 🚀 Quickstart

```bash
# 1. Clone
git clone https://github.com/<your-username>/Dementia_Detection.git
cd Dementia_Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Preprocess & split
python -m src.preprocessing
python -m src.split_data

# 4. Train classical models
python -m src.train_classical

# 5. Train quantum models
python -m src.train_quantum
```

---

## 📦 Requirements

```
pennylane
pennylane-lightning
scikit-learn
numpy
pandas
matplotlib
seaborn
openpyxl
xgboost
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📁 Key Output Files

| File | Description |
|---|---|
| `results/metrics/model_results.xlsx` | Classical model comparison table |
| `results/quantum_results.json` | Quantum model metrics |
| `results/Quantum_confusion_matrix/*.png` | Confusion matrices per VQC |
| `results/visuals/` | EDA plots, preprocessing figures |

---

## 📌 Notes

- Quantum training is slow on CPU — VQC4/VQC5 (all-to-all) are the most expensive due to 45 CRX pairs per layer. Consider running with `pennylane-lightning` on GPU or reducing the number of layers.
- The `Converted` class (14 subjects) is severely underrepresented, which limits quantum model performance on that class.
- Classical models significantly outperform quantum models at this scale, consistent with NISQ-era expectations on tabular data.
- Angle Embedding is used across all VQC variants, with features scaled to [-π, π] to match the rotation angle range of RY gates.

---

## 📜 Citation

```
OASIS: Longitudinal: Principal Investigators: D. Marcus, R. Buckner,
J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276,
R01 AG021910, P20 MH071616, U24 RR021382
```
