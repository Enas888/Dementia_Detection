import os
import joblib
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')


# =========================================================
# PATHS
# =========================================================

BASE_DIR = r"C:\Users\emade\Downloads\Dementia_Detection"

RAW_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "oasis_longitudinal_demographics.xlsx"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

VISUALS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "visuals"
)

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

demo = pd.read_excel(RAW_DATA_PATH)

print("✓ Raw data loaded")


# =========================================================
# COPY DATAFRAME
# =========================================================

df = demo.copy()


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

for col in ['SES', 'MMSE']:
    df[col] = df[col].fillna(df[col].mean())


# =========================================================
# ENCODE CATEGORICAL VARIABLES
# =========================================================

le = LabelEncoder()

df['Group_enc'] = le.fit_transform(df['Group'])

df['Sex_enc'] = (df['M/F'] == 'M').astype(int)


# =========================================================
# FEATURES & TARGET
# =========================================================

FEATURES = [
    'Age',
    'Sex_enc',
    'EDUC',
    'SES',
    'MMSE',
    'CDR',
    'eTIV',
    'nWBV',
    'ASF',
    'MR Delay'
]

X = df[FEATURES].values

y = df['Group_enc'].values


# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("✓ Features scaled")


# =========================================================
# VISUALIZATION SETTINGS
# =========================================================

PALETTE = {
    "Nondemented": "#2196F3",
    "Demented": "#F44336",
    "Converted": "#FF9800"
}

sns.set_theme(style="whitegrid", font_scale=1.0)


# =========================================================
# PREPROCESSING FIGURE
# =========================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle(
    "Preprocessing Insights",
    fontsize=16,
    fontweight='bold'
)


# =========================================================
# 1. CLASS BALANCE
# =========================================================

ax = axes[0, 0]

class_names = le.classes_

counts = [np.sum(y == i) for i in range(len(class_names))]

bars = ax.bar(
    class_names,
    counts,
    color=[PALETTE[c] for c in class_names],
    edgecolor='white',
    width=0.55
)

for bar, cnt in zip(bars, counts):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        str(cnt),
        ha='center',
        fontsize=11,
        fontweight='bold'
    )

ax.set_title("Class Balance")
ax.set_ylabel("Count")


# =========================================================
# 2. STANDARDIZED FEATURE DISTRIBUTIONS
# =========================================================

ax = axes[0, 1]

feature_labels = [
    'Age',
    'Sex',
    'EDUC',
    'SES',
    'MMSE',
    'CDR',
    'eTIV',
    'nWBV',
    'ASF',
    'MR Delay'
]

ax.boxplot(
    X_scaled,
    labels=feature_labels,
    patch_artist=True
)

ax.set_title("Feature Distributions (Scaled)")
ax.tick_params(axis='x', rotation=45)
ax.set_ylabel("Z-score")


# =========================================================
# 3. NULL VALUE HEATMAP
# =========================================================

ax = axes[0, 2]

null_map = demo[
    [
        'Age',
        'EDUC',
        'SES',
        'MMSE',
        'CDR',
        'eTIV',
        'nWBV',
        'ASF',
        'MR Delay'
    ]
].isnull().astype(int)

if null_map.any().any():

    sns.heatmap(
        null_map.T,
        cbar=False,
        cmap='Reds',
        ax=ax
    )

    ax.set_title("Missing Values")

else:

    ax.text(
        0.5,
        0.5,
        "No Missing Values",
        ha='center',
        va='center',
        fontsize=13,
        transform=ax.transAxes
    )

    ax.axis('off')

ax.set_xlabel("Session Index")


# =========================================================
# 4. DUPLICATE CHECK
# =========================================================

ax = axes[1, 0]

dups = demo.duplicated().sum()

ax.bar(
    ['Unique', 'Duplicate'],
    [len(demo) - dups, dups],
    color=['green', 'red']
)

ax.set_title(f"Duplicate Rows: {dups}")
ax.set_ylabel("Count")


# =========================================================
# 5. MMSE IMPUTATION
# =========================================================

ax = axes[1, 1]

missing_idx = demo['MMSE'].isnull()

ax.hist(
    demo['MMSE'].dropna(),
    bins=15,
    alpha=0.7,
    edgecolor='white'
)

ax.axvline(
    demo['MMSE'].mean(),
    color='red',
    linestyle='--',
    linewidth=2
)

ax.set_title("MMSE Mean Imputation")
ax.set_xlabel("MMSE")


# =========================================================
# 6. SES IMPUTATION
# =========================================================

ax = axes[1, 2]

ax.hist(
    demo['SES'].dropna(),
    bins=8,
    alpha=0.7,
    edgecolor='white'
)

ax.axvline(
    demo['SES'].mean(),
    color='red',
    linestyle='--',
    linewidth=2
)

ax.set_title("SES Mean Imputation")
ax.set_xlabel("SES")


# =========================================================
# SAVE FIGURE
# =========================================================

plt.tight_layout()

figure_path = os.path.join(
    VISUALS_DIR,
    "fig2_preprocessing.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("✓ Visualization saved")


# =========================================================
# SAVE PROCESSED DATA
# =========================================================

df.to_csv(
    os.path.join(PROCESSED_DIR, "processed_data.csv"),
    index=False
)

np.save(
    os.path.join(PROCESSED_DIR, "X_scaled.npy"),
    X_scaled
)

np.save(
    os.path.join(PROCESSED_DIR, "y.npy"),
    y
)

joblib.dump(
    scaler,
    os.path.join(PROCESSED_DIR, "scaler.pkl")
)

joblib.dump(
    le,
    os.path.join(PROCESSED_DIR, "label_encoder.pkl")
)

print("✓ Processed files saved successfully")