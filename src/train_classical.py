import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from openpyxl import Workbook
from openpyxl.styles import (
    Font,
    PatternFill,
    Alignment,
    Border,
    Side
)

warnings.filterwarnings('ignore')


# =========================================================
# PATHS
# =========================================================

BASE_DIR = r"C:\Users\emade\Downloads\Dementia_Detection"

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "splits"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

VISUALS_DIR = os.path.join(
    RESULTS_DIR,
    "visuals"
)

MODELS_DIR = os.path.join(
    RESULTS_DIR,
    "models"
)

METRICS_DIR = os.path.join(
    RESULTS_DIR,
    "metrics"
)

os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# =========================================================
# LOAD SPLITS
# =========================================================

X_train = np.load(os.path.join(SPLIT_DIR, "X_train.npy"))
y_train = np.load(os.path.join(SPLIT_DIR, "y_train.npy"))

X_val = np.load(os.path.join(SPLIT_DIR, "X_val.npy"))
y_val = np.load(os.path.join(SPLIT_DIR, "y_val.npy"))

X_test = np.load(os.path.join(SPLIT_DIR, "X_test.npy"))
y_test = np.load(os.path.join(SPLIT_DIR, "y_test.npy"))

print("✓ Data splits loaded")


# =========================================================
# SCALE DATA
# Fit ONLY on train
# =========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)

X_test = scaler.transform(X_test)

joblib.dump(
    scaler,
    os.path.join(MODELS_DIR, "final_scaler.pkl")
)

print("✓ Scaling complete")


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "Demented",
    "Nondemented",
    "Converted"
]


# =========================================================
# MODELS
# =========================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            eval_metric='mlogloss',
            random_state=42,
            verbosity=0
        ),

    "SVM":
        SVC(
            probability=True,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Naive Bayes":
        GaussianNB(),
}


# =========================================================
# TRAINING
# =========================================================

results = {}

conf_matrices = {}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for name, model in models.items():

    print(f"\nTraining: {name}")

    t0 = time.time()

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring='accuracy'
    )

    model.fit(X_train, y_train)

    train_time = time.time() - t0

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted'
    )

    auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class='ovr',
        average='weighted'
    )

    results[name] = {

        'Accuracy':
            round(acc, 4),

        'F1 (weighted)':
            round(f1, 4),

        'AUC-ROC (OvR)':
            round(auc, 4),

        'CV Acc (mean)':
            round(cv_scores.mean(), 4),

        'CV Acc (std)':
            round(cv_scores.std(), 4),

        'Train Time (s)':
            round(train_time, 3),
    }

    conf_matrices[name] = confusion_matrix(
        y_test,
        y_pred
    )

    # Save trained model
    model_path = os.path.join(
        MODELS_DIR,
        f"{name.replace(' ', '_')}.pkl"
    )

    joblib.dump(model, model_path)

    print(
        f"✓ {name} | "
        f"Acc={acc:.3f} | "
        f"F1={f1:.3f} | "
        f"AUC={auc:.3f}"
    )


# =========================================================
# RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(results).T.reset_index()

results_df.rename(
    columns={'index': 'Model'},
    inplace=True
)

print("\nResults:")
print(results_df)


# =========================================================
# FIGURE 3
# =========================================================

fig3 = plt.figure(figsize=(22, 18))

fig3.suptitle(
    "Classical Model Performance",
    fontsize=18,
    fontweight='bold'
)

gs3 = gridspec.GridSpec(
    3,
    3,
    figure=fig3,
    hspace=0.55,
    wspace=0.38
)

model_names = list(results.keys())

colors_bar = plt.cm.tab10(
    np.linspace(0, 1, len(model_names))
)


# =========================================================
# ACCURACY
# =========================================================

ax = fig3.add_subplot(gs3[0, 0])

acc_vals = [results[m]['Accuracy'] for m in model_names]

bars = ax.barh(
    model_names,
    acc_vals,
    color=colors_bar
)

ax.set_xlim(0, 1.05)

ax.set_title("Test Accuracy")


# =========================================================
# F1 SCORE
# =========================================================

ax = fig3.add_subplot(gs3[0, 1])

f1_vals = [
    results[m]['F1 (weighted)']
    for m in model_names
]

ax.barh(
    model_names,
    f1_vals,
    color=colors_bar
)

ax.set_xlim(0, 1.05)

ax.set_title("Weighted F1 Score")


# =========================================================
# AUC
# =========================================================

ax = fig3.add_subplot(gs3[0, 2])

auc_vals = [
    results[m]['AUC-ROC (OvR)']
    for m in model_names
]

ax.barh(
    model_names,
    auc_vals,
    color=colors_bar
)

ax.set_xlim(0, 1.05)

ax.set_title("AUC-ROC")


# =========================================================
# CV RESULTS
# =========================================================

ax = fig3.add_subplot(gs3[1, 0])

cv_means = [
    results[m]['CV Acc (mean)']
    for m in model_names
]

cv_stds = [
    results[m]['CV Acc (std)']
    for m in model_names
]

x_pos = np.arange(len(model_names))

ax.bar(
    x_pos,
    cv_means,
    yerr=cv_stds,
    capsize=5,
    color=colors_bar
)

ax.set_xticks(x_pos)

ax.set_xticklabels(
    [m.replace(' ', '\n') for m in model_names],
    fontsize=8
)

ax.set_ylim(0, 1.1)

ax.set_title("5-Fold CV Accuracy")


# =========================================================
# TRAINING TIME
# =========================================================

ax = fig3.add_subplot(gs3[1, 1])

times = [
    results[m]['Train Time (s)']
    for m in model_names
]

ax.bar(
    x_pos,
    times,
    color=colors_bar
)

ax.set_xticks(x_pos)

ax.set_xticklabels(
    [m.replace(' ', '\n') for m in model_names],
    fontsize=8
)

ax.set_title("Training Time")


# =========================================================
# TOP 3 MODELS
# =========================================================

top3 = results_df.nlargest(
    3,
    'Accuracy'
)['Model'].tolist()

metrics_compare = [
    'Accuracy',
    'F1 (weighted)',
    'AUC-ROC (OvR)',
    'CV Acc (mean)'
]

ax = fig3.add_subplot(gs3[1, 2])

x = np.arange(len(metrics_compare))

w = 0.25

for i, m in enumerate(top3):

    vals = [
        results[m][k]
        for k in metrics_compare
    ]

    ax.bar(
        x + i*w,
        vals,
        w,
        label=m
    )

ax.set_xticks(x + w)

ax.set_xticklabels(
    metrics_compare,
    rotation=15
)

ax.set_ylim(0, 1.15)

ax.legend(fontsize=8)

ax.set_title("Top 3 Models")


# =========================================================
# CONFUSION MATRICES
# =========================================================

for i, m_name in enumerate(top3):

    ax = fig3.add_subplot(gs3[2, i])

    sns.heatmap(
        conf_matrices[m_name],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax
    )

    ax.set_title(m_name)

    ax.set_xlabel("Predicted")

    ax.set_ylabel("Actual")


# =========================================================
# SAVE FIGURE 3
# =========================================================

fig3_path = os.path.join(
    VISUALS_DIR,
    "fig3_model_results.png"
)

plt.savefig(
    fig3_path,
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("✓ Figure 3 saved")


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

rf_model = models['Random Forest']

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

fi = pd.Series(
    rf_model.feature_importances_,
    index=feature_labels
).sort_values()


fig4, ax = plt.subplots(
    figsize=(10, 6)
)

ax.barh(
    fi.index,
    fi.values
)

ax.set_title(
    "Random Forest Feature Importance"
)

ax.set_xlabel("Importance")

fig4_path = os.path.join(
    VISUALS_DIR,
    "fig4_feature_importance.png"
)

plt.savefig(
    fig4_path,
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("✓ Figure 4 saved")


# =========================================================
# SAVE RESULTS TABLE
# =========================================================

excel_path = os.path.join(
    METRICS_DIR,
    "model_results.xlsx"
)

wb = Workbook()

ws = wb.active

ws.title = "Model Results"

headers = list(results_df.columns)

for col_i, h in enumerate(headers, 1):

    cell = ws.cell(
        row=1,
        column=col_i,
        value=h
    )

    cell.font = Font(
        bold=True,
        color='FFFFFF'
    )

    cell.fill = PatternFill(
        'solid',
        start_color='1565C0'
    )

    cell.alignment = Alignment(
        horizontal='center'
    )

for row in results_df.itertuples(index=False):

    ws.append(list(row))

wb.save(excel_path)

print("✓ Excel results saved")

print("\n=== ALL DONE ===")