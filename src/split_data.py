
'''
Because we have multiple MRI sessions per subject, the
split must be subject-wise, not row-wise. So all sessions
of one subject stay together
'''

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# =========================================================
# PATHS
# =========================================================

BASE_DIR = r"C:\Users\emade\Downloads\Dementia_Detection"

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "splits"
)

os.makedirs(SPLIT_DIR, exist_ok=True)


# =========================================================
# LOAD PROCESSED DATA
# =========================================================

df = pd.read_csv(
    os.path.join(PROCESSED_DIR, "processed_data.csv")
)

X_scaled = np.load(
    os.path.join(PROCESSED_DIR, "X_scaled.npy")
)

y = np.load(
    os.path.join(PROCESSED_DIR, "y.npy")
)

print("✓ Processed data loaded")


# =========================================================
# SUBJECT-LEVEL LABELS
# =========================================================

# One label per subject
subject_df = df.groupby('Subject ID')['Group_enc'].first().reset_index()

subjects = subject_df['Subject ID'].values
subject_labels = subject_df['Group_enc'].values


# =========================================================
# TRAIN + TEMP SPLIT
# =========================================================

train_subjects, temp_subjects, train_labels, temp_labels = train_test_split(
    subjects,
    subject_labels,
    test_size=0.30,
    stratify=subject_labels,
    random_state=42
)


# =========================================================
# VALIDATION + TEST SPLIT
# =========================================================

val_subjects, test_subjects, val_labels, test_labels = train_test_split(
    temp_subjects,
    temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=42
)

print(f"✓ Train subjects: {len(train_subjects)}")
print(f"✓ Validation subjects: {len(val_subjects)}")
print(f"✓ Test subjects: {len(test_subjects)}")


# =========================================================
# EXTRACT ROWS FOR EACH SPLIT
# =========================================================

train_df = df[df['Subject ID'].isin(train_subjects)]
val_df   = df[df['Subject ID'].isin(val_subjects)]
test_df  = df[df['Subject ID'].isin(test_subjects)]


# =========================================================
# GET FEATURES & LABELS
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


# ---------------- TRAIN ----------------

X_train = train_df[FEATURES].values
y_train = train_df['Group_enc'].values


# ---------------- VALIDATION ----------------

X_val = val_df[FEATURES].values
y_val = val_df['Group_enc'].values


# ---------------- TEST ----------------

X_test = test_df[FEATURES].values
y_test = test_df['Group_enc'].values


# =========================================================
# SAVE SPLITS
# =========================================================

np.save(os.path.join(SPLIT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(SPLIT_DIR, "y_train.npy"), y_train)

np.save(os.path.join(SPLIT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(SPLIT_DIR, "y_val.npy"), y_val)

np.save(os.path.join(SPLIT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(SPLIT_DIR, "y_test.npy"), y_test)


# =========================================================
# SAVE SUBJECT LISTS
# =========================================================

pd.DataFrame({
    'Subject ID': train_subjects
}).to_csv(
    os.path.join(SPLIT_DIR, "train_subjects.csv"),
    index=False
)

pd.DataFrame({
    'Subject ID': val_subjects
}).to_csv(
    os.path.join(SPLIT_DIR, "val_subjects.csv"),
    index=False
)

pd.DataFrame({
    'Subject ID': test_subjects
}).to_csv(
    os.path.join(SPLIT_DIR, "test_subjects.csv"),
    index=False
)


print("✓ Data splits saved successfully")