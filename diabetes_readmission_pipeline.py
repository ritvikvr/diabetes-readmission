
import os
from pathlib import Path
import zipfile
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             precision_recall_curve, auc)
from sklearn.calibration import CalibrationDisplay
from sklearn.preprocessing import OneHotEncoder


try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    _has_kaggle = True
except Exception:
    _has_kaggle = False

# -----------------------
# Config / paths
# -----------------------
DATA_DIR = Path("data/diabetes_ucireadmit")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_FILE = DATA_DIR / "diabetic_data.csv"
ZIP_FILE = DATA_DIR / "diabetic_data.zip"

UCI_RAW_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetic_data.zip"


KAGGLE_SLUGS = [
    "bhanuc/diabetes-130-us-hospitals-for-years-1999-2008",   # some copies
    "abdelazizsami/diabetes-130-us-hospitals-for-years-1999-2008",
    "brandao/diabetes"
]

def try_kaggle_download(slugs, out_dir):
    if not _has_kaggle:
        return False
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print("Kaggle API present but authentication failed:", e)
        return False
    for s in slugs:
        try:
            print(f"Trying Kaggle dataset: {s}")
            api.dataset_download_files(s, path=str(out_dir), unzip=True)
            found = list(out_dir.glob("**/*.csv"))
            if found:
                found0 = found[0]
                (out_dir / "diabetic_data.csv").write_bytes(found0.read_bytes())
                print("Downloaded via Kaggle:", found0.name)
                return True
        except Exception as e:
            print("Kaggle download failed for", s, e)
    return False

def try_uci_download(url, out_zip):
    print("Attempting UCI download...")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        out_zip.write_bytes(r.content)
        with zipfile.ZipFile(out_zip, "r") as z:
            for name in z.namelist():
                if name.endswith(".csv"):
                    z.extract(name, out_zip.parent)
                    extracted = out_zip.parent / name
                    extracted.rename(out_zip.parent / "diabetic_data.csv")
                    print("Extracted:", name)
                    return True
        return False
    except Exception as e:
        print("UCI download failed:", e)
        return False

if not CSV_FILE.exists():
    ok = try_kaggle_download(KAGGLE_SLUGS, DATA_DIR)
    if not ok:
        ok = try_uci_download(UCI_RAW_URL, ZIP_FILE)
    if not ok:
        raise SystemExit(f"Place 'diabetic_data.csv' inside {DATA_DIR} and re-run.")

print("Using CSV:", CSV_FILE)

# -----------------------
# Load CSV
# -----------------------
df = pd.read_csv(CSV_FILE)
print("Rows, cols:", df.shape)
print("Columns sample:", df.columns.tolist())

# -----------------------
# Target engineering
# -----------------------
# 'readmitted' values: '<30', '>30', 'NO'
if 'readmitted' not in df.columns:
    raise SystemExit("Expected column 'readmitted' not found.")

df['readmit_30'] = (df['readmitted'] == '<30').astype(int)
print("Target distribution (counts):\n", df['readmit_30'].value_counts())

# -----------------------
# Basic cleaning
# -----------------------
# Drop identity columns if present
for c in ['encounter_id','patient_nbr']:
    if c in df.columns:
        df = df.drop(columns=[c])

# Standard sentinel replacement
df = df.replace('?', np.nan)

# Convert common numeric columns to numeric
numeric_maybe = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                 'num_medications', 'number_outpatient', 'number_emergency',
                 'number_inpatient', 'number_diagnoses']
for c in numeric_maybe:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# diag columns as string
for c in ['diag_1','diag_2','diag_3']:
    if c in df.columns:
        df[c] = df[c].astype(object).fillna('UNK')

# Fill object NaNs with 'UNK' except original readmitted
for c in df.select_dtypes(include=['object']).columns:
    if c != 'readmitted':
        df[c] = df[c].fillna('UNK')

# -----------------------
# Feature selection (robust, no duplicates)
# -----------------------

candidate_numeric = [c for c in [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
] if c in df.columns]


candidate_cats = [c for c in [
    'race','gender','age','admission_type_id','discharge_disposition_id','admission_source_id',
    'payer_code','medical_specialty','diag_1','diag_2','diag_3','max_glu_serum','A1Cresult',
    'metformin','insulin','change','diabetesMed'
] if c in df.columns]


numeric_features = [c for c in candidate_numeric if c in df.columns]
categorical_features = [c for c in candidate_cats if c in df.columns and c not in numeric_features]

# Final features
features = numeric_features + categorical_features
print(f"Selected features (total={len(features)}). Numeric={len(numeric_features)} Cat={len(categorical_features)}")
print("Numeric:", numeric_features)
print("Categorical:", categorical_features)

# Rebuild X and y
X = df[features].copy()
y = df['readmit_30'].copy()

# -----------------------
# Preprocessing pipelines (compatibility-safe)
# -----------------------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Use sparse_output=False for compatibility with sklearn in your environment
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='UNK')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
], remainder='drop')

# -----------------------
# Train/test split (stratified)
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)
print("Train/test shapes:", X_train.shape, X_test.shape)
print("Train class balance:", y_train.value_counts(normalize=True).to_dict())

# -----------------------
# Pipelines
# -----------------------
lr_pipe = Pipeline(steps=[
    ('pre', preprocessor),
    ('clf', LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000, random_state=42))
])

rf_pipe = Pipeline(steps=[
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1))
])

# -----------------------
# Fit models
# -----------------------
print("Fitting LogisticRegression...")
lr_pipe.fit(X_train, y_train)
probs_lr = lr_pipe.predict_proba(X_test)[:, 1]
pred_lr = (probs_lr >= 0.5).astype(int)

print("Fitting RandomForest...")
rf_pipe.fit(X_train, y_train)
probs_rf = rf_pipe.predict_proba(X_test)[:, 1]
pred_rf = (probs_rf >= 0.5).astype(int)

# -----------------------
# Evaluation helpers
# -----------------------
def eval_print(y_true, y_pred, probs, name):
    print(f"\n=== {name} ===")
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4))
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f}")
    p, r, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(r, p)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")

eval_print(y_test, pred_lr, probs_lr, "LogisticRegression (th=0.5)")
eval_print(y_test, pred_rf, probs_rf, "RandomForest (th=0.5)")

# -----------------------
# Calibration plots (safe across sklearn versions)
# -----------------------
plt.figure(figsize=(8,5))
try:
    
    CalibrationDisplay.from_predictions(y_test, probs_lr, n_bins=10, name='LogReg')
    CalibrationDisplay.from_predictions(y_test, probs_rf, n_bins=10, name='RF')
    plt.title("Calibration plots")
    plt.tight_layout()
    plt.show()
except Exception:
    
    from sklearn.calibration import calibration_curve
    plt.figure(figsize=(8,5))
    for probs, label in [(probs_lr, 'LogReg'), (probs_rf, 'RF')]:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
        plt.plot(mean_pred, frac_pos, marker='o', label=label)
    plt.plot([0,1],[0,1], linestyle='--', color='k')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.legend()
    plt.title("Calibration curves (fallback)")
    plt.tight_layout()
    plt.show()

# -----------------------
# Threshold tuning for recall priority
# -----------------------
def threshold_search(y_true, probs, metric='recall', thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.0,1.0,101)
    best = {'th':None, 'score':-1, 'prec':None, 'rec':None}
    for th in thresholds:
        pred = (probs >= th).astype(int)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred)
        score = rec if metric == 'recall' else prec
        if score > best['score']:
            best.update({'th':th, 'score':score, 'prec':prec, 'rec':rec})
    return best

best_lr = threshold_search(y_test, probs_lr, metric='recall')
best_rf = threshold_search(y_test, probs_rf, metric='recall')
print("\nBest thresholds for recall (LogReg):", best_lr)
print("Best thresholds for recall (RF):", best_rf)

# PR curve plot
plt.figure(figsize=(7,5))
p_lr, r_lr, _ = precision_recall_curve(y_test, probs_lr)
plt.plot(r_lr, p_lr, label='LogReg')
p_rf, r_rf, _ = precision_recall_curve(y_test, probs_rf)
plt.plot(r_rf, p_rf, label='RF')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------
# Map feature importances back to feature names (compatibility-safe)
# -----------------------
# Fit preprocessor to training data (already fitted via pipeline, but ensure ref)
preprocessor.fit(X_train)

# numeric names are numeric_features
num_names = numeric_features.copy()

# get onehot names robustly
cat_onehot_names = []
if categorical_features:
    try:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        # attempt get_feature_names_out then fallback
        try:
            cat_onehot_names = ohe.get_feature_names_out(categorical_features).tolist()
        except Exception:
            try:
                cat_onehot_names = ohe.get_feature_names(categorical_features).tolist()
            except Exception:
                # fallback: approximate names (col_val)
                categories = ohe.categories_
                names = []
                for col, cats in zip(categorical_features, categories):
                    names += [f"{col}__{str(val)}" for val in cats]
                cat_onehot_names = names
    except Exception as e:
        print("Warning: could not extract OHE feature names due to sklearn API differences:", e)
        cat_onehot_names = []

all_feature_names = num_names + cat_onehot_names
print(f"Total preprocessed feature count (approx): {len(all_feature_names)}")

# RF feature importances
rf_clf = rf_pipe.named_steps['clf']
if hasattr(rf_clf, "feature_importances_") and len(all_feature_names) == len(rf_clf.feature_importances_):
    fi = pd.Series(rf_clf.feature_importances_, index=all_feature_names).sort_values(ascending=False)
    print("\nTop RF features:\n", fi.head(15).to_string())
    plt.figure(figsize=(8,6))
    sns.barplot(x=fi.head(15).values, y=fi.head(15).index)
    plt.title("Top RF feature importances")
    plt.tight_layout()
    plt.show()
else:
    # if counts mismatch, still try to print top raw importances (index positions)
    try:
        fi_raw = rf_clf.feature_importances_
        top_idx = np.argsort(fi_raw)[::-1][:15]
        print("\nTop RF importances by index (could not map to names):")
        for i in top_idx:
            print(f"idx={i} importance={fi_raw[i]:.6f}")
    except Exception:
        print("Could not obtain RF importances.")

# -----------------------
# Pick & save model prioritizing recall
# -----------------------
lr_pred_at_best = (probs_lr >= best_lr['th']).astype(int)
rf_pred_at_best = (probs_rf >= best_rf['th']).astype(int)
rec_lr = recall_score(y_test, lr_pred_at_best)
rec_rf = recall_score(y_test, rf_pred_at_best)
print(f"\nRecall at tuned thresholds -> LR: {rec_lr:.4f} | RF: {rec_rf:.4f}")

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
if rec_rf >= rec_lr:
    chosen_pipe = rf_pipe
    joblib.dump(rf_pipe, models_dir / "diabetes_rf_pipeline.joblib")
    chosen = "rf"
else:
    chosen_pipe = lr_pipe
    joblib.dump(lr_pipe, models_dir / "diabetes_lr_pipeline.joblib")
    chosen = "lr"

with open(models_dir / "README_actions.txt-2", "w") as f:
    f.write("Chosen model prioritizes recall to capture patients at high risk of readmission within 30 days.\n")
    f.write("Suggested interventions: post-discharge follow-up calls, medication reconciliation, scheduling outpatient appointments.\n")
    f.write("Tune thresholds based on operational capacity (outreach volume vs acceptable false positive rate).\n")

print("Saved chosen model:", chosen)
print("Done.")

