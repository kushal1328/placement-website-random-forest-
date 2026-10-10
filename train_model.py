"""
train_model.py  -  trains the placement models and exports everything the website needs.

Run once before starting the website:
    python train_model.py

Outputs (in the ./model folder):
    model_bundle.joblib   the trained models (Random Forest + Logistic Regression)
    model_info.json       metrics, feature ranges, cut-offs and chart data shown on the website
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "placement_train.csv"
OUT_DIR = BASE_DIR / "model"
SEED = 42
THRESHOLD = 0.5            # probability at or above this  ->  class "Placed"

TARGET = "Placement_Status"
NUMERIC = ["CGPA", "Aptitude_Test_Score", "Backlogs", "Internships", "Projects",
           "Certifications", "Coding_Skills", "Communication_Skills", "Soft_Skills_Rating"]
CATEGORICAL = ["Degree", "Branch"]
# Deliberately NOT used as inputs: Student_ID (meaningless label), Gender and Age (fairness).

LABELS = {
    "CGPA": "CGPA", "Aptitude_Test_Score": "Aptitude score", "Backlogs": "Backlogs",
    "Internships": "Internships", "Projects": "Projects", "Certifications": "Certifications",
    "Coding_Skills": "Coding skills", "Communication_Skills": "Communication skills",
    "Soft_Skills_Rating": "Soft skills", "Degree": "Degree", "Branch": "Branch",
}
LOWER_IS_BETTER = {"Backlogs"}


def build_pipeline(model, scale):
    """Preprocessing + model in one object, so the website can feed raw values straight in."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),            # missing numbers -> median
                        ("scale", StandardScaler() if scale else "passthrough")])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),  # missing text -> most common
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])  # text -> 0/1 columns
    pre = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    return Pipeline([("pre", pre), ("model", model)])


def metrics_for(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    fpr, tpr, _ = roc_curve(y_test, proba)
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 60)).astype(int))
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in keep],
    }, pred


def main():
    # ------------------------------------------------------------ 1. load and clean
    df = pd.read_csv(DATA_FILE)
    needed = set(NUMERIC + CATEGORICAL + [TARGET])
    missing_cols = needed - set(df.columns)
    if missing_cols:
        raise SystemExit(f"Columns missing from the CSV: {sorted(missing_cols)}")
    rows_before = len(df)
    df = df.drop_duplicates().dropna(subset=[TARGET]).reset_index(drop=True)
    y = (df[TARGET] == "Placed").astype(int)
    X = df[NUMERIC + CATEGORICAL]
    print(f"Loaded {rows_before:,} rows, using {len(df):,} after cleaning. "
          f"Placed: {y.mean():.1%}  |  Not placed: {1 - y.mean():.1%}")

    # ------------------------------------------------------------ 2. split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)

    rf = build_pipeline(RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1), scale=False)
    lr = build_pipeline(LogisticRegression(max_iter=2000), scale=True)
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    # ------------------------------------------------------------ 3. evaluate on unseen students
    rf_metrics, rf_pred = metrics_for(rf, X_test, y_test)
    lr_metrics, _ = metrics_for(lr, X_test, y_test)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, model, m in [("random forest", rf, rf_metrics), ("logistic regression", lr, lr_metrics)]:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        m["cv_roc_auc_mean"], m["cv_roc_auc_std"] = float(scores.mean()), float(scores.std())
        print(f"{name:20s} accuracy {m['accuracy']:.3f}  precision {m['precision']:.3f}  "
              f"recall {m['recall']:.3f}  F1 {m['f1']:.3f}  ROC-AUC {m['roc_auc']:.3f}  "
              f"(5-fold AUC {scores.mean():.3f})")

    # ------------------------------------------------------------ 4. what drives the model
    names = rf.named_steps["pre"].get_feature_names_out()
    raw = rf.named_steps["model"].feature_importances_
    importance = {}
    for n, v in zip(names, raw):
        n = n.split("__", 1)[1]
        original = next((c for c in CATEGORICAL if n.startswith(c + "_")), n)   # merge one-hot columns back
        importance[original] = importance.get(original, 0.0) + float(v)
    importance = sorted(({"feature": k, "label": LABELS[k], "value": v / sum(importance.values())}
                         for k, v in importance.items()), key=lambda d: -d["value"])

    # ------------------------------------------------------------ 5. data-driven minimum levels
    # "No student below this level was placed" - found from the data, not typed by hand.
    cutoffs = {}
    placed = df[y == 1]
    for f in NUMERIC:
        lo, hi = float(df[f].min()), float(df[f].max())
        if f in LOWER_IS_BETTER:
            c = float(placed[f].max())
            if c < hi:
                cutoffs[f] = {"direction": "max", "value": c}
        else:
            c = float(placed[f].min())
            if c > lo:
                cutoffs[f] = {"direction": "min", "value": c}
    for f, d in cutoffs.items():
        mask = df[f] < d["value"] if d["direction"] == "min" else df[f] > d["value"]
        d["students_outside"] = int(mask.sum())
        d["placed_outside"] = int(y[mask].sum())

    # ------------------------------------------------------------ 6. fairness audit (Gender is NOT a model input)
    audit = pd.DataFrame({"Gender": df.loc[X_test.index, "Gender"], "actual": y_test, "pred": rf_pred})
    fairness = [{"group": g, "n": int(len(d)), "actual_rate": float(d["actual"].mean()),
                 "predicted_rate": float(d["pred"].mean()),
                 "recall": float(recall_score(d["actual"], d["pred"]))}
                for g, d in audit.groupby("Gender")]

    # ------------------------------------------------------------ 7. save
    OUT_DIR.mkdir(exist_ok=True)
    joblib.dump({"rf": rf, "lr": lr, "threshold": THRESHOLD}, OUT_DIR / "model_bundle.joblib")
    info = {
        "dataset": {"rows": int(len(df)), "placed": int(y.sum()), "not_placed": int((1 - y).sum()),
                    "placement_rate": float(y.mean()), "test_rows": int(len(X_test))},
        "threshold": THRESHOLD,
        "numeric_features": NUMERIC, "categorical_features": CATEGORICAL, "labels": LABELS,
        "ranges": {f: [float(df[f].min()), float(df[f].max())] for f in NUMERIC},
        "defaults": {**{f: float(df[f].median()) for f in NUMERIC},
                     **{f: str(df[f].mode()[0]) for f in CATEGORICAL}},
        "categories": {f: sorted(df[f].dropna().unique().tolist()) for f in CATEGORICAL},
        "models": {"random_forest": rf_metrics, "logistic_regression": lr_metrics},
        "importance": importance, "cutoffs": cutoffs, "fairness": fairness,
    }
    (OUT_DIR / "model_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nSaved model files to {OUT_DIR}")
    print("Minimum levels found in the data:")
    for f, d in cutoffs.items():
        sign = ">=" if d["direction"] == "min" else "<="
        print(f"  {LABELS[f]:22s} {sign} {d['value']:g}   (placed outside: {d['placed_outside']} of {d['students_outside']:,})")


if __name__ == "__main__":
    main()
