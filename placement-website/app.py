"""
app.py  -  the website's backend.

Start it with:   python app.py        (then open http://127.0.0.1:5000)
Needs the ./model folder, which is created by:   python train_model.py
"""
import json
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


def load_model_files():
    bundle_path, info_path = MODEL_DIR / "model_bundle.joblib", MODEL_DIR / "model_info.json"
    if not (bundle_path.exists() and info_path.exists()):
        raise SystemExit("Model files not found.\nRun this first:   python train_model.py")
    return joblib.load(bundle_path), json.loads(info_path.read_text())


BUNDLE, INFO = load_model_files()
RF, LR, THRESHOLD = BUNDLE["rf"], BUNDLE["lr"], BUNDLE["threshold"]
NUMERIC, CATEGORICAL = INFO["numeric_features"], INFO["categorical_features"]
RANGES, LABELS = INFO["ranges"], INFO["labels"]

app = Flask(__name__)

# ---------------------------------------------------------------------------
# The input form is generated from the data, so it always matches the trained model.
# ---------------------------------------------------------------------------
FIELD_TEXT = {
    "CGPA": ("Cumulative grade point average, out of 10", 0.01),
    "Aptitude_Test_Score": ("Aptitude test score, out of 100", 1),
    "Backlogs": ("Subjects failed or still pending", 1),
    "Internships": ("Internships completed", 1),
    "Projects": ("Projects completed", 1),
    "Certifications": ("Courses or certificates completed", 1),
    "Coding_Skills": ("Your own rating, 1 (beginner) to 10 (expert)", 1),
    "Communication_Skills": ("Your own rating, 1 (beginner) to 10 (expert)", 1),
    "Soft_Skills_Rating": ("Teamwork, adaptability, time management; 1 to 10", 1),
}
FORM_GROUPS = [
    ("Academics", ["CGPA", "Aptitude_Test_Score", "Backlogs"]),
    ("Experience", ["Internships", "Projects", "Certifications"]),
    ("Skills", ["Coding_Skills", "Communication_Skills", "Soft_Skills_Rating"]),
]


def build_form_spec():
    groups = []
    for title, fields in FORM_GROUPS:
        items = []
        for f in fields:
            lo, hi = RANGES[f]
            hint, step = FIELD_TEXT[f]
            items.append({"name": f, "label": LABELS[f], "hint": hint, "min": lo, "max": hi,
                          "step": step, "value": INFO["defaults"][f]})
        groups.append({"title": title, "fields": items})
    return groups


# Improvements the "fastest gains" panel tries: feature -> (step size, max steps, +1 or -1 direction)
ACTIONS = {
    "CGPA": (0.5, 3, +1), "Aptitude_Test_Score": (5, 3, +1), "Coding_Skills": (1, 3, +1),
    "Communication_Skills": (1, 3, +1), "Projects": (1, 2, +1), "Certifications": (1, 2, +1),
    "Internships": (1, 2, +1), "Backlogs": (1, 2, -1),
}


# ---------------------------------------------------------------------------
# Validation and prediction
# ---------------------------------------------------------------------------
def parse_input(data):
    row, errors = {}, {}
    for f in NUMERIC:
        try:
            v = float(data.get(f))
            if not math.isfinite(v):
                raise ValueError
        except (TypeError, ValueError):
            errors[f] = "Enter a number."
            continue
        lo, hi = RANGES[f]
        if not lo <= v <= hi:
            errors[f] = f"Use a value from {lo:g} to {hi:g}, the range the model was trained on."
            continue
        row[f] = v
    for f in CATEGORICAL:
        v = data.get(f)
        if v not in INFO["categories"][f]:
            errors[f] = "Choose one of the listed options."
        else:
            row[f] = v
    return row, errors


def p_placed(model, row):
    return float(model.predict_proba(pd.DataFrame([row]))[0, 1])


def outlook(p):
    return "High" if p >= 0.70 else "Medium" if p >= 0.40 else "Low"


def build_checklist(row):
    items = []
    for f, c in INFO["cutoffs"].items():
        v, target = row[f], c["value"]
        passed = v >= target if c["direction"] == "min" else v <= target
        items.append({"feature": f, "label": LABELS[f], "value": v, "target": target,
                      "direction": c["direction"], "passed": bool(passed)})
    return items


def fastest_gains(row, base_p):
    """Try small improvements one at a time and report which raise the probability most."""
    candidates = []
    for f, (step, max_steps, sign) in ACTIONS.items():
        lo, hi = RANGES[f]
        for k in range(1, max_steps + 1):
            new = min(max(row[f] + sign * step * k, lo), hi)
            if new == row[f]:
                break
            candidates.append((f, new, {**row, f: new}))
    if not candidates:
        return []
    probs = RF.predict_proba(pd.DataFrame([c[2] for c in candidates]))[:, 1]
    best = {}
    for (f, new, _), p in zip(candidates, probs):
        gain = float(p) - base_p
        if gain > best.get(f, {"gain": 0.0})["gain"] + 1e-9:     # strictly better, so the smallest effort wins ties
            best[f] = {"feature": f, "label": LABELS[f], "from": row[f], "to": float(new),
                       "new_probability": float(p), "gain": gain}
    return sorted((b for b in best.values() if b["gain"] >= 0.02), key=lambda b: -b["gain"])[:4]


def probability_if_gaps_closed(row, checklist):
    failed = [c for c in checklist if not c["passed"]]
    if not failed:
        return None
    fixed = dict(row)
    for c in failed:
        fixed[c["feature"]] = c["target"]
    return p_placed(RF, fixed)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return render_template("index.html", info=INFO, form_groups=build_form_spec(),
                           degrees=INFO["categories"]["Degree"], branches=INFO["categories"]["Branch"],
                           defaults=INFO["defaults"])


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "errors": {"_": "Send the student details as JSON."}}), 400
    row, errors = parse_input(data)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 422

    p_rf, p_lr = p_placed(RF, row), p_placed(LR, row)
    label = "Placed" if p_rf >= THRESHOLD else "Not placed"
    checklist = build_checklist(row)
    return jsonify({
        "ok": True,
        "prediction": {                                   # <- the two outputs the competition asks for
            "class": int(p_rf >= THRESHOLD),
            "label": label,
            "probability_placed": round(p_rf, 4),
            "probability_not_placed": round(1 - p_rf, 4),
            "threshold": THRESHOLD,
            "outlook": outlook(p_rf),
            "model": "Random Forest",
        },
        "second_opinion": {"model": "Logistic Regression", "probability_placed": round(p_lr, 4),
                           "label": "Placed" if p_lr >= THRESHOLD else "Not placed"},
        "models_agree": (p_rf >= THRESHOLD) == (p_lr >= THRESHOLD),
        "checklist": checklist,
        "probability_if_gaps_closed": probability_if_gaps_closed(row, checklist),
        "fastest_gains": fastest_gains(row, p_rf),
    })


@app.get("/api/info")
def info():
    return jsonify(INFO)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
