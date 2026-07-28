"""Student dropout risk predictor."""

from __future__ import annotations

import os
import warnings

import joblib
import pandas as pd
from flask import Flask, render_template, request

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_FILE = os.path.join(MODELS_DIR, "academic_indicator_model.pkl")

app = Flask(__name__)

model_bundle = joblib.load(MODEL_FILE)
model = model_bundle["model"]
FEATURE_NAMES = model_bundle["feature_names"]
CLASSIFICATION_THRESHOLD = float(model_bundle.get("threshold", 0.5))
RISK_THRESHOLDS = {
    "medium": 0.35,
    "high": 0.60,
    **model_bundle.get("risk_thresholds", {}),
}
MODEL_METRICS = {
    **model_bundle.get("metrics", {}),
    "model_name": model_bundle.get("model_name", "Ensemble Model"),
}

PROFILE_FIELDS = [
    ("name", "Name", "text", {"placeholder": "Student name"}),
    (
        "dept",
        "Dept",
        "select",
        [
            ("CSE", "CSE"),
            ("AI&DS", "AI&DS"),
            ("IT", "IT"),
            ("ECE", "ECE"),
            ("EEE", "EEE"),
            ("MECH", "MECH"),
            ("CIVIL", "CIVIL"),
        ],
    ),
    ("roll_no", "Roll No", "text", {"placeholder": "Roll number"}),
    (
        "sem",
        "Sem",
        "select",
        [(str(num), str(num)) for num in range(1, 9)],
    ),
]

# Each field is: slug, training column, label, input type, options/attrs, valid range.
INDICATOR_FIELDS = [
    (
        "attendance",
        "Attendance",
        "Attendance",
        "number",
        {"min": 0, "max": 100, "step": 1, "placeholder": "0-100"},
        (0, 100),
    ),
    (
        "assignments",
        "Assignments",
        "Assignments",
        "number",
        {"min": 0, "max": 100, "step": 1, "placeholder": "0-100"},
        (0, 100),
    ),
    (
        "marks",
        "Marks",
        "Marks",
        "number",
        {"min": 0, "max": 100, "step": 1, "placeholder": "0-100"},
        (0, 100),
    ),
    (
        "study_hrs",
        "Study Hrs",
        "Study Hrs",
        "number",
        {"min": 0, "max": 20, "step": 0.5, "placeholder": "0-20"},
        (0, 20),
    ),
    (
        "fees_up_to_date",
        "Fees Up To Date",
        "Fees Up To Date",
        "select",
        [("1", "Yes"), ("0", "No")],
        (0, 1),
    ),
]

assert [field[1] for field in INDICATOR_FIELDS] == FEATURE_NAMES


def parse_profile(form_data: dict) -> dict[str, str]:
    profile = {}
    for field_slug, label, _, _ in PROFILE_FIELDS:
        value = form_data.get(field_slug, "").strip()
        if field_slug in {"name", "roll_no"} and not value:
            raise ValueError(f'Field "{label}" is required.')
        profile[field_slug] = value
    return profile


def parse_indicators(form_data: dict) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    values = []
    warnings_list = []
    clean_form_data = {}

    for field_slug, _, label, field_type, field_config, (vmin, vmax) in INDICATOR_FIELDS:
        raw = form_data.get(field_slug, "").strip()
        if raw == "":
            raise ValueError(f'Field "{label}" is required.')

        if field_type == "select":
            allowed_values = {value for value, _ in field_config}
            if raw not in allowed_values:
                raise ValueError(f'Field "{label}" must be a valid selection.')

        try:
            val = float(raw)
        except ValueError as exc:
            raise ValueError(f'Field "{label}" must be a valid number.') from exc

        if val < vmin:
            warnings_list.append(f'"{label}" was below {vmin} and was clamped.')
            val = float(vmin)
        elif val > vmax:
            warnings_list.append(f'"{label}" was above {vmax} and was clamped.')
            val = float(vmax)

        values.append(val)
        clean_form_data[field_slug] = val

    return pd.DataFrame([values], columns=FEATURE_NAMES), warnings_list, clean_form_data


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        profile_fields=PROFILE_FIELDS,
        indicator_fields=INDICATOR_FIELDS,
        metrics=MODEL_METRICS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        profile_data = parse_profile(request.form)
        x_raw, input_warnings, indicator_data = parse_indicators(request.form)
    except ValueError as exc:
        return render_template(
            "index.html",
            profile_fields=PROFILE_FIELDS,
            indicator_fields=INDICATOR_FIELDS,
            metrics=MODEL_METRICS,
            error=str(exc),
        ), 400

    try:
        proba = model.predict_proba(x_raw)[0]
    except Exception as exc:  # pragma: no cover - defensive app boundary.
        return render_template(
            "index.html",
            profile_fields=PROFILE_FIELDS,
            indicator_fields=INDICATOR_FIELDS,
            metrics=MODEL_METRICS,
            error=f"Model prediction error: {exc}",
        ), 500

    dropout_prob = float(proba[1])
    success_prob = float(proba[0])
    prediction = int(dropout_prob >= CLASSIFICATION_THRESHOLD)

    if dropout_prob >= RISK_THRESHOLDS["high"]:
        risk_level, risk_class = "HIGH RISK", "high"
    elif dropout_prob >= RISK_THRESHOLDS["medium"]:
        risk_level, risk_class = "MEDIUM RISK", "medium"
    else:
        risk_level, risk_class = "LOW RISK", "low"

    return render_template(
        "result.html",
        dropout_prob=round(dropout_prob * 100, 1),
        success_prob=round(success_prob * 100, 1),
        risk_level=risk_level,
        risk_class=risk_class,
        prediction=prediction,
        profile_data=profile_data,
        indicator_data=indicator_data,
        input_warnings=input_warnings,
        metrics=MODEL_METRICS,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
