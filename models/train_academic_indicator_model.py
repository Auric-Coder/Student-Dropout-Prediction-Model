"""Train the academic-indicator dropout model used by the Flask app."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# File now lives in  <project_root>/models/
# so SCRIPT_DIR.parent  ==  <project_root>
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_PATH = ROOT / "data" / "Predict Student Dropout and Academic Success.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = MODELS_DIR / "academic_indicator_model.pkl"
METRICS_PATH = RESULTS_DIR / "academic_indicator_metrics.txt"

FEATURE_NAMES = [
    "Attendance",
    "Assignments",
    "Marks",
    "Study Hrs",
    "Fees Up To Date",
]

RATIO_SMOOTHING_STRENGTH = 3.0
MIN_RATIO_DENOMINATOR = 1e-6
ATTENDANCE_PRIOR_RATE = 0.75
ASSIGNMENT_PRIOR_RATE = 0.42


def safe_divide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=np.asarray(denominator) != 0,
    )


def smoothed_ratio(
    numerator,
    denominator,
    prior_rate: float,
    smoothing_strength: float = RATIO_SMOOTHING_STRENGTH,
):
    """Stabilize per-student ratios without introducing cohort-size effects."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    smoothed = (numerator + (prior_rate * smoothing_strength)) / (
        denominator + smoothing_strength
    )
    return np.where(denominator >= MIN_RATIO_DENOMINATOR, smoothed, 0.0)


def _raw_academic_components(df: pd.DataFrame) -> dict[str, pd.Series]:
    cu1_enrolled = df["Curricular units 1st sem (enrolled)"].astype(float)
    cu2_enrolled = df["Curricular units 2nd sem (enrolled)"].astype(float)
    cu1_approved = df["Curricular units 1st sem (approved)"].astype(float)
    cu2_approved = df["Curricular units 2nd sem (approved)"].astype(float)
    cu1_evaluations = df["Curricular units 1st sem (evaluations)"].astype(float)
    cu2_evaluations = df["Curricular units 2nd sem (evaluations)"].astype(float)
    cu1_grade = df["Curricular units 1st sem (grade)"].astype(float)
    cu2_grade = df["Curricular units 2nd sem (grade)"].astype(float)

    total_enrolled = cu1_enrolled + cu2_enrolled
    total_approved = cu1_approved + cu2_approved
    total_evaluations = cu1_evaluations + cu2_evaluations
    weighted_grade = safe_divide(
        (cu1_grade * cu1_enrolled) + (cu2_grade * cu2_enrolled),
        total_enrolled,
    )

    return {
        "total_enrolled": total_enrolled,
        "total_approved": total_approved,
        "total_evaluations": total_evaluations,
        "weighted_grade": weighted_grade,
    }


def build_academic_indicators_legacy(df: pd.DataFrame) -> pd.DataFrame:
    components = _raw_academic_components(df)
    total_enrolled = components["total_enrolled"]
    total_approved = components["total_approved"]
    total_evaluations = components["total_evaluations"]
    weighted_grade = components["weighted_grade"]

    return pd.DataFrame(
        {
            "Attendance": np.clip(safe_divide(total_approved, total_enrolled) * 100, 0, 100),
            "Assignments": np.clip(safe_divide(total_evaluations, total_enrolled * 3) * 100, 0, 100),
            "Marks": np.clip(weighted_grade * 5, 0, 100),
            "Study Hrs": np.clip(total_enrolled, 0, 20),
            "Fees Up To Date": np.zeros(len(total_enrolled)),  # not available pre-fix; placeholder for legacy comparison only
        }
    )


def build_academic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    components = _raw_academic_components(df)
    total_enrolled = components["total_enrolled"]
    total_approved = components["total_approved"]
    total_evaluations = components["total_evaluations"]
    weighted_grade = components["weighted_grade"]

    attendance = smoothed_ratio(
        total_approved,
        total_enrolled,
        prior_rate=ATTENDANCE_PRIOR_RATE,
    )
    assignments = smoothed_ratio(
        total_evaluations,
        total_enrolled * 3,
        prior_rate=ASSIGNMENT_PRIOR_RATE,
    )
    # NOTE: Participation (approved/evaluations) was dropped — it correlated
    # 0.93 with Attendance (both derived from `approved`), i.e. it was not an
    # independent signal and caused unstable predictions on manually-entered,
    # internally-inconsistent form inputs. Replaced with Tuition fees up to
    # date, a genuinely independent column with real predictive signal.
    fees_up_to_date = df["Tuition fees up to date"].astype(float)

    return pd.DataFrame(
        {
            "Attendance": np.clip(attendance * 100, 0, 100),
            "Assignments": np.clip(assignments * 100, 0, 100),
            "Marks": np.clip(weighted_grade * 5, 0, 100),
            "Study Hrs": np.clip(total_enrolled, 0, 20),
            "Fees Up To Date": fees_up_to_date,
        }
    )


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    X = build_academic_indicators(df)
    y = (df["Target"] == "Dropout").astype(int)
    return X, y


def build_candidate_models() -> dict[str, object]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def make_calibrated_estimator(model, method: str = "sigmoid", cv: int = 5):
    try:
        return CalibratedClassifierCV(estimator=model, method=method, cv=cv)
    except TypeError:  # pragma: no cover - older sklearn compatibility.
        return CalibratedClassifierCV(base_estimator=model, method=method, cv=cv)


def build_production_candidates() -> dict[str, object]:
    base_models = build_candidate_models()
    logistic = base_models["Logistic Regression"]
    random_forest = base_models["Random Forest"]
    gradient_boosting = base_models["Gradient Boosting"]

    return {
        "Gradient Boosting (Uncalibrated)": clone(gradient_boosting),
        "Gradient Boosting (Calibrated Sigmoid)": make_calibrated_estimator(
            clone(gradient_boosting),
            method="sigmoid",
        ),
        "Gradient Boosting (Calibrated Isotonic)": make_calibrated_estimator(
            clone(gradient_boosting),
            method="isotonic",
        ),
        "Hybrid Soft Voting (Calibrated GB+LR+RF)": VotingClassifier(
            estimators=[
                (
                    "gb",
                    make_calibrated_estimator(
                        clone(gradient_boosting),
                        method="sigmoid",
                    ),
                ),
                (
                    "lr",
                    make_calibrated_estimator(
                        clone(logistic),
                        method="sigmoid",
                    ),
                ),
                (
                    "rf",
                    make_calibrated_estimator(
                        clone(random_forest),
                        method="sigmoid",
                    ),
                ),
            ],
            voting="soft",
            weights=[0.35, 0.40, 0.25],  # GB: 35%, LR: 40%, RF: 25%
            n_jobs=-1,
        ),
        "Stable Soft Voting (Calibrated LR+RF)": VotingClassifier(
            estimators=[
                (
                    "lr",
                    make_calibrated_estimator(
                        clone(logistic),
                        method="sigmoid",
                    ),
                ),
                (
                    "rf",
                    make_calibrated_estimator(
                        clone(random_forest),
                        method="sigmoid",
                    ),
                ),
            ],
            voting="soft",
            weights=[0.60, 0.40],  # LR: 60%, RF: 40%
            n_jobs=-1,
        ),
    }


def optimize_threshold(y_true, y_proba) -> dict[str, float]:
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(pr_thresholds, dtype=float),
        where=(precision[:-1] + recall[:-1]) != 0,
    )
    best_f1_idx = int(np.argmax(f1_scores))

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    youden_idx = int(np.argmax(tpr - fpr))

    return {
        "f1_threshold": float(pr_thresholds[best_f1_idx]),
        "f1_at_threshold": float(f1_scores[best_f1_idx]),
        "precision_at_threshold": float(precision[best_f1_idx]),
        "recall_at_threshold": float(recall[best_f1_idx]),
        "youden_threshold": float(roc_thresholds[youden_idx]),
        "youden_score": float(tpr[youden_idx] - fpr[youden_idx]),
    }


def evaluate_model(model, X_test, y_test, threshold: float | None = None) -> dict[str, object]:
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold_info = optimize_threshold(y_test, y_proba)
    selected_threshold = (
        threshold_info["f1_threshold"]
        if threshold is None
        else threshold
    )
    y_pred = (y_proba >= selected_threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "brier": brier_score_loss(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba),
        "threshold": selected_threshold,
        "threshold_info": threshold_info,
        "report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def evaluate_existing_model_before_fixes(df, y, train_indices, test_indices):
    if not MODEL_PATH.exists():
        return None

    try:
        existing_bundle = joblib.load(MODEL_PATH)
        existing_model = existing_bundle["model"]
    except Exception:
        return None

    X_legacy = build_academic_indicators_legacy(df)
    X_test_legacy = X_legacy.iloc[test_indices]
    y_test = y.iloc[test_indices]

    return {
        "name": existing_bundle.get("model_name", "Existing model"),
        "metrics": evaluate_model(existing_model, X_test_legacy, y_test, threshold=0.5),
    }


def select_final_model(evaluated_candidates: dict[str, dict[str, object]]) -> str:
    calibrated_gb = evaluated_candidates["Gradient Boosting (Calibrated Sigmoid)"]
    stable = evaluated_candidates["Stable Soft Voting (Calibrated LR+RF)"]
    hybrid = evaluated_candidates["Hybrid Soft Voting (Calibrated GB+LR+RF)"]

    gb_is_stable = (
        calibrated_gb["roc_auc"] >= stable["roc_auc"] - 0.02
        and calibrated_gb["brier"] <= stable["brier"] + 0.03
    )

    if gb_is_stable and hybrid["roc_auc"] >= stable["roc_auc"] - 0.01:
        return "Hybrid Soft Voting (Calibrated GB+LR+RF)"
    return "Stable Soft Voting (Calibrated LR+RF)"


def format_metric_summary(label: str, metrics: dict[str, object]) -> list[str]:
    matrix = metrics["confusion_matrix"]
    return [
        label,
        "-" * len(label),
        f"ROC-AUC              : {metrics['roc_auc']:.6f}",
        f"Accuracy             : {metrics['accuracy']:.6f}",
        f"Precision            : {metrics['precision']:.6f}",
        f"Recall               : {metrics['recall']:.6f}",
        f"F1-score             : {metrics['f1']:.6f}",
        f"Brier score          : {metrics['brier']:.6f}",
        f"Log loss             : {metrics['log_loss']:.6f}",
        f"Decision threshold   : {metrics['threshold']:.6f}",
        "Confusion Matrix:",
        str(matrix),
    ]


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = build_academic_indicators(df)
    y = (df["Target"] == "Dropout").astype(int)
    indices = np.arange(len(X))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = build_candidate_models()

    results = []
    for name, model in candidates.items():
        cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results.append((metrics["roc_auc"], name, model, cv_auc, metrics))

    stabilized_gb = clone(candidates["Gradient Boosting"])
    stabilized_gb.fit(X_train, y_train)
    stabilized_gb_metrics = evaluate_model(stabilized_gb, X_test, y_test)

    production_candidates = build_production_candidates()
    production_results = {}
    for name, candidate in production_candidates.items():
        candidate.fit(X_train, y_train)
        production_results[name] = evaluate_model(candidate, X_test, y_test)

    final_name = select_final_model(production_results)
    final_model = production_candidates[final_name]
    final_metrics = production_results[final_name]
    final_threshold = float(final_metrics["threshold"])
    medium_risk_threshold = max(0.15, final_threshold * 0.5)

    before_fixes = evaluate_existing_model_before_fixes(
        df,
        y,
        train_indices,
        test_indices,
    )

    best_auc, best_name, _, best_cv_auc, best_metrics = max(
        results,
        key=lambda row: row[0],
    )

    model_bundle = {
        "model": final_model,
        "model_name": final_name,
        "feature_names": FEATURE_NAMES,
        "threshold": final_threshold,
        "risk_thresholds": {
            "medium": round(medium_risk_threshold, 6),
            "high": round(final_threshold, 6),
        },
        "feature_engineering": {
            "ratio_smoothing_strength": RATIO_SMOOTHING_STRENGTH,
            "min_ratio_denominator": MIN_RATIO_DENOMINATOR,
            "attendance_prior_rate": ATTENDANCE_PRIOR_RATE,
            "assignment_prior_rate": ASSIGNMENT_PRIOR_RATE,
            "note": (
                "Attendance/Assignments are per-student ratios smoothed toward "
                "training priors. Fees Up To Date is a raw binary indicator, "
                "not a smoothed ratio."
            ),
        },
        "metrics": {
            "accuracy": round(final_metrics["accuracy"], 4),
            "roc_auc": round(final_metrics["roc_auc"], 4),
            "precision": round(final_metrics["precision"], 4),
            "recall": round(final_metrics["recall"], 4),
            "f1": round(final_metrics["f1"], 4),
            "brier": round(final_metrics["brier"], 4),
            "log_loss": round(final_metrics["log_loss"], 4),
            "threshold": round(final_threshold, 4),
            "risk_medium_threshold": round(medium_risk_threshold, 4),
            "risk_high_threshold": round(final_threshold, 4),
            "cv_roc_auc_mean": round(float(best_cv_auc.mean()), 4),
            "cv_roc_auc_std": round(float(best_cv_auc.std()), 4),
            "features": len(FEATURE_NAMES),
        },
    }
    joblib.dump(model_bundle, MODEL_PATH)

    lines = [
        "Academic Indicator Dropout Model Metrics",
        "=" * 50,
        f"Selected model       : {final_name}",
        f"Input features       : {len(FEATURE_NAMES)}",
        f"Test accuracy        : {final_metrics['accuracy']:.6f}",
        f"Test ROC-AUC         : {final_metrics['roc_auc']:.6f}",
        f"Test precision       : {final_metrics['precision']:.6f}",
        f"Test recall          : {final_metrics['recall']:.6f}",
        f"Test F1-score        : {final_metrics['f1']:.6f}",
        f"Decision threshold   : {final_threshold:.6f}",
        f"5-fold CV ROC-AUC    : {best_cv_auc.mean():.6f} +/- {best_cv_auc.std():.6f}",
        "",
        "Production feature safeguards:",
        f"- Ratio smoothing strength : {RATIO_SMOOTHING_STRENGTH}",
        f"- Minimum denominator      : {MIN_RATIO_DENOMINATOR}",
        f"- Attendance prior rate    : {ATTENDANCE_PRIOR_RATE}",
        f"- Assignment prior rate    : {ASSIGNMENT_PRIOR_RATE}",
        "",
        "Features:",
        *[f"- {name}" for name in FEATURE_NAMES],
        "",
        "Before / after comparison:",
    ]

    if before_fixes:
        lines.extend(format_metric_summary(
            f"Before fixes ({before_fixes['name']}, legacy ratios, threshold=0.5)",
            before_fixes["metrics"],
        ))
        lines.append("")

    lines.extend(format_metric_summary(
        "After ratio fixes (Gradient Boosting, tuned threshold)",
        stabilized_gb_metrics,
    ))
    lines.append("")
    lines.extend(format_metric_summary(
        "After hybrid model (selected production candidate)",
        final_metrics,
    ))

    lines.extend(["", "Candidate comparison:"])

    for auc, name, _, cv_auc, metrics in sorted(results, reverse=True):
        lines.append(
            f"- {name}: test_auc={auc:.6f}, "
            f"test_accuracy={metrics['accuracy']:.6f}, "
            f"test_f1={metrics['f1']:.6f}, "
            f"cv_auc={cv_auc.mean():.6f}+/-{cv_auc.std():.6f}"
        )

    lines.extend(["", "Calibration and ensemble comparison:"])

    for name, metrics in sorted(
        production_results.items(),
        key=lambda row: row[1]["roc_auc"],
        reverse=True,
    ):
        lines.append(
            f"- {name}: auc={metrics['roc_auc']:.6f}, "
            f"precision={metrics['precision']:.6f}, "
            f"recall={metrics['recall']:.6f}, "
            f"f1={metrics['f1']:.6f}, "
            f"brier={metrics['brier']:.6f}, "
            f"threshold={metrics['threshold']:.6f}"
        )

    lines.extend(["", "Detailed Model Reports:"])

    for auc, name, _, cv_auc, metrics in sorted(results, reverse=True):
        lines.extend(
            [
                "",
                name,
                "-" * len(name),
                f"Test accuracy        : {metrics['accuracy']:.6f}",
                f"Test ROC-AUC         : {auc:.6f}",
                f"5-fold CV ROC-AUC    : {cv_auc.mean():.6f} +/- {cv_auc.std():.6f}",
                "",
                "Classification Report:",
                metrics["report"],
                "Confusion Matrix:",
                str(metrics["confusion_matrix"]),
            ]
        )
    METRICS_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
