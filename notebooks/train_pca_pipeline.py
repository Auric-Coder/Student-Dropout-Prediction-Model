"""
train_pca_pipeline.py
=====================
Trains a sklearn Pipeline: StandardScaler → PCA → RandomForestClassifier
on the student dropout dataset and saves the full pipeline as a single
joblib artifact (models/model_pca_pipeline.pkl).

Usage:
    python notebooks/train_pca_pipeline.py

Run from the project root OR from the notebooks/ directory — path resolution
is automatic via the script's own __file__ location.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT        = SCRIPT_DIR.parent
DATA_PATH   = ROOT / "data" / "Predict Student Dropout and Academic Success.csv"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_PATH = MODELS_DIR / "model_pca_pipeline.pkl"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Target"] = df["Target"].apply(lambda x: 1 if x == "Dropout" else 0)
    X = df.drop("Target", axis=1)
    y = df["Target"]
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 2. Determine optimal PCA n_components
# ══════════════════════════════════════════════════════════════════════════════
def determine_n_components(X_train_scaled: np.ndarray,
                            threshold: float = 0.95) -> int:
    """
    Fits PCA with all components on X_train_scaled and returns the minimum
    number of components needed to capture `threshold` cumulative variance.
    Also saves the variance plot.
    """
    pca_full = PCA(n_components=X_train_scaled.shape[1], random_state=42)
    pca_full.fit(X_train_scaled)

    explained   = pca_full.explained_variance_ratio_
    cumulative  = np.cumsum(explained)
    n_comp      = int(np.argmax(cumulative >= threshold)) + 1

    # ── variance plot (dark theme) ──
    component_numbers = np.arange(1, len(explained) + 1)
    ACCENT = "#6366f1"; GREEN = "#22c55e"; AMBER = "#f59e0b"; MUTED = "#94a3b8"
    BG = "#0b0f1a"; BG_CARD = "#111827"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)

    for ax in axes:
        ax.set_facecolor(BG_CARD)
        ax.tick_params(colors=MUTED)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1f2937")

    colors = [ACCENT if i < n_comp else "#374151" for i in range(len(explained))]
    axes[0].bar(component_numbers, explained * 100, color=colors, width=0.7, zorder=3)
    axes[0].set_xlabel("Principal Component", fontsize=10)
    axes[0].set_ylabel("Explained Variance (%)", fontsize=10)
    axes[0].set_title("Individual Explained Variance", fontsize=12, fontweight="bold", pad=12)
    axes[0].grid(axis="y", color="#1f2937", linewidth=0.8, zorder=0)
    axes[0].set_xticks(component_numbers[::3])

    axes[1].plot(component_numbers, cumulative * 100,
                 color=ACCENT, linewidth=2.5, marker="o", markersize=4)
    axes[1].axhline(90, color=AMBER, linestyle="--", linewidth=1.4, label="90%")
    axes[1].axhline(95, color=GREEN, linestyle="--", linewidth=1.4, label="95%")
    axes[1].axvline(n_comp, color=GREEN, linestyle=":", linewidth=1.4)
    axes[1].scatter([n_comp], [cumulative[n_comp - 1] * 100],
                    color=GREEN, s=100, zorder=5,
                    label=f"PC{n_comp} ({cumulative[n_comp-1]*100:.1f}%)")
    axes[1].set_xlabel("Number of Components", fontsize=10)
    axes[1].set_ylabel("Cumulative Variance (%)", fontsize=10)
    axes[1].set_title("Cumulative Explained Variance", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_ylim(0, 103)
    axes[1].grid(color="#1f2937", linewidth=0.8, zorder=0)
    axes[1].legend(fontsize=9, facecolor="#1f2937", edgecolor="#374151", labelcolor="white")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())

    plt.tight_layout(pad=2.5)
    plot_path = PLOTS_DIR / "pca_variance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [plot] Variance chart saved → {plot_path}")

    return n_comp, cumulative[n_comp - 1], explained, cumulative


# ══════════════════════════════════════════════════════════════════════════════
# 3. Build & train pipeline
# ══════════════════════════════════════════════════════════════════════════════
def build_pipeline(n_components: int) -> Pipeline:
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components, random_state=42)),
        ("model",  RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 4. Evaluate & save metrics
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(pipeline: Pipeline, X_test, y_test,
             n_comp: int, var_captured: float, label: str) -> dict:
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred)
    cm      = confusion_matrix(y_test, y_pred)

    text = (
        f"{label}\n"
        f"{'=' * 50}\n"
        f"n_components        : {n_comp}\n"
        f"Variance captured   : {var_captured * 100:.4f}%\n"
        f"\nAccuracy  : {acc:.6f}\n"
        f"ROC-AUC   : {auc:.6f}\n\n"
        f"Classification Report:\n{report}\n"
        f"Confusion Matrix:\n{cm}\n"
    )
    return {"accuracy": acc, "auc": auc, "report": report, "text": text}


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60)
    print("  PCA Pipeline Training")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset …")
    X, y = load_data()
    print(f"  Samples : {len(X)} | Features : {X.shape[1]} | Dropout rate: {y.mean():.2%}")

    # ── Split ─────────────────────────────────────────────────────────────────
    print("\n[2/5] Train/test split (80/20, seed=42) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # ── PCA analysis on scaled training data ──────────────────────────────────
    print("\n[3/5] Running PCA analysis (95% variance threshold) …")
    scaler_probe = StandardScaler()
    X_train_scaled = scaler_probe.fit_transform(X_train)
    n_comp, var_captured, explained, cumulative = determine_n_components(
        X_train_scaled, threshold=0.95
    )
    print(f"  → Chosen n_components = {n_comp}  ({var_captured*100:.2f}% variance captured)")

    # Save PCA findings
    findings_lines = [
        "PCA Analysis Findings",
        "=" * 50,
        f"Total features     : {X.shape[1]}",
        f"Training samples   : {X_train.shape[0]}",
        "",
        f"n_components (≥90%): {int(np.argmax(cumulative >= 0.90)) + 1}",
        f"n_components (≥95%): {n_comp}",
        "",
        f"SELECTED n_components = {n_comp}",
        f"Variance captured     = {var_captured*100:.4f}%",
        f"Dimensionality ratio  = {n_comp}/{X.shape[1]} = {n_comp/X.shape[1]:.2%}",
        "",
        "Top 10 components:",
    ]
    for i in range(min(10, len(explained))):
        findings_lines.append(
            f"  PC{i+1:2d}: {explained[i]*100:6.3f}%  (cumulative: {cumulative[i]*100:.3f}%)"
        )
    (RESULTS_DIR / "pca_findings.txt").write_text("\n".join(findings_lines), encoding="utf-8")
    print(f"  [saved] pca_findings.txt")

    # ── Train pipeline ────────────────────────────────────────────────────────
    print("\n[4/5] Training Pipeline (Scaler → PCA → RandomForest) …")
    pipeline = build_pipeline(n_comp)
    pipeline.fit(X_train, y_train)
    print("  Training complete.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating …")
    metrics = evaluate(pipeline, X_test, y_test, n_comp, var_captured,
                       "PCA RandomForest Pipeline Metrics")
    print(metrics["text"])

    # Save metrics
    metrics_path = RESULTS_DIR / "pca_rf_metrics.txt"
    metrics_path.write_text(metrics["text"], encoding="utf-8")
    print(f"  [saved] pca_rf_metrics.txt  (Accuracy={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f})")

    # ── Save pipeline ─────────────────────────────────────────────────────────
    joblib.dump(pipeline, PIPELINE_PATH)
    size_mb = PIPELINE_PATH.stat().st_size / (1024 * 1024)
    print(f"\n  [saved] model_pca_pipeline.pkl  ({size_mb:.1f} MB)")

    # ── Compare to original baseline ─────────────────────────────────────────
    baseline_path = RESULTS_DIR / "rf_metrics.txt"
    if baseline_path.exists():
        baseline_text = baseline_path.read_text(encoding="utf-8", errors="ignore")
        import re
        orig_acc = re.search(r"Accuracy:\s*([\d.]+)", baseline_text)
        if orig_acc:
            orig = float(orig_acc.group(1))
            delta = metrics["accuracy"] - orig
            direction = "improvement" if delta >= 0 else "regression"
            print(f"\n  -- Baseline Comparison --")
            print(f"  Original RF accuracy    : {orig:.4f}")
            print(f"  PCA pipeline accuracy   : {metrics['accuracy']:.4f}")
            print(f"  Delta                   : {delta:+.4f}  ({direction})")

    print("\n" + "=" * 60)
    print("  Pipeline saved to models/model_pca_pipeline.pkl")
    print("  Update app.py to load this file.")
    print("=" * 60 + "\n")

    return pipeline


if __name__ == "__main__":
    main()
