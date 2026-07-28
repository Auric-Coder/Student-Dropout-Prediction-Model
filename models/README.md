# DropoutQI — Student Dropout Risk Predictor

Predicts a student's dropout risk from five academic/administrative indicators using a
calibrated soft-voting ensemble (Gradient Boosting + Logistic Regression + Random Forest).

## Model inputs

| Feature | Description | Range |
|---|---|---|
| Attendance | Smoothed approved/enrolled curricular-unit ratio | 0–100 |
| Assignments | Smoothed evaluations/enrolled ratio | 0–100 |
| Marks | Enrollment-weighted grade across both semesters | 0–100 |
| Study Hrs | Total curricular units enrolled | 0–20 |
| Fees Up To Date | Whether tuition is currently paid up (1 = Yes, 0 = No) | 0 or 1 |

## Changelog: Participation → Fees Up To Date (2026-07-28)

**Problem:** The original 5th feature, `Participation`, was computed as
`approved / evaluations`. Since `Attendance` is `approved / enrolled` and
`Assignments` is `evaluations / enrolled`, `Participation` was algebraically
derivable from the other two — not an independent signal.

Measured correlation between `Participation` and `Attendance` on the training
set: **0.927**. This near-duplication meant a user could type an internally
inconsistent combination on the form (e.g. high attendance, low assignments,
high participation) that never occurs in real student records. The model,
having never seen that region of feature space, produced unstable/anomalous
predictions for such inputs.

**Fix:** Replaced `Participation` with `Tuition fees up to date`, a raw
column from the source dataset that carries genuinely independent
information (financial standing) not derivable from the academic ratios.

**Result — same train/test split, same production pipeline, same model
selection logic:**

| | Test ROC-AUC | 5-fold CV ROC-AUC | Test Accuracy | Test F1 |
|---|---|---|---|---|
| Before (Participation) | 0.9162 | 0.8916 ± 0.0137 | 0.8633 | 0.7873 |
| **After (Fees Up To Date)** | **0.9284** | **0.9086 ± 0.0083** | **0.8780** | **0.8099** |

- ROC-AUC improved **+1.2 points** on the test set, **+1.7 points** on 5-fold CV.
- CV variance dropped from ±0.0137 to ±0.0083 — the model also got more stable
  across folds, not just more accurate on average.
- Selected production model remains **Hybrid Soft Voting (Calibrated GB+LR+RF)**.

Full metrics: see `results/academic_indicator_metrics.txt`.

## Retraining

```bash
python models/train_academic_indicator_model.py
```

Requires `data/Predict Student Dropout and Academic Success.csv` (UCI —
"Predict Students' Dropout and Academic Success" dataset) in place. Writes
`models/academic_indicator_model.pkl` and `results/academic_indicator_metrics.txt`.

## Running the app

```bash
python app.py
```

Flask app on `localhost:5000`. Loads the bundled model, validates/clamps the
five form inputs, and returns a LOW/MEDIUM/HIGH risk classification with
calibrated dropout probability.
