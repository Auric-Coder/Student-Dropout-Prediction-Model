# data/

This folder holds the raw dataset and reference material used by the project.

## Required file — must be downloaded manually

The raw CSV is **not tracked by Git** (see `.gitignore`).  
Download it and place it here **before** running the training script:

```
data/Predict Student Dropout and Academic Success.csv
```

**Download links:**
- Kaggle: <https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention>
- UCI ML Repository: <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success>

## Contents

| File | Tracked | Description |
|------|---------|-------------|
| `Predict Student Dropout and Academic Success.csv` | ❌ No | Raw dataset (~4 424 students, 36 features) |
| `Early-Prediction-of-University-Student-Dropout-Using-Machine-Learning-Models.pdf` | ✅ Yes | Reference research paper |
| `README.md` | ✅ Yes | This file |

## Dataset overview

| Property | Value |
|----------|-------|
| Records | ~4 424 students |
| Original features | 36 (demographic, socio-economic, academic) |
| Target classes | Graduate · Enrolled · **Dropout** |
| Source institution | Multiple Portuguese university programmes |

The training script (`models/train_academic_indicator_model.py`) reduces the 36 raw features down to **5 Bayesian-smoothed academic indicators** before model training.
