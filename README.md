# Student Dropout & Academic Success Predictor 🎓

A machine-learning web application that predicts a university student's **dropout risk** based on five academic and fee-status indicators. The production model is a **Hybrid Soft-Voting Ensemble** (Calibrated Gradient Boosting + Logistic Regression + Random Forest) served through a Flask web interface.

---

## 📋 Table of Contents
- [Demo](#demo)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Model Performance](#model-performance)
- [Getting Started](#getting-started)
- [Running the App](#running-the-app)
- [Retraining the Model](#retraining-the-model)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Demo

Enter five simple academic metrics and get an instant risk classification:

| Risk Level | Dropout Probability |
|------------|---------------------|
| 🟢 LOW RISK | < 35% |
| 🟡 MEDIUM RISK | 35% – 60% |
| 🔴 HIGH RISK | ≥ 60% |

The app also surfaces the student's name, department, roll number, and semester alongside the prediction to provide full context.

---

## Project Structure

```
SuccessAnddropout2.0/
│
├── app/                              # Flask web application
│   ├── app.py                        # Main Flask app & prediction logic
│   ├── static/
│   │   ├── style.css                 # UI styles
│   │   └── main.js                   # Frontend interactivity
│   └── templates/
│       ├── index.html                # Input form page
│       └── result.html               # Prediction result page
│
├── data/                             # Dataset (not tracked by git — see Dataset section)
│   ├── Predict Student Dropout and Academic Success.csv
│   ├── Early-Prediction-of-University-Student-Dropout-Using-Machine-Learning-Models.pdf
│   └── README.md
│
├── models/
│   ├── train_academic_indicator_model.py   # Full training & evaluation script
│   ├── model_pipeline.ipynb                # Pipeline exploration notebook
│   └── academic_indicator_model.pkl        # Trained model bundle (generated — not tracked)
│
├── notebooks/                        # EDA & experimentation notebooks
│   ├── dataExploration.ipynb
│   ├── featureEngineering.ipynb
│   ├── modelTraining.ipynb           # Trains LR, DT, RF, SVM + reports
│   ├── pca_analysis.ipynb
│   ├── academic_indicator_model.ipynb
│   └── train_pca_pipeline.py
│
├── results/
│   ├── academic_indicator_metrics.txt      # Last training run metrics
│   └── plots/                             # Generated evaluation plots
│       ├── ageVsDropout.png
│       ├── dropoutVsApproved.png
│       ├── dropoutVsCurriculum.png
│       ├── feesVsDropout.png
│       ├── heatmap.png
│       ├── pca_variance.png
│       └── scholarshipVsDropout.png
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## How It Works

Raw dataset columns are transformed into **5 human-readable academic indicators** before being passed to the model. Ratios are **Bayesian-smoothed** toward training-set priors to prevent extreme values from small cohorts from destabilising predictions:

| Indicator | Derived From | Range |
|-----------|-------------|-------|
| **Attendance** | Smoothed ratio of approved / enrolled units × 100 | 0 – 100 |
| **Assignments** | Smoothed ratio of evaluations / (enrolled × 3) × 100 | 0 – 100 |
| **Marks** | Weighted semester grade (by enrolled units) × 5 | 0 – 100 |
| **Study Hrs** | Total enrolled curricular units (capped at 20) | 0 – 20 |
| **Fees Up To Date** | Whether the student's tuition payments are current (1 = Yes, 0 = No) | 0 or 1 |

**Smoothing parameters used in production:**

| Parameter | Value |
|-----------|-------|
| Ratio smoothing strength | 3.0 |
| Attendance prior rate | 0.75 |
| Assignment prior rate | 0.42 |

---

## Model Architecture

Training proceeds in two stages:

### Stage 1 — Candidate comparison
Four individual models are trained, cross-validated (5-fold stratified), and compared by ROC-AUC:

| Model | Test ROC-AUC | Test Accuracy | CV ROC-AUC |
|-------|-------------|---------------|------------|
| Gradient Boosting | 0.9254 | 86.3% | 0.909 ± 0.008 |
| Logistic Regression | 0.9181 | 87.3% | 0.897 ± 0.006 |
| Random Forest | 0.9201 | 86.6% | 0.900 ± 0.010 |
| **SVM (RBF)** | 0.9008 | 84.9% | 0.884 ± 0.009 |

> *Numbers above are from Stage 1 exploration; final production model is selected from Stage 2 calibrated ensembles.*

### Stage 2 — Calibrated ensemble selection
Five calibrated / ensemble variants are built and the best is selected automatically:

| Ensemble Variant | Calibrated Weights | ROC-AUC | F1 | Brier |
|------------------|--------------------|---------|-----|-------|
| **Hybrid Soft Voting (Cal. GB+LR+RF)** ✅ | GB 35% · LR 40% · RF 25% | **0.9284** | **0.810** | **0.0929** |
| Stable Soft Voting (Cal. LR+RF) | LR 60% · RF 40% | 0.9256 | 0.807 | 0.0938 |
| GB Calibrated Isotonic | — | 0.9247 | 0.793 | 0.0945 |
| GB Calibrated Sigmoid | — | 0.9242 | 0.795 | 0.0942 |
| GB Uncalibrated | — | 0.9254 | 0.801 | 0.0941 |

**Ensemble weight rationale:**
- **Hybrid (GB+LR+RF):** LR carries the most weight (40%) for its stable probability outputs; GB contributes 35% to capture non-linear interactions; RF is down-weighted (25%) to reduce variance.
- **Stable (LR+RF):** LR is dominant (60%) for its strong calibration; RF provides 40% to capture complex patterns.

The **Hybrid Soft Voting** ensemble is automatically selected when it is within 1% AUC of the stable LR+RF baseline — ensuring the production model is both accurate and well-calibrated.

The decision threshold is optimised for F1 (Precision-Recall curve) rather than fixed at 0.5, and the serialised model bundle stores it alongside risk-band thresholds so the Flask app needs zero manual configuration.

---

## Model Performance

Metrics from the last training run (`results/academic_indicator_metrics.txt`):

| Metric | Score |
|--------|-------|
| **Selected Model** | Hybrid Soft Voting (Calibrated GB+LR+RF) |
| **Ensemble Weights** | GB 35% · LR 40% · RF 25% |
| **Test Accuracy** | **87.8%** |
| **Test ROC-AUC** | **0.9284** |
| **Test Precision** | **0.810** |
| **Test Recall** | **0.810** |
| **Test F1-score** | **0.810** |
| **Brier Score** | **0.0929** |
| **Decision Threshold** | **0.432** |
| **5-Fold CV ROC-AUC** | **0.9086 ± 0.0083** |

**Confusion Matrix (Test Set — 885 students):**
```
              Predicted: Graduate  Predicted: Dropout
Actual: Graduate        547               54
Actual: Dropout          54              230
```

---

## Getting Started

### Prerequisites
- Python **3.10+**

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/SuccessAnddropout2.0.git
cd SuccessAnddropout2.0
```

### 2. Create & activate a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention) or the [UCI ML Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) and place it at:
```
data/Predict Student Dropout and Academic Success.csv
```

### 5. Train the model
```bash
python models/train_academic_indicator_model.py
```
This creates `models/academic_indicator_model.pkl` and writes full metrics to `results/academic_indicator_metrics.txt`.

---

## Running the App

```bash
python app/app.py
```

Open your browser at **http://127.0.0.1:5000**

The form accepts:
- **Profile fields** — student name, department, roll number, semester
- **Academic indicators** — attendance, assignments, marks, study hours, and fees up to date (whether the student's tuition payments are current; 1 = Yes, 0 = No)

Results page shows dropout probability, success probability, risk level badge, and a summary of all entered indicators.

---

## Retraining the Model

Simply re-run the training script at any time:

```bash
python models/train_academic_indicator_model.py
```

The script will:
1. Engineer five academic indicators from the raw dataset, including the binary Fees Up To Date field.
2. Train and cross-validate 4 candidate models (LR, RF, GB, SVM).
3. Build and evaluate 5 calibrated / ensemble variants.
4. Auto-select the best production model.
5. Optimise the decision threshold for F1.
6. Serialise the full model bundle to `models/academic_indicator_model.pkl`.
7. Write a detailed metric report to `results/academic_indicator_metrics.txt`.

---

## Dataset

**Predict Students' Dropout and Academic Success**
- **Source:** [Kaggle](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention) / [UCI ML Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Records:** ~4 424 students across multiple Portuguese university programmes
- **Target classes:** Graduate / Enrolled / **Dropout**
- **Original features:** 36 demographic, socio-economic, and academic columns

> The raw CSV is excluded from version control (see `.gitignore`). Download it separately and place it in the `data/` folder.

---

## Tech Stack

| Layer | Library / Tool |
|-------|---------------|
| Web framework | Flask 3.x |
| ML — ensemble | scikit-learn `VotingClassifier` (GB + LR + RF) |
| ML — candidates | scikit-learn GBM, RF, LR, SVM (RBF) |
| Calibration | scikit-learn `CalibratedClassifierCV` |
| Data | pandas, numpy |
| Serialisation | joblib |
| Visualisation | matplotlib |
| Frontend | HTML5 · Vanilla CSS · JavaScript |

---

## License

This project is released under the [MIT License](LICENSE).
