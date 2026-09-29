# Credit Card Fraud Detection — Anomaly Detection under Extreme Class Imbalance

An end-to-end fraud detection system built on the classic ULB "Credit Card
Fraud Detection" dataset (284,807 transactions, only 492 fraud — **0.172%**
positive class). This project demonstrates how to detect rare, high-cost
events when accuracy is a meaningless metric.

## ⚡ Quick Start ( one command )

```bash
# 1. Unzip this project, cd into it
cd fraud_project

# 2. Make sure your dataset is at: data/creditcard.csv
#    (download from https://www.kaggle.com/mlg-ulb/creditcardfraud
#     if you don't have it — full steps in section 2 below)

# 3. Run everything (creates venv, installs deps, runs EDA + trains + evaluates
#    + opens a browser dashboard at http://localhost:8501)
bash run_all.sh
```
That's it. Results land in `outputs/` (plots + `metrics.json`), trained
models in `models/`, and a browser tab opens automatically showing an
interactive dashboard (EDA, model comparison, live prediction demo).

## 🌐 Browser Dashboard

This project includes a **Streamlit** dashboard (`app.py`) so you can view
everything in your browser instead of just the terminal — useful for demos
(e.g. showing recruiters/interviewers a live app instead of a script).

```bash
streamlit run app.py
```
This opens `http://localhost:8501` in your default browser automatically.
Pages available (sidebar navigation):
- **Overview** — dataset summary & project explanation
- **EDA** — class imbalance, amount/time patterns, feature correlation
- **Model Results** — PR-AUC/F1/Precision/Recall table, PR curves, confusion matrices
- **Live Prediction Demo** — draw a random real transaction and see the trained
  XGBoost model predict fraud probability in real time

Note: this needs `eda.py` and `main.py` to have been run at least once first
(so `outputs/metrics.json` and `models/xgboost.joblib` exist) — `run_all.sh`
does this automatically before launching the dashboard.

---

## 1. Problem Statement

Fraudulent transactions are extremely rare compared to legitimate ones.
A naive model that always predicts "not fraud" achieves >99.8% accuracy
while catching **zero** fraud — completely useless in production. This
project builds and compares three models that are specifically designed
to handle this imbalance, and evaluates them with metrics that actually
reflect fraud-catching performance: **Precision-Recall AUC** and
**F1-score on the minority (fraud) class**.

## 2. Dataset

| | |
|---|---|
| Source | [Kaggle — mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) |
| Rows | 284,807 transactions over 2 days |
| Features | `Time`, `Amount`, `V1`...`V28` (PCA-anonymized due to confidentiality), `Class` (target) |
| Fraud rate | 0.172% (492 of 284,807) |

### How to get the real dataset
```bash
# 1. Create a Kaggle account, then Account -> Create New API Token (downloads kaggle.json)
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 2. Download
python src/data_utils.py --download
# saves to data/creditcard.csv
```
If you run the pipeline without the real file and without Kaggle credentials,
`data_utils.py` automatically falls back to a **synthetic dataset** with the
same schema and imbalance ratio, so you can develop/demo the pipeline
immediately. **Swap in the real CSV before putting results on your resume.**

## 3. Project Structure
```
fraud_project/
├── data/                     # creditcard.csv goes here (gitignored)
├── models/                   # saved .joblib model files
├── outputs/                  # metrics.json, PR-curve + confusion matrix plots
├── src/
│   ├── data_utils.py         # dataset download / synthetic fallback
│   ├── preprocessing.py      # scaling, stratified split, SMOTE
│   ├── train_models.py       # Isolation Forest, Logistic Regression, XGBoost
│   ├── evaluate.py           # PR-AUC, F1, confusion matrix, threshold tuning
│   └── main.py                # orchestrates the full pipeline
├── requirements.txt
└── README.md
```

## 4. Setup & Run
```bash
python -m venv venv && source venv/bin/activate     # or venv\Scripts\activate on Windows
pip install -r requirements.txt

python src/eda.py     # generates EDA plots (class balance, amount, time, correlation)
python src/main.py    # trains + evaluates all 3 models
```
Outputs: trained models in `models/`, metrics in `outputs/metrics.json`,
EDA plots + PR-curve comparison + per-model confusion matrices as PNGs
in `outputs/`.

## 5. Methodology (the "advance" parts — talk about these in interviews)

### 5.1 Why Precision-Recall AUC, not ROC-AUC?
With 99.8% of transactions legitimate, ROC-AUC is misleadingly high because
the false-positive rate stays tiny even with many false positives relative
to the small number of frauds. **PR-AUC** (average precision) focuses only
on the positive class and doesn't get inflated by the huge true-negative
count — the correct metric when the minority class is what you actually
care about.

### 5.2 Imbalance handling — two complementary strategies
- **SMOTE (Synthetic Minority Oversampling)** — applied *only* to the
  training set (never the test set, to avoid leakage), and only to a
  **10% minority ratio** rather than a full 50/50 balance. Over-balancing
  tends to make the model overfit synthetic fraud patterns and hurts
  precision on real, held-out data.
- **`class_weight='balanced'` / `scale_pos_weight`** — used for Logistic
  Regression and XGBoost as an alternative that doesn't synthesize any
  data at all, penalizing misclassified minority samples more heavily
  during training.

### 5.3 Three models, two paradigms
| Model | Type | Why included |
|---|---|---|
| Isolation Forest | Unsupervised anomaly detection | Doesn't need fraud labels to build its trees — realistic for scenarios where labeled fraud is scarce or delayed |
| Logistic Regression | Supervised, interpretable | Fast, interpretable baseline; coefficients show which PCA components drive fraud risk |
| XGBoost | Supervised, gradient boosting | Typically the strongest performer; captures non-linear interactions between features |

### 5.4 Threshold tuning
Default 0.5 probability thresholds are almost never right for imbalanced
problems. `evaluate.py` sweeps the full precision-recall curve and picks
the threshold that maximizes F1 on the fraud class. In a real production
setting, you'd instead pick a threshold based on **business cost**: e.g.
if a missed fraud costs $500 and a false-positive customer-service call
costs $5, you'd tune the threshold to minimize `500*FN + 5*FP` rather than
plain F1 — mention this tradeoff explicitly in interviews, it shows
business awareness beyond just ML metrics.

### 5.5 Robust scaling
`Amount` is heavily right-skewed with extreme outliers (large legitimate
purchases). `RobustScaler` (median/IQR-based) is used instead of
`StandardScaler` (mean/std-based) so scaling isn't distorted by outliers.

## 6. Results (on the REAL ULB dataset — 284,807 transactions, 492 frauds)

Trained on a stratified 75/25 split (test set untouched by SMOTE, 71,202
transactions incl. 123 real frauds). Full numbers in `outputs/metrics.json`.

| Model | PR-AUC | F1 (fraud) | Precision (fraud) | Recall (fraud) |
|---|---|---|---|---|
| Isolation Forest (unsupervised) | 0.158 | 0.272 | 0.261 | 0.285 |
| Logistic Regression | 0.711 | 0.814 | 0.850 | 0.781 |
| **XGBoost** | **0.838** | **0.856** | **0.925** | **0.797** |

**Key findings:**
- XGBoost is the clear winner: catches **~80% of all fraud** in the test
  set while being right **92.5% of the time** when it flags a transaction
  as fraud — out of 71,079 legitimate transactions, only a handful get
  wrongly flagged.
- Isolation Forest, being fully unsupervised (never sees fraud labels
  during training), performs far worse — expected, but it's a valuable
  fallback for scenarios where labeled fraud data is scarce or delayed
  (e.g. a brand-new merchant category with no fraud history yet).
- Logistic Regression is a strong, fast, interpretable baseline — only
  ~4 points of PR-AUC behind XGBoost, useful when model explainability
  matters more than raw performance (e.g. regulatory / compliance settings).
- Top features driving fraud predictions by correlation: **V17, V14, V12**
  (negative correlation — lower values indicate higher fraud likelihood)
  and **V11, V4, V2** (positive correlation). See `outputs/eda_feature_correlation.png`.

## 7. Possible Extensions (great "future work" section for resume/GitHub)
- SHAP values for XGBoost to explain individual fraud predictions
- Cost-sensitive threshold optimization using an assumed $ cost matrix
- Real-time scoring simulation with a sliding time window (data is time-ordered)
- Autoencoder-based anomaly detection as a 4th model (deep learning angle)
- Deploy as a Flask/FastAPI endpoint + Streamlit dashboard for live demo

## 8. Resume Bullet Points (copy-paste ready)
- Built an end-to-end fraud detection pipeline on 284,807 real credit card
  transactions with extreme class imbalance (0.17% fraud), combining
  unsupervised (Isolation Forest) and supervised (Logistic Regression,
  XGBoost) anomaly detection approaches.
- Engineered an imbalance-handling strategy using targeted SMOTE
  oversampling (10% minority ratio) and cost-sensitive class weighting
  (scale_pos_weight / class_weight), avoiding overfitting to synthetic
  samples while correcting majority-class bias.
- Evaluated models using Precision-Recall AUC and minority-class F1-score
  instead of accuracy (which is >99.8% even for a useless model here);
  achieved **0.838 PR-AUC and 0.856 F1-score** on the fraud class with
  XGBoost — a **~5x improvement** over the no-skill baseline (0.0017 PR-AUC).
- Achieved **92.5% precision at 79.7% recall** on held-out fraud
  transactions, meaning the model catches 4 out of 5 frauds while keeping
  false alarms low enough for real-world deployment.
- Implemented business-aware threshold tuning by optimizing the
  precision/recall tradeoff on the PR curve instead of a default 0.5 cutoff.
- Conducted EDA identifying V17, V14, and V12 (PCA components) as the
  strongest fraud indicators via correlation analysis.

## 9. Tech Stack
Python, Pandas, NumPy, Scikit-Learn, imbalanced-learn (SMOTE), XGBoost,
Matplotlib, Seaborn, Joblib.
