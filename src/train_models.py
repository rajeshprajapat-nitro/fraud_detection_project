"""
train_models.py
-----------------
Trains three complementary models so the project demonstrates BOTH
unsupervised anomaly detection and supervised imbalance-aware classification
-- a strong signal for interviews:

1. Isolation Forest   -> unsupervised anomaly detector (doesn't need labels
                          to build the trees; labels only used to evaluate).
2. Logistic Regression-> supervised baseline with class_weight='balanced'.
3. XGBoost            -> supervised gradient boosting with scale_pos_weight,
                          typically the best performer on this task.

All three are evaluated on the SAME held-out test set so results are
directly comparable.
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


def train_isolation_forest(X_train_raw, y_train_raw, contamination=None, random_state=42):
    """
    Isolation Forest is unsupervised: we fit it on the RAW (non-SMOTE)
    training features only, and set `contamination` to the true fraud
    ratio observed in the training labels (a reasonable prior when the
    approximate fraud rate is known from historical data).
    """
    if contamination is None:
        contamination = max(y_train_raw.mean(), 1e-4)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_raw)
    return model


def isolation_forest_scores(model, X):
    """
    IsolationForest.decision_function returns higher = more normal.
    We flip sign so higher score = more anomalous = more "fraud-like",
    matching the convention of predict_proba's fraud column in other models.
    """
    return -model.decision_function(X)


def train_logistic_regression(X_train, y_train, random_state=42):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state=42):
    n_pos = max(y_train.sum(), 1)
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos  # counteracts class imbalance directly

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",   # optimize for PR-AUC directly during training
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, name, out_dir=None):
    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.joblib")
    joblib.dump(model, path)
    return path
