"""
main.py
--------
End-to-end pipeline entry point:
  1. Load data (real Kaggle dataset if present, else synthetic fallback)
  2. EDA summary (class balance, feature stats)
  3. Preprocess (scale, stratified split, SMOTE on train only)
  4. Train Isolation Forest, Logistic Regression, XGBoost
  5. Evaluate all three on the SAME held-out test set
  6. Save models, metrics.json, and PR-curve / confusion-matrix plots

Run:
    python src/main.py
"""

import os
import sys
import json

sys.path.append(os.path.dirname(__file__))

from data_utils import load_dataset
from preprocessing import prepare_data
from train_models import (
    train_isolation_forest,
    isolation_forest_scores,
    train_logistic_regression,
    train_xgboost,
    save_model,
)
from evaluate import full_report, plot_pr_curves, plot_confusion_matrix

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading dataset")
    print("=" * 60)
    df, source = load_dataset()
    fraud_rate = df["Class"].mean()
    print(f"Source: {source} | Rows: {len(df):,} | Fraud rate: {fraud_rate:.4%}")

    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing (scale + stratified split + SMOTE)")
    print("=" * 60)
    data = prepare_data(df, use_smote=True, smote_ratio=0.1)
    print(f"Train (post-SMOTE): {data['X_train'].shape}, "
          f"fraud ratio: {data['y_train'].mean():.4f}")
    print(f"Test (untouched):   {data['X_test'].shape}, "
          f"fraud ratio: {data['y_test'].mean():.4%}")

    print("\n" + "=" * 60)
    print("STEP 3: Training models")
    print("=" * 60)

    print("-> Isolation Forest (unsupervised)")
    iso_model = train_isolation_forest(data["X_train_raw"], data["y_train_raw"])

    print("-> Logistic Regression (class_weight=balanced)")
    lr_model = train_logistic_regression(data["X_train"], data["y_train"])

    print("-> XGBoost (scale_pos_weight)")
    xgb_model = train_xgboost(data["X_train"], data["y_train"])

    print("\n" + "=" * 60)
    print("STEP 4: Evaluating on held-out test set")
    print("=" * 60)

    y_test = data["y_test"]

    scores = {
        "IsolationForest": isolation_forest_scores(iso_model, data["X_test"]),
        "LogisticRegression": lr_model.predict_proba(data["X_test"])[:, 1],
        "XGBoost": xgb_model.predict_proba(data["X_test"])[:, 1],
    }

    reports = {}
    for name, y_scores in scores.items():
        rep = full_report(y_test, y_scores, model_name=name)
        reports[name] = rep
        print(f"\n--- {name} ---")
        print(f"PR-AUC: {rep['pr_auc']} | F1(fraud): {rep['f1_fraud']} | "
              f"Precision: {rep['precision_fraud']} | Recall: {rep['recall_fraud']}")
        print(rep["classification_report"])

        cm_path = os.path.join(OUT_DIR, f"confusion_matrix_{name}.png")
        plot_confusion_matrix(rep["confusion_matrix"], name, cm_path)

    pr_plot_path = os.path.join(OUT_DIR, "precision_recall_curves.png")
    plot_pr_curves(scores, y_test, pr_plot_path)
    print(f"\nSaved PR-curve comparison plot -> {pr_plot_path}")

    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {k: {kk: vv for kk, vv in v.items() if kk != "classification_report"}
             for k, v in reports.items()},
            f, indent=2
        )
    print(f"Saved metrics -> {metrics_path}")

    print("\n" + "=" * 60)
    print("STEP 5: Saving trained models")
    print("=" * 60)
    for name, model in [("isolation_forest", iso_model),
                         ("logistic_regression", lr_model),
                         ("xgboost", xgb_model)]:
        path = save_model(model, name)
        print(f"Saved {name} -> {path}")

    print("\nDone. Best model by PR-AUC:",
          max(reports.items(), key=lambda kv: kv[1]["pr_auc"])[0])


if __name__ == "__main__":
    main()
