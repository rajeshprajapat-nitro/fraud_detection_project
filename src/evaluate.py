"""
evaluate.py
------------
Evaluation utilities focused on metrics that ACTUALLY matter for extreme
imbalance (<0.2% fraud). Accuracy is intentionally NEVER reported as a
headline metric here -- a model predicting "not fraud" for everything
would score >99.8% accuracy while being useless. That's the whole point
of this project and a great line for your resume/interview.

Metrics used:
- Precision-Recall AUC (average_precision_score): the primary metric.
  Unlike ROC-AUC, PR-AUC is not inflated by the huge number of true
  negatives, so it reflects performance on the minority (fraud) class
  honestly.
- F1-score on the minority class specifically.
- Confusion matrix (to show business-readable false positives/negatives).
- Precision-Recall curve plot, to justify the chosen decision threshold.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def evaluate_scores(y_true, y_scores, model_name="model"):
    """
    y_scores: continuous anomaly score / predicted probability of fraud
    (not hard 0/1 labels) -- required for PR-AUC.
    """
    pr_auc = average_precision_score(y_true, y_scores)
    return {"model": model_name, "pr_auc": pr_auc}


def best_threshold_by_f1(y_true, y_scores):
    """
    Sweeps thresholds along the PR curve and returns the threshold that
    maximizes F1 on the minority (fraud) class. In production you'd
    instead pick a threshold based on business cost of false positives
    vs false negatives (see README), but max-F1 is a solid default for
    a portfolio project.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = np.argmax(f1s[:-1])  # last point has no corresponding threshold
    return thresholds[best_idx], f1s[best_idx]


def full_report(y_true, y_scores, model_name="model", threshold=None):
    if threshold is None:
        threshold, _ = best_threshold_by_f1(y_true, y_scores)

    y_pred = (y_scores >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_scores)
    f1_fraud = f1_score(y_true, y_pred, pos_label=1)
    precision_fraud = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_fraud = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    report = {
        "model": model_name,
        "threshold": round(float(threshold), 4),
        "pr_auc": round(float(pr_auc), 4),
        "f1_fraud": round(float(f1_fraud), 4),
        "precision_fraud": round(float(precision_fraud), 4),
        "recall_fraud": round(float(recall_fraud), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, digits=4),
    }
    return report


def plot_pr_curves(results: dict, y_true, out_path):
    """
    results: dict of {model_name: y_scores}
    Draws all models' PR curves on one plot for direct comparison.
    """
    plt.figure(figsize=(7, 6))
    for name, y_scores in results.items():
        precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        plt.plot(recalls, precisions, label=f"{name} (AP={ap:.3f})")

    baseline = y_true.mean()
    plt.hlines(baseline, 0, 1, colors="gray", linestyles="--",
               label=f"No-skill baseline ({baseline:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — Fraud Detection Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, model_name, out_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
