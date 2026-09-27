"""
app.py
-------
A browser-based dashboard for the Fraud Detection project, built with
Streamlit. Run with:

    streamlit run app.py

This opens automatically in your default web browser (usually at
http://localhost:8501). It shows:
1. Dataset overview & EDA plots
2. Model comparison (PR-AUC, F1, Precision, Recall)
3. Precision-Recall curves & confusion matrices
4. A live "test a transaction" demo using the trained XGBoost model
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

BASE = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, "data", "creditcard.csv")
OUT_DIR = os.path.join(BASE, "outputs")
MODEL_DIR = os.path.join(BASE, "models")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide", page_icon="💳")

st.title("💳 Credit Card Fraud Detection Dashboard")
st.caption("Anomaly detection under extreme class imbalance — Isolation Forest, "
           "Logistic Regression & XGBoost compared on the ULB Kaggle dataset.")

# ---------------------------------------------------------------
# Sidebar: dataset status
# ---------------------------------------------------------------
st.sidebar.header("Status")
data_exists = os.path.exists(DATA_PATH)
metrics_exist = os.path.exists(os.path.join(OUT_DIR, "metrics.json"))

st.sidebar.write("Real dataset found: ", "✅" if data_exists else "❌ (place creditcard.csv in data/)")
st.sidebar.write("Trained models found: ", "✅" if metrics_exist else "❌ (run `python src/main.py` first)")

page = st.sidebar.radio("Navigate", ["Overview", "EDA", "Model Results", "Live Prediction Demo"])

# ---------------------------------------------------------------
# Overview
# ---------------------------------------------------------------
if page == "Overview":
    col1, col2, col3 = st.columns(3)
    if data_exists:
        df = pd.read_csv(DATA_PATH)
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Fraud Cases", f"{int(df['Class'].sum()):,}")
        col3.metric("Fraud Rate", f"{df['Class'].mean():.4%}")
    else:
        col1.metric("Total Transactions", "284,807 (expected)")
        col2.metric("Fraud Cases", "492 (expected)")
        col3.metric("Fraud Rate", "0.172% (expected)")

    st.markdown("""
    ### What this project does
    A fraud detector always faces the same core problem: **fraud is incredibly rare**
    (here, ~0.17% of all transactions). A model that predicts "not fraud" for
    everything scores 99.8% accuracy while catching zero fraud — so accuracy is
    thrown out entirely here in favor of **Precision-Recall AUC** and
    **F1-score on the fraud class**.

    Three models are trained and compared:
    - **Isolation Forest** — unsupervised anomaly detection (no fraud labels used during training)
    - **Logistic Regression** — supervised, interpretable, `class_weight='balanced'`
    - **XGBoost** — supervised gradient boosting, `scale_pos_weight` tuned for imbalance

    Use the sidebar to explore EDA, compare model results, or try a live prediction.
    """)

# ---------------------------------------------------------------
# EDA
# ---------------------------------------------------------------
elif page == "EDA":
    st.header("Exploratory Data Analysis")
    plots = [
        ("eda_class_distribution.png", "Class Distribution (log scale) — shows the extreme imbalance"),
        ("eda_amount_distribution.png", "Transaction Amount: Fraud vs Legit"),
        ("eda_time_pattern.png", "Transaction Timing Pattern (hour of day)"),
        ("eda_feature_correlation.png", "Feature Correlation with Fraud Label"),
    ]
    for fname, caption in plots:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            st.image(fpath, caption=caption, use_container_width=True)
        else:
            st.warning(f"Missing: {fname} — run `python src/eda.py` first.")

# ---------------------------------------------------------------
# Model Results
# ---------------------------------------------------------------
elif page == "Model Results":
    st.header("Model Comparison")

    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

        rows = []
        for name, m in metrics.items():
            rows.append({
                "Model": name,
                "PR-AUC": m["pr_auc"],
                "F1 (fraud)": m["f1_fraud"],
                "Precision (fraud)": m["precision_fraud"],
                "Recall (fraud)": m["recall_fraud"],
                "Threshold": m["threshold"],
            })
        results_df = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        best_model = results_df.iloc[0]["Model"]
        st.success(f"🏆 Best model by PR-AUC: **{best_model}**")

        st.bar_chart(results_df.set_index("Model")[["PR-AUC", "F1 (fraud)"]])
    else:
        st.warning("No results yet — run `python src/main.py` first.")

    st.subheader("Precision-Recall Curves")
    pr_path = os.path.join(OUT_DIR, "precision_recall_curves.png")
    if os.path.exists(pr_path):
        st.image(pr_path, use_container_width=True)

    st.subheader("Confusion Matrices")
    cols = st.columns(3)
    for i, name in enumerate(["IsolationForest", "LogisticRegression", "XGBoost"]):
        cm_path = os.path.join(OUT_DIR, f"confusion_matrix_{name}.png")
        if os.path.exists(cm_path):
            cols[i].image(cm_path, caption=name, use_container_width=True)

# ---------------------------------------------------------------
# Live Prediction Demo
# ---------------------------------------------------------------
elif page == "Live Prediction Demo":
    st.header("Live Prediction Demo (XGBoost)")
    st.write("Pick a real transaction from the test data (or a random one) "
             "and see what the trained model predicts.")

    model_path = os.path.join(MODEL_DIR, "xgboost.joblib")
    if not (os.path.exists(model_path) and data_exists):
        st.warning("Need both the trained model and the dataset. Run `python src/main.py` first.")
    else:
        model = joblib.load(model_path)
        df = pd.read_csv(DATA_PATH)

        mode = st.radio("Pick a sample", ["Random legit transaction", "Random known fraud transaction"])
        if st.button("Draw sample & Predict"):
            subset = df[df["Class"] == (1 if "fraud" in mode else 0)]
            row = subset.sample(1)
            X_row = row.drop(columns=["Class"])
            true_label = int(row["Class"].values[0])

            proba = model.predict_proba(X_row)[0, 1]
            pred_label = int(proba >= 0.5)

            c1, c2, c3 = st.columns(3)
            c1.metric("True Label", "FRAUD" if true_label else "Legit")
            c2.metric("Predicted Fraud Probability", f"{proba:.4f}")
            c3.metric("Predicted Label", "FRAUD" if pred_label else "Legit")

            if pred_label == true_label:
                st.success("✅ Correct prediction")
            else:
                st.error("❌ Incorrect prediction")

            with st.expander("Show raw transaction features"):
                st.dataframe(X_row.T.rename(columns={X_row.index[0]: "value"}))
