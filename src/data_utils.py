"""
data_utils.py
--------------
Handles dataset acquisition for the Credit Card Fraud Detection project.

Real dataset: "Credit Card Fraud Detection" by ULB (Machine Learning Group)
Kaggle URL : https://www.kaggle.com/mlg-ulb/creditcardfraud
Schema     : Time, V1..V28 (PCA-anonymized features), Amount, Class (0=legit, 1=fraud)
Size       : 284,807 transactions, only 492 frauds (~0.172%)

This module tries to:
1. Use the Kaggle API to download the real dataset (requires kaggle.json creds).
2. If that's not available (e.g. no internet / no creds), it generates a
   SYNTHETIC dataset that mimics the real one's structure and imbalance ratio,
   so the rest of the pipeline (EDA, modeling, evaluation) can be built,
   tested and demoed end-to-end without the real file.

IMPORTANT: For your resume project, you should download the REAL dataset.
Steps:
    1. Create a Kaggle account -> https://www.kaggle.com
    2. Go to Account -> Create New API Token -> downloads kaggle.json
    3. Place kaggle.json at ~/.kaggle/kaggle.json  (chmod 600)
    4. Run: python src/data_utils.py --download
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH = os.path.join(DATA_DIR, "creditcard.csv")


def download_from_kaggle():
    """Downloads the real ULB credit card fraud dataset via Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        print("Kaggle package not configured:", e)
        return False

    try:
        api = KaggleApi()
        api.authenticate()
        os.makedirs(DATA_DIR, exist_ok=True)
        api.dataset_download_files(
            "mlg-ulb/creditcardfraud", path=DATA_DIR, unzip=True
        )
        print(f"Downloaded real dataset to {DATA_DIR}")
        return True
    except Exception as e:
        print("Kaggle download failed (no creds / no internet):", e)
        return False


def generate_synthetic_dataset(n_samples=150_000, fraud_ratio=0.0017, seed=42):
    """
    Generates a synthetic dataset that mimics the real ULB Credit Card
    Fraud dataset's SCHEMA and CLASS IMBALANCE, so the pipeline can be
    developed/tested without the actual file.

    Structure matches: Time, V1..V28, Amount, Class
    Fraud transactions are drawn from a shifted distribution so that
    the classes are separable but noisy (realistic anomaly-detection task).
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    n_features = 28  # V1..V28

    # Legitimate transactions: standard multivariate normal (PCA-like)
    legit_features = rng.normal(loc=0.0, scale=1.0, size=(n_legit, n_features))
    legit_amount = np.abs(rng.normal(loc=88, scale=120, size=n_legit))  # matches real dataset's mean amount ~88
    legit_time = rng.uniform(0, 172792, size=n_legit)  # real dataset spans ~2 days in seconds
    legit_class = np.zeros(n_legit, dtype=int)

    # Fraudulent transactions: shifted mean + higher variance on a subset
    # of "informative" components, mimicking real fraud patterns
    shift = rng.normal(loc=0.0, scale=2.5, size=n_features)
    fraud_features = rng.normal(loc=shift, scale=1.8, size=(n_fraud, n_features))
    fraud_amount = np.abs(rng.normal(loc=120, scale=250, size=n_fraud))
    fraud_time = rng.uniform(0, 172792, size=n_fraud)
    fraud_class = np.ones(n_fraud, dtype=int)

    features = np.vstack([legit_features, fraud_features])
    amount = np.concatenate([legit_amount, fraud_amount])
    time = np.concatenate([legit_time, fraud_time])
    label = np.concatenate([legit_class, fraud_class])

    cols = [f"V{i}" for i in range(1, n_features + 1)]
    df = pd.DataFrame(features, columns=cols)
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df["Class"] = label

    # shuffle rows
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def load_dataset(force_synthetic=False):
    """
    Main entry point used by the rest of the pipeline.
    Priority: real CSV on disk > kaggle download > synthetic fallback.
    """
    if not force_synthetic and os.path.exists(RAW_PATH):
        print(f"Loading real dataset from {RAW_PATH}")
        return pd.read_csv(RAW_PATH), "real"

    if not force_synthetic:
        ok = download_from_kaggle()
        if ok and os.path.exists(RAW_PATH):
            return pd.read_csv(RAW_PATH), "real"

    print("Using SYNTHETIC fallback dataset (for pipeline development/demo only).")
    df = generate_synthetic_dataset()
    os.makedirs(DATA_DIR, exist_ok=True)
    synth_path = os.path.join(DATA_DIR, "creditcard_synthetic.csv")
    df.to_csv(synth_path, index=False)
    return df, "synthetic"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Force download real dataset from Kaggle")
    parser.add_argument("--synthetic", action="store_true", help="Force generate synthetic dataset")
    args = parser.parse_args()

    if args.download:
        ok = download_from_kaggle()
        sys.exit(0 if ok else 1)
    else:
        df, source = load_dataset(force_synthetic=args.synthetic)
        print(f"Loaded {len(df)} rows from source={source}")
        print(df["Class"].value_counts(normalize=True))
