"""
preprocessing.py
-----------------
Feature engineering + train/test split + imbalance handling utilities.

Key ideas used (interview-worthy talking points):
1. Stratified split -> preserves the tiny fraud ratio in both train & test.
2. Scaling only 'Time' and 'Amount' -> V1..V28 are already PCA components
   (mean 0, unit variance) in the real dataset, so scaling them again is
   unnecessary / can hurt interpretability of the PCA loadings.
3. Imbalance handling done ONLY on the training set (never on test set) to
   avoid data leakage - SMOTE oversamples the minority class synthetically.
4. class_weight='balanced' as an alternative to SMOTE for models like
   Logistic Regression, avoiding synthetic sample generation entirely.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def scale_features(X_train, X_test):
    """RobustScaler is used for 'Amount'/'Time' since fraud amounts often
    contain extreme outliers; robust scaling (median/IQR) is less sensitive
    to them than StandardScaler."""
    scaler = RobustScaler()
    cols_to_scale = [c for c in ["Time", "Amount"] if c in X_train.columns]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train_scaled, X_test_scaled, scaler


def stratified_split(X, y, test_size=0.25, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def apply_smote(X_train, y_train, sampling_strategy=0.1, random_state=42):
    """
    sampling_strategy=0.1 means: after resampling, minority class will be
    10% of the majority class count (NOT full 50/50 — going all the way to
    50/50 tends to overfit synthetic fraud patterns and hurts precision).
    This ratio is a tunable hyperparameter worth mentioning in interviews.
    """
    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


def prepare_data(df, use_smote=True, smote_ratio=0.1, test_size=0.25, random_state=42):
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size, random_state)
    X_train, X_test, scaler = scale_features(X_train, X_test)

    if use_smote:
        X_train_final, y_train_final = apply_smote(
            X_train, y_train, sampling_strategy=smote_ratio, random_state=random_state
        )
    else:
        X_train_final, y_train_final = X_train, y_train

    return {
        "X_train": X_train_final,
        "y_train": y_train_final,
        "X_train_raw": X_train,   # unresampled, needed for class_weight models
        "y_train_raw": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
    }
