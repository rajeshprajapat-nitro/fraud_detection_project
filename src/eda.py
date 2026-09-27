"""
eda.py
-------
Generates exploratory data analysis plots for the report/README:
1. Class distribution (log scale) - visualizes the extreme imbalance
2. Amount distribution: fraud vs legit (boxplot, log scale)
3. Time-of-transaction distribution: fraud vs legit
4. Correlation of V1..V28 with the target (top drivers)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def run_eda():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    # 1. Class distribution
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x="Class", data=df, hue="Class", legend=False, palette=["#4C72B0", "#C44E52"])
    ax.set_yscale("log")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.xticks([0, 1], ["Legit (0)", "Fraud (1)"])
    plt.ylabel("Count (log scale)")
    plt.title(f"Class Distribution — Fraud rate = {df['Class'].mean():.4%}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eda_class_distribution.png"), dpi=150)
    plt.close()

    # 2. Amount distribution: fraud vs legit
    plt.figure(figsize=(7, 5))
    plot_df = df.copy()
    plot_df["Amount_log1p"] = np.log1p(plot_df["Amount"])
    sns.boxplot(x="Class", y="Amount_log1p", data=plot_df, hue="Class", legend=False,
                palette=["#4C72B0", "#C44E52"])
    plt.xticks([0, 1], ["Legit", "Fraud"])
    plt.ylabel("log(1 + Amount)")
    plt.title("Transaction Amount: Fraud vs Legit")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eda_amount_distribution.png"), dpi=150)
    plt.close()

    # 3. Time-of-transaction pattern
    plt.figure(figsize=(9, 5))
    plot_df["Hour"] = (plot_df["Time"] % 86400) / 3600  # seconds -> hour of day
    sns.kdeplot(data=plot_df[plot_df["Class"] == 0], x="Hour", label="Legit", fill=True, alpha=0.3)
    sns.kdeplot(data=plot_df[plot_df["Class"] == 1], x="Hour", label="Fraud", fill=True, alpha=0.5, color="red")
    plt.xlabel("Hour of Day")
    plt.title("Transaction Timing Pattern: Fraud vs Legit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eda_time_pattern.png"), dpi=150)
    plt.close()

    # 4. Correlation of features with target
    corr = df.corr(numeric_only=True)["Class"].drop("Class").sort_values()
    plt.figure(figsize=(7, 8))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in corr.values]
    plt.barh(corr.index, corr.values, color=colors)
    plt.title("Feature Correlation with Fraud (Class)")
    plt.xlabel("Correlation coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eda_feature_correlation.png"), dpi=150)
    plt.close()

    top_pos = corr.sort_values(ascending=False).head(3)
    top_neg = corr.sort_values().head(3)

    print("EDA plots saved to", OUT_DIR)
    print("\nTop positively correlated features with fraud:\n", top_pos)
    print("\nTop negatively correlated features with fraud:\n", top_neg)


if __name__ == "__main__":
    run_eda()
