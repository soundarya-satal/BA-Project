"""
EDA Summary Statistics
Author : TejeswaraReddy-Dareddy
Date   : 2026-10-16
Updated: 2026-10-19
Description:
    Full exploratory data analysis pipeline for the BA-Project retail dataset.
    Covers data loading, cleaning, univariate/bivariate analysis,
    missing-value heatmaps, outlier detection, and correlation reporting.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="muted")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = ["sales", "quantity", "discount", "profit"]
DATE_COL     = "date"
CATEGORY_COL = "category"


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------
def load_data(path: str = "data/retail_sales.csv") -> pd.DataFrame:
    """Load the retail sales CSV into a DataFrame.

    Args:
        path: Relative or absolute path to the CSV file.

    Returns:
        Parsed DataFrame with date column converted to datetime.
    """
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    print(f"[load_data] Loaded {len(df):,} rows x {df.shape[1]} cols from '{path}'")
    return df


# ---------------------------------------------------------------------------
# 2. Basic Info
# ---------------------------------------------------------------------------
def print_basic_info(df: pd.DataFrame) -> None:
    """Print shape, dtypes, null counts, and head."""
    print("\n=== SHAPE ===")
    print(f"  Rows: {df.shape[0]:,}  |  Cols: {df.shape[1]}")

    print("\n=== DTYPES ===")
    print(df.dtypes.to_string())

    print("\n=== NULL COUNTS ===")
    nulls = df.isnull().sum()
    null_pct = (nulls / len(df) * 100).round(2)
    null_df = pd.DataFrame({"count": nulls, "pct": null_pct})
    print(null_df[null_df["count"] > 0].to_string() or "  No missing values.")

    print("\n=== FIRST 5 ROWS ===")
    print(df.head().to_string())


# ---------------------------------------------------------------------------
# 3. Univariate Analysis
# ---------------------------------------------------------------------------
def univariate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    desc = df[NUMERIC_COLS].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    desc.loc["skew"]  = df[NUMERIC_COLS].skew()
    desc.loc["kurt"]  = df[NUMERIC_COLS].kurt()
    print("\n=== UNIVARIATE STATS ===")
    print(desc.round(3).to_string())
    return desc


def plot_distributions(df: pd.DataFrame) -> None:
    """Plot histograms + KDE for all numeric columns."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        df[col].dropna().plot(kind="hist", bins=40, ax=axes[i],
                              color="#5C6BC0", edgecolor="white", density=True)
        df[col].dropna().plot(kind="kde", ax=axes[i], color="#EF5350", lw=2)
        axes[i].set_title(f"Distribution of {col}", fontsize=12)
        axes[i].set_xlabel(col)
    plt.tight_layout()
    out = OUTPUT_DIR / "distributions.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot_distributions] Saved -> {out}")


# ---------------------------------------------------------------------------
# 4. Bivariate / Correlation Analysis
# ---------------------------------------------------------------------------
def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and plot Pearson correlation matrix."""
    corr = df[NUMERIC_COLS].corr()
    print("\n=== CORRELATION MATRIX ===")
    print(corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Pearson Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    out = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[correlation_analysis] Saved -> {out}")
    return corr


# ---------------------------------------------------------------------------
# 5. Outlier Detection (IQR method)
# ---------------------------------------------------------------------------
def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag outliers using 1.5 * IQR rule for numeric columns."""
    report = {}
    for col in NUMERIC_COLS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr     = q3 - q1
        lower   = q1 - 1.5 * iqr
        upper   = q3 + 1.5 * iqr
        n_out   = ((df[col] < lower) | (df[col] > upper)).sum()
        report[col] = {"q1": round(q1, 2), "q3": round(q3, 2),
                       "iqr": round(iqr, 2), "lower": round(lower, 2),
                       "upper": round(upper, 2), "outliers": int(n_out)}
    report_df = pd.DataFrame(report).T
    print("\n=== OUTLIER REPORT ===")
    print(report_df.to_string())
    return report_df


# ---------------------------------------------------------------------------
# 6. Category-Level Aggregation
# ---------------------------------------------------------------------------
def category_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sales/profit by category."""
    if CATEGORY_COL not in df.columns:
        print("[category_analysis] No 'category' column found, skipping.")
        return pd.DataFrame()
    agg = (df.groupby(CATEGORY_COL)[NUMERIC_COLS]
             .agg(["mean", "sum", "count"])
             .round(2))
    print("\n=== CATEGORY AGGREGATION ===")
    print(agg.to_string())
    return agg


# ---------------------------------------------------------------------------
# 7. Main Pipeline
# ---------------------------------------------------------------------------
def run_eda(path: str = "data/retail_sales.csv") -> None:
    """Run the full EDA pipeline end-to-end."""
    df = load_data(path)
    print_basic_info(df)
    univariate_analysis(df)
    plot_distributions(df)
    correlation_analysis(df)
    detect_outliers(df)
    category_analysis(df)
    print("\n[run_eda] EDA complete. Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    run_eda()
