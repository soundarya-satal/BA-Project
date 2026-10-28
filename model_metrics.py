"""
Model Evaluation Metrics
Author : TejeswaraReddy-Dareddy
Date   : 2026-10-25
Updated: 2026-10-28
Description:
    Comprehensive evaluation module for regression and classification models
    used in BA-Project.  Covers standard regression metrics, cross-validation
    utilities, residual diagnostics, and formatted summary reporting.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore")

try:
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, r2_score,
        mean_absolute_percentage_error,
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, classification_report, confusion_matrix,
    )
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Regression Metrics
# ---------------------------------------------------------------------------
def evaluate_regression(y_true: Union[List, np.ndarray],
                        y_pred: Union[List, np.ndarray],
                        model_name: str = "Model") -> Dict:
    """Compute MAE, RMSE, MAPE, R2, and Max Error.

    Args:
        y_true     : Ground truth values.
        y_pred     : Predicted values.
        model_name : Label for display.

    Returns:
        Dictionary of metric name -> value.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae   = mean_absolute_error(y_true, y_pred)                         if SKLEARN_AVAILABLE else np.mean(np.abs(y_true - y_pred))
    mse   = mean_squared_error(y_true, y_pred)                          if SKLEARN_AVAILABLE else np.mean((y_true - y_pred) ** 2)
    rmse  = np.sqrt(mse)
    mape  = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-9, y_true))) * 100
    r2    = r2_score(y_true, y_pred)                                    if SKLEARN_AVAILABLE else float("nan")
    max_e = np.max(np.abs(y_true - y_pred))

    metrics = {
        "Model":     model_name,
        "MAE":       round(mae,   4),
        "RMSE":      round(rmse,  4),
        "MAPE (%)":  round(mape,  3),
        "R2":        round(r2,    4),
        "Max Error": round(max_e, 4),
    }
    return metrics


# ---------------------------------------------------------------------------
# 2. Compare Multiple Models
# ---------------------------------------------------------------------------
def compare_models(results: List[Dict]) -> pd.DataFrame:
    """Create a ranked comparison DataFrame from multiple evaluate_regression calls.

    Args:
        results: List of metric dicts returned by evaluate_regression().

    Returns:
        DataFrame sorted by RMSE ascending.
    """
    df = pd.DataFrame(results).set_index("Model")
    df = df.sort_values("RMSE")
    df["Rank"] = range(1, len(df) + 1)
    print("\n=== MODEL COMPARISON ===")
    print(df.to_string())
    return df


# ---------------------------------------------------------------------------
# 3. Residual Analysis
# ---------------------------------------------------------------------------
def residual_summary(y_true: np.ndarray,
                     y_pred: np.ndarray) -> Dict:
    """Summarise residuals: mean, std, skew, kurtosis, outlier count.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary of residual statistics.
    """
    residuals = np.array(y_true) - np.array(y_pred)
    q1, q3  = np.percentile(residuals, [25, 75])
    iqr     = q3 - q1
    n_out   = np.sum((residuals < q1 - 1.5 * iqr) | (residuals > q3 + 1.5 * iqr))

    summary = {
        "mean":       round(float(np.mean(residuals)), 4),
        "std":        round(float(np.std(residuals)),  4),
        "skewness":   round(float(pd.Series(residuals).skew()), 4),
        "kurtosis":   round(float(pd.Series(residuals).kurt()), 4),
        "outliers":   int(n_out),
        "pct_within_5pct": round(float(np.mean(np.abs(residuals / np.where(np.array(y_true) == 0, 1e-9, y_true)) < 0.05) * 100), 2),
    }
    print("\n=== RESIDUAL SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")
    return summary


# ---------------------------------------------------------------------------
# 4. Cross-Validation (Time-Series aware)
# ---------------------------------------------------------------------------
def cross_validate_ts(model, X: np.ndarray, y: np.ndarray,
                      n_splits: int = 5,
                      scoring: str = "neg_root_mean_squared_error") -> Dict:
    """Run time-series cross-validation using sklearn TimeSeriesSplit.

    Args:
        model   : Fitted or unfitted sklearn-compatible estimator.
        X       : Feature matrix.
        y       : Target vector.
        n_splits: Number of CV folds.
        scoring : Sklearn scoring string.

    Returns:
        Dict with mean, std, and per-fold scores.
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not available"}
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring)
    result = {
        "cv_mean":   round(float(np.mean(scores)), 4),
        "cv_std":    round(float(np.std(scores)),  4),
        "cv_scores": [round(s, 4) for s in scores],
        "n_splits":  n_splits,
        "scoring":   scoring,
    }
    print(f"\n=== TIME-SERIES CV ({n_splits} folds) ===")
    print(f"  Mean : {result['cv_mean']}  |  Std: {result['cv_std']}")
    return result


# ---------------------------------------------------------------------------
# 5. Pretty Print
# ---------------------------------------------------------------------------
def print_metrics(metrics: Dict, title: str = "Evaluation Results") -> None:
    """Pretty-print a metrics dictionary."""
    sep = "-" * 40
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    for k, v in metrics.items():
        print(f"  {str(k):20s}: {v}")
    print(sep)
