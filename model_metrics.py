"""
Model Evaluation Metrics
Author: TejeswaraReddy-Dareddy
Date  : 2026-10-25
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regression(y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100
    r2   = r2_score(y_true, y_pred)
    return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'MAPE': round(mape, 2), 'R2': round(r2, 4)}

def print_metrics(metrics: dict) -> None:
    for k, v in metrics.items():
        print(f"  {k:8s}: {v}")
