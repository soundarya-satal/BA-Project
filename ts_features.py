"""
Time-Series Feature Engineering
Author: TejeswaraReddy-Dareddy
Date  : 2026-10-18
"""
import pandas as pd

def add_lag_features(df: pd.DataFrame, col: str, lags: list) -> pd.DataFrame:
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, col: str, windows: list) -> pd.DataFrame:
    for w in windows:
        df[f'{col}_roll_mean_{w}'] = df[col].rolling(w).mean()
        df[f'{col}_roll_std_{w}']  = df[col].rolling(w).std()
    return df

def add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month']       = df[date_col].dt.month
    df['quarter']     = df[date_col].dt.quarter
    df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
    return df
