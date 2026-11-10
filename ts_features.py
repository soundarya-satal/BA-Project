"""
Time-Series Feature Engineering
Author : TejeswaraReddy-Dareddy
Date   : 2026-10-18
Updated: 2026-10-22
Description:
    Comprehensive feature engineering module for retail time-series data.
    Provides lag features, rolling window statistics, calendar features,
    Fourier terms for seasonality, and train/test split utilities.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# 1. Lag Features
# ---------------------------------------------------------------------------
def add_lag_features(df: pd.DataFrame, col: str,
                     lags: List[int]) -> pd.DataFrame:
    """Create lag columns shifted by given periods.

    Args:
        df  : Input DataFrame (sorted by date ascending).
        col : Target column to lag.
        lags: List of lag periods, e.g. [1, 7, 14, 28].

    Returns:
        DataFrame with new lag columns appended.
    """
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


# ---------------------------------------------------------------------------
# 2. Rolling Window Statistics
# ---------------------------------------------------------------------------
def add_rolling_features(df: pd.DataFrame, col: str,
                         windows: List[int],
                         stats: List[str] = None) -> pd.DataFrame:
    """Add rolling mean, std, min, max for given windows.

    Args:
        df     : Input DataFrame.
        col    : Target column.
        windows: List of window sizes, e.g. [7, 14, 30].
        stats  : Statistics to compute. Defaults to ['mean', 'std', 'min', 'max'].

    Returns:
        DataFrame with rolling feature columns appended.
    """
    if stats is None:
        stats = ["mean", "std", "min", "max"]
    df = df.copy()
    for w in windows:
        rolled = df[col].rolling(window=w, min_periods=1)
        for stat in stats:
            df[f"{col}_roll_{stat}_{w}"] = getattr(rolled, stat)()
    return df


# ---------------------------------------------------------------------------
# 3. Expanding (Cumulative) Features
# ---------------------------------------------------------------------------
def add_expanding_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add expanding (cumulative) mean and sum features."""
    df = df.copy()
    df[f"{col}_expand_mean"] = df[col].expanding().mean()
    df[f"{col}_expand_sum"]  = df[col].expanding().sum()
    return df


# ---------------------------------------------------------------------------
# 4. Calendar Features
# ---------------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame,
                           date_col: str = "date") -> pd.DataFrame:
    """Extract rich calendar-based features from a datetime column.

    Features added:
        day_of_week, day_of_month, day_of_year,
        week_of_year, month, quarter, year,
        is_weekend, is_month_start, is_month_end,
        is_quarter_start, is_quarter_end
    """
    df = df.copy()
    dt = df[date_col]
    df["day_of_week"]     = dt.dt.dayofweek
    df["day_of_month"]    = dt.dt.day
    df["day_of_year"]     = dt.dt.dayofyear
    df["week_of_year"]    = dt.dt.isocalendar().week.astype(int)
    df["month"]           = dt.dt.month
    df["quarter"]         = dt.dt.quarter
    df["year"]            = dt.dt.year
    df["is_weekend"]      = dt.dt.dayofweek.isin([5, 6]).astype(int)
    df["is_month_start"]  = dt.dt.is_month_start.astype(int)
    df["is_month_end"]    = dt.dt.is_month_end.astype(int)
    df["is_quarter_start"]= dt.dt.is_quarter_start.astype(int)
    df["is_quarter_end"]  = dt.dt.is_quarter_end.astype(int)
    return df


# ---------------------------------------------------------------------------
# 5. Fourier Terms (Seasonality Encoding)
# ---------------------------------------------------------------------------
def add_fourier_terms(df: pd.DataFrame, date_col: str = "date",
                      period: int = 365,
                      n_terms: int = 3) -> pd.DataFrame:
    """Add Fourier sine/cosine terms to model periodic seasonality.

    Args:
        df      : Input DataFrame.
        date_col: Name of datetime column.
        period  : Seasonality period in days (365 = annual, 7 = weekly).
        n_terms : Number of Fourier harmonics to include.

    Returns:
        DataFrame with 2*n_terms new columns.
    """
    df = df.copy()
    t = df[date_col].dt.dayofyear.values
    for k in range(1, n_terms + 1):
        df[f"fourier_sin_{period}_{k}"] = np.sin(2 * np.pi * k * t / period)
        df[f"fourier_cos_{period}_{k}"] = np.cos(2 * np.pi * k * t / period)
    return df


# ---------------------------------------------------------------------------
# 6. Train / Test Split
# ---------------------------------------------------------------------------
def time_series_split(df: pd.DataFrame,
                      date_col: str = "date",
                      test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train/test preserving temporal order.

    Args:
        df       : Input DataFrame sorted by date.
        date_col : Name of datetime column.
        test_size: Fraction of data to use as test set.

    Returns:
        (train_df, test_df) tuple.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    print(f"[time_series_split] Train: {len(train):,} rows | Test: {len(test):,} rows")
    return train, test


# ---------------------------------------------------------------------------
# 7. Full Feature Pipeline
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame,
                   target_col: str = "sales",
                   date_col: str   = "date",
                   lags: Optional[List[int]] = None,
                   windows: Optional[List[int]] = None) -> pd.DataFrame:
    """Run the complete feature engineering pipeline.

    Args:
        df        : Raw input DataFrame.
        target_col: Column to engineer features for.
        date_col  : Datetime column name.
        lags      : Lag periods. Defaults to [1, 7, 14, 28].
        windows   : Rolling windows. Defaults to [7, 14, 30].

    Returns:
        Feature-enriched DataFrame (NaN rows from lags still present).
    """
    if lags    is None: lags    = [1, 7, 14, 28]
    if windows is None: windows = [7, 14, 30]

    df = add_calendar_features(df, date_col)
    df = add_lag_features(df, target_col, lags)
    df = add_rolling_features(df, target_col, windows)
    df = add_expanding_features(df, target_col)
    df = add_fourier_terms(df, date_col, period=365, n_terms=3)
    df = add_fourier_terms(df, date_col, period=7,   n_terms=2)
    n_new = df.shape[1]
    print(f"[build_features] Feature matrix: {df.shape[0]:,} rows x {n_new} cols")
    return df
