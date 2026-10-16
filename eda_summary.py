"""
EDA Summary Statistics
Author: TejeswaraReddy-Dareddy
Date  : 2026-10-16
"""
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    return df

def summarise(df: pd.DataFrame) -> None:
    print("Shape        :", df.shape)
    print("Dtypes:\n",    df.dtypes)
    print("Nulls:\n",     df.isnull().sum())
    print("Describe:\n",  df.describe(include='all'))

if __name__ == '__main__':
    df = load_data('data/retail_sales.csv')
    summarise(df)
