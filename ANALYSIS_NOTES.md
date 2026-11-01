# Analysis Notes

## Oct 12 2026
- Reviewed raw dataset for missing values
- Identified key date columns for time-series alignment

## Oct 14 2026
- Dropped duplicate rows and normalised column headers
- Applied forward-fill imputation for sparse numeric columns
- Converted date strings to datetime objects

## Oct 18 2026
- Engineered lag-7, lag-14, rolling-mean-7, rolling-std-7 features
- Added calendar features: day_of_week, month, quarter, is_weekend

## Oct 21 2026
- Ran TextBlob sentiment analysis on customer review corpus
- Labelled 62% positive, 23% neutral, 15% negative reviews

## Oct 25 2026
- XGBoost regressor: RMSE=142.3, MAPE=4.7%, R2=0.91
- SARIMA baseline: RMSE=198.6, MAPE=7.2%, R2=0.85
- XGBoost outperforms SARIMA on all metrics

## Oct 29 2026
- Pinned all dependency versions in requirements.txt
- Verified reproducibility on clean venv

## Nov 1 2026 - Key Findings
- Positive sentiment correlates (r=0.63) with higher weekly sales
- Lag-7 feature is the strongest predictor in XGBoost (feature importance rank 1)
- Weekend sales are on average 34% higher than weekday sales
- Q4 shows consistent uplift of ~18% -- recommend early inventory build
