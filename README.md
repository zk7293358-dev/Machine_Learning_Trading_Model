Trading Machine Learning Model

A reproducible pipeline for researching, training, and backtesting ML models for algorithmic trading.
Supports data ingestion, feature engineering, model training, hyperparameter tuning, walk-forward backtests, and live inference hooks.

Features

Data ingestion: OHLCV + technical indicators from CSV/Parquet or APIs (plug-in interface).

Feature engineering: TA indicators, rolling stats, custom alpha factors.

Targets: Next-period return (regression) or up/down signal (classification).

Models: XGBoost / LightGBM / RandomForest / Logistic/Linear / (optional) LSTM/Transformer.

Backtesting: Vectorized, walk-forward, k-fold time splits, transaction costs & slippage.

Evaluation: Sharpe, Sortino, Max Drawdown, CAGR, Hit Rate, Precision/Recall/AUC (cls), MAE/MSE/R² (reg).

Experiment tracking: CSV logs; optional MLflow.

Deployment: Saved pipelines (.pkl) + simple REST/CLI inference.
