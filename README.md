# Stock Market Risk and Investment Quality Forecasting

A graduate-level machine learning and deep learning project for short-horizon stock market forecasting with a focus on **risk-aware investment assessment**, not naive price prediction.

This project builds a full end-to-end pipeline for:
- downloading market data,
- constructing a panel dataset,
- engineering technical and market-context features,
- defining forward-looking return and risk targets,
- building binary investment-quality labels,
- comparing classical ML models with sequence-based deep learning models.

The central question is:

> Can neural sequence models outperform strong tabular baselines in forecasting short-horizon market risk and identifying favorable investment opportunities?

---

## Project Objective

Most weak stock-prediction projects try to forecast whether price goes up or down. That is not enough for serious financial modeling, because investment decisions depend on both **return** and **risk**.

This project was designed to address that.

### Main objectives
1. Build a multi-asset panel dataset from liquid U.S. ETFs and large-cap equities.
2. Engineer return, volatility, technical, market-state, and calendar features.
3. Define forward-looking targets for:
   - future return,
   - future realized volatility,
   - future downside semivolatility,
   - a Sharpe-like future score,
   - a binary investment-quality label.
4. Compare:
   - Ridge Regression,
   - Logistic Regression,
   - Random Forest,
   - Single-Task LSTM with attention,
   - Multi-Task LSTM with attention.
5. Evaluate whether deep sequence models add predictive value beyond strong engineered tabular features.

---

## Why this project matters

This is not just a stock-prediction notebook. It is a structured computational framework for **forecasting under uncertainty**.

### Possible applications
The methodology used here can be extended to:
- equity screening,
- portfolio construction,
- short-horizon risk management,
- volatility targeting,
- tactical allocation,
- algorithmic trading research,
- energy-market forecasting,
- insurance risk modeling,
- supply-chain opportunity/risk scoring,
- macro-financial stress monitoring.

The general idea is transferable anywhere decisions depend on both expected performance and uncertainty.

---

## Asset Universe

The project uses 24 liquid U.S. assets:

### Broad-market ETFs
- SPY
- QQQ
- IWM

### Sector ETFs
- XLK
- XLF
- XLV
- XLE
- XLI
- XLP
- XLU
- XLY
- XLB
- XLRE

### Large-cap equities
- AAPL
- MSFT
- NVDA
- AMZN
- GOOGL
- META
- JPM
- XOM
- JNJ
- UNH
- PG

### Time span
- Start: `2012-01-01`
- End: `2026-01-01`

---

## What has been done

### 1. Environment setup
A clean Python environment was created after resolving package conflicts involving:
- Python version compatibility,
- PyTorch installation,
- NumPy binary compatibility,
- pandas import issues,
- XGBoost/OpenMP dependency issues.

Final stable environment used:
- Python 3.11
- NumPy 1.26.4
- pandas 2.2.3
- torch 2.2.2

### 2. Data download
Historical daily data were downloaded using `yfinance`.

### 3. Panel dataset construction
Data were reshaped into a long panel format with one row per:
- asset,
- date.

### 4. Feature engineering
The following categories of features were created:

#### Price and return features
- log close
- 1-day return
- simple return
- rolling return means
- rolling return standard deviations
- momentum over multiple windows
- high-low range
- open-close return

#### Volume features
- log volume
- volume change
- rolling volume average

#### Technical indicators
- RSI(14)
- MACD
- MACD signal
- MACD difference
- Bollinger Band width
- ATR(14)

#### Market context features
- SPY return
- SPY rolling volatility
- SPY momentum
- VIX level
- VIX return

#### Calendar features
- day of week
- month
- quarter
- cyclical encodings of time variables

### 5. Forward-looking target construction
We created multi-horizon targets for:
- `future_ret_h`
- `future_rv_h`
- `future_dsv_h`
- `future_score_h`

for horizons:
- 1 day
- 5 days
- 21 days

### 6. Binary investment-quality label
For each horizon, an asset is labeled as `1` if:
- future return is positive, and
- future realized volatility is below the cross-sectional median on that date.

Otherwise it is labeled `0`.

This makes the classification task risk-aware.

### 7. Time-based split
A proper chronological split was used:
- **Train:** through 2020-12-31
- **Validation:** 2021-01-04 to 2023-12-29
- **Test:** 2024-01-02 to 2025-12-30

This avoids temporal leakage.

### 8. Baseline models
The following classical baselines were trained:
- Ridge Regression for `future_rv_5`
- Logistic Regression for `quality_label_5`
- Random Forest Regressor for `future_rv_5`
- Random Forest Classifier for `quality_label_5`

### 9. Deep learning models
The following deep models were implemented:
- Single-Task LSTM with attention for `future_rv_5`
- Original Multi-Task LSTM with attention for:
  - `future_rv_5`
  - `future_dsv_5`
  - `future_score_5`
  - `quality_label_5`
- Fixed Multi-Task LSTM with:
  - standardized regression targets,
  - smaller hidden dimension,
  - stronger regularization,
  - early stopping.

---

## Project structure

A typical project layout is:

```text
stock-market-risk-forecasting/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── model_panel_dataset.csv
│
├── notebooks/
│   ├── data_download_and_features.ipynb
│   ├── baselines.ipynb
│   ├── single_task_lstm.ipynb
│   └── multitask_lstm.ipynb
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── logs/
│
├── report/
│   ├── main.tex
│   └── figures/
│
├── README.md
└── requirements.txt
