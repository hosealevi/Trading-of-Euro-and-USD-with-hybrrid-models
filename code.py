
"""
Title : Compare multiple Machine Learning Architectures for EUR/USD Swing Trading (Capstone  690 Data Analysis)

Authors: 
- Rakotoarimanana Nomenjanahary Levi Hosea
- Qihu Zhang

Part of the program Master of Science in Financial Engineering at WorldQuant University. 

"""

!pip install hmmlearn
!pip install pgmpy
!pip install yfinance

# Core
import os
import datetime as dt

# Data handling
import pandas as pd
import numpy as np

# Data sources
import yfinance as yf
import pandas_datareader.data as web

# Plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches

# ML preprocessing
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

# Probabilistic models
from hmmlearn.hmm import GaussianHMM
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Deep learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Config
# ---------------------------
start = dt.datetime(2018, 1, 1)
end = dt.datetime.today()

"""# Chapter 1: Data Loading,  Preprocessing (Minute-level),Exploratory data analysis, ARIMA and GARCH

## 1.1. Loading and preprocessing data
"""

# --- 1.1 Load EUR/USD minute data ---
# Columns in CSV: datetime, open, high, low, close, volume
eurusd = pd.read_csv("eurusd.csv", parse_dates=['datetime'])

# Rename columns to match expected names
eurusd.rename(columns={
    'datetime': 'DateTime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

eurusd.set_index('DateTime', inplace=True)
eurusd = eurusd.sort_index()

# --- 1.2 Load daily macroeconomic indicators ---
# Handles macro CSV with date as index
macro = pd.read_csv("macro_data_daily.csv", index_col=0)
macro.index = pd.to_datetime(macro.index)  # ensure datetime index
macro = macro.sort_index()

# --- 1.3 Forward-fill macro indicators to match minute data ---
# Each minute inherits the latest available macro value
macro_minute = macro.reindex(eurusd.index, method='ffill')
data = eurusd.join(macro_minute, how='inner')

# --- 1.4 Basic sanity checks ---
print("Missing values per column:\n", data.isna().sum())
data = data.dropna()
print("Data shape after cleaning:", data.shape)

# --- 1.5 Feature engineering: log returns and rolling volatility ---
data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
# For minute data, volatility over last 60 minutes (~1 hour)
data['volatility'] = data['log_ret'].rolling(window=60).std()

if "GVZCLS" in data.columns:
    data = data.rename(columns={"GVZCLS": "Gold_Price_USD"})
else:
    print("GVZCLS column missing — nothing renamed.")

# --- 1.6 Standardization ---
scaler = StandardScaler()

# Adjust feature names to match your macro_data_daily.csv columns
# Example: 'US_GDP', 'Median_CPI', '10Y_Treasury_FF_Spread'
features_to_scale = ['log_ret', 'volatility', 'US_GDP', 'Median_CPI', '10Y_Treasury_FF_Spread']
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# --- 1.7 Preview ---
print(data.tail())

"""## 1.2. Exploratory data analysis (EDA)

### 1.2.1. Data structure
"""

if not os.path.exists("eda_outputs"):
    os.makedirs("eda_outputs")

# ----------------------------------
# 1) DATA STRUCTURE
# ----------------------------------
data_structure = pd.DataFrame({
    "n_rows": [len(data)],
    "n_cols": [len(data.columns)],
    "start_date": [data.index.min()],
    "end_date": [data.index.max()],
    "index_type": [type(data.index)]
})

print("\n--- DATA STRUCTURE ---")
print(data_structure)

# Export table
data_structure.to_csv("eda_outputs/eda_structure.csv", index=False)


# ----------------------------------
# 2) MISSING DATA ANALYSIS
# ----------------------------------
missing_table = data.isna().sum().to_frame("missing_count")
missing_table["missing_percent"] = (
    missing_table["missing_count"] / len(data) * 100
).round(4)

print("\n--- MISSING DATA ---")
print(missing_table)

# Export missing table
missing_table.to_csv("eda_outputs/eda_missing.csv")

# Plot missing values
plt.figure(figsize=(12,4))
missing_table["missing_count"].plot(kind="bar")
plt.title("Missing Values Per Column")
plt.tight_layout()

# Save and show
plt.savefig("eda_outputs/missing_values.png")
plt.show()


# ----------------------------------
# 3) SUMMARY STATISTICS
# ----------------------------------
summary_stats = data.describe().T

print("\n--- SUMMARY STATISTICS ---")
print(summary_stats)

# Export summary stats
summary_stats.to_csv("eda_outputs/eda_summary_stats.csv")

# Histogram of Close prices
plt.figure(figsize=(10,4))
plt.hist(data["Close"], bins=50)
plt.title("Distribution of Close Prices")
plt.xlabel("Close")
plt.tight_layout()
plt.savefig("eda_outputs/distribution_close_prices.png")
plt.show()

# Histogram of log returns — only if exists
if "log_ret" in data.columns:
    plt.figure(figsize=(10,4))
    plt.hist(data["log_ret"].dropna(), bins=50)
    plt.title("Distribution of Log Returns")
    plt.xlabel("log_ret")
    plt.tight_layout()
    plt.savefig("eda_outputs/distribution_log_returns.png")
    plt.show()

"""### 1.2.2. Time-series analysis"""

# Make folder
if not os.path.exists("ts_outputs"):
    os.makedirs("ts_outputs")

# =========================================
# FAST STATIONARITY CHECK
# =========================================

# Price
plt.figure(figsize=(12,5))
plt.plot(data["Close"], label="Close")
plt.plot(data["Close"].rolling(500).mean(), label="Rolling Mean (500)", linewidth=2)
plt.plot(data["Close"].rolling(500).std(), label="Rolling Std (500)", linewidth=2)
plt.title("Close Price – Rolling Mean & Std (Stationarity Check)")
plt.legend()
plt.tight_layout()
plt.savefig("ts_outputs/rolling_stationarity_check_close.png")
plt.show()

# Returns
plt.figure(figsize=(12,5))
plt.plot(data["log_ret"], label="Log Returns", linewidth=0.7)
plt.plot(data["log_ret"].rolling(500).std(), label="Rolling Volatility (500)", linewidth=2)
plt.title("Log Returns – Rolling Volatility (Stationarity Check)")
plt.legend()
plt.tight_layout()
plt.savefig("ts_outputs/rolling_stationarity_check_returns.png")
plt.show()

# ---------------------------------------------------
# 3) RETURN DYNAMICS ANALYSIS
# ---------------------------------------------------

plt.figure(figsize=(12,4))
plt.plot(data["volatility"], linewidth=0.8)
plt.title("Rolling Volatility (Existing Feature)")
plt.tight_layout()
plt.savefig("ts_outputs/rolling_volatility.png")
plt.show()

# ---------------------------------------------------
# 4) AUTOCORRELATION / PACF (Returns)
# ---------------------------------------------------
# 1) Resample log returns to daily mean (or sum — both are used in FX research)
daily_ret = data["log_ret"].resample("1D").mean().dropna()

# 2) ACF (fast + meaningful)
plt.figure(figsize=(10,4))
plot_acf(daily_ret, lags=20)
plt.title("ACF – Daily Log Returns")
plt.tight_layout()
plt.savefig("ts_outputs/acf_daily_returns.png")
plt.show()

# 3) PACF (much lighter than minute data)
plt.figure(figsize=(10,4))
plot_pacf(daily_ret, lags=20, method="ywm")
plt.title("PACF – Daily Log Returns")
plt.tight_layout()
plt.savefig("ts_outputs/pacf_daily_returns.png")
plt.show()

#5) CHECK FOR ARCH EFFECTS (Volatility Clustering)

# 1) Daily returns (you can reuse if already defined)
daily_ret = data["log_ret"].resample("1D").mean().dropna()

# 2) Squared daily returns
sq_daily = daily_ret ** 2

# Plot squared daily returns (volatility clustering)
plt.figure(figsize=(10,4))
plt.plot(sq_daily, linewidth=0.8)
plt.title("Squared Daily Returns (Volatility Clustering Check)")
plt.tight_layout()
plt.savefig("ts_outputs/squared_daily_returns.png")
plt.show()

# 3) Very fast custom ACF for squared returns
def quick_acf(series, max_lag=20):
    x = series.dropna().values
    x = x - x.mean()
    acf_vals = [1.0]  # lag 0
    for lag in range(1, max_lag + 1):
        corr = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        acf_vals.append(corr)
    return acf_vals

acf_sq = quick_acf(sq_daily, max_lag=20)

# 4) Plot ACF for squared returns (lags 1–20 only)
lags = range(1, 21)

plt.figure(figsize=(10,4))
plt.stem(lags, acf_sq[1:])
plt.axhline(0, linewidth=0.8, color="black")
plt.title("ACF – Squared Daily Returns (ARCH Check, 20 Lags)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.tight_layout()
plt.savefig("ts_outputs/acf_squared_daily_returns.png")
plt.show()

# ---------------------------------------------------
# 6-) MACRO–FX Relationships Over Time
# ---------------------------------------------------

macro_cols = [
    'Federal_Funds_Rate', '10Y_Treasury_FF_Spread', 'US_Unemployment_Rate', 'M2',
    'US_GDP', 'Euro_HICP_All_Items', 'Germany_10Y_Bond_Yield',
    'Short_Term_EUR_Interest', 'Euro_Unemployment_Rate',
    'Euro_GDP', 'Median_CPI', 'Gold_Price_USD'
]

for col in macro_cols:
    if col in data.columns:
        plt.figure(figsize=(12,3))
        plt.plot(data[col], linewidth=0.8)
        plt.title(f"{col} Over Time")
        plt.tight_layout()
        plt.savefig(f"ts_outputs/macro_{col}.png")
        plt.show()

# Rolling correlations: macro vs returns
roll_corr = {}

for col in macro_cols:
    if col in data.columns:
        roll_corr[col] = data["log_ret"].rolling(2000).corr(data[col])

roll_corr_df = pd.DataFrame(roll_corr)
roll_corr_df.to_csv("ts_outputs/rolling_macro_fx_correlations.csv")

plt.figure(figsize=(12,6))
plt.plot(roll_corr_df)
plt.legend(roll_corr_df.columns, fontsize=6)
plt.title("Rolling Correlations (Log Returns vs Macro Variables)")
plt.tight_layout()
plt.savefig("ts_outputs/rolling_correlations.png")
plt.show()

"""### 1.2.3. Corrrelation analysis"""

# ===========================================
#  MACRO CORRELATION + MULTICOLLINEARITY EDA
# ===========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -------------------------------------------
# 0) OUTPUT FOLDER
# -------------------------------------------
if not os.path.exists("eda_outputs"):
    os.makedirs("eda_outputs")

# -------------------------------------------
# 1) DETECT MACRO COLUMNS
# -------------------------------------------
fx_cols = ["Open", "High", "Low", "Close", "Volume", "log_ret", "volatility"]

macro_cols = [c for c in data.columns if c not in fx_cols]

print("Macro variables detected:")
print(macro_cols)

macro_df = data[macro_cols].dropna()

# -------------------------------------------
# 2) CORRELATION MATRIX (MACRO ONLY)
# -------------------------------------------
corr_matrix = macro_df.corr()

print("\n--- Macro Correlation Matrix (head) ---")
print(corr_matrix.head())

# Save full matrix
corr_matrix.to_csv("eda_outputs/macro_correlation_matrix.csv")

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, center=0)
plt.title("Macro Variables Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_outputs/macro_correlation_heatmap.png")
plt.show()

# -------------------------------------------
# 3) TOP CORRELATED PAIRS (to spot multicollinearity)
# -------------------------------------------
corr_pairs = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
corr_pairs.columns = ["Var1", "Var2", "Correlation"]
corr_pairs["AbsCorr"] = corr_pairs["Correlation"].abs()

corr_top20 = corr_pairs.sort_values("AbsCorr", ascending=False).head(20)

print("\n--- Top 20 strongest macro correlations ---")
print(corr_top20)

corr_top20.to_csv("eda_outputs/macro_corr_top20.csv", index=False)

# -------------------------------------------
# 4) VARIANCE INFLATION FACTOR (VIF)
# -------------------------------------------
# Fill any remaining NaNs for VIF computation
macro_clean = macro_df.fillna(method="ffill").fillna(method="bfill")

X = macro_clean.values
vif_data = []

for i, col in enumerate(macro_cols):
    vif = variance_inflation_factor(X, i)
    vif_data.append([col, vif])

vif_df = pd.DataFrame(vif_data, columns=["Variable", "VIF"])

print("\n--- VIF Table (Multicollinearity Check) ---")
print(vif_df)

vif_df.to_csv("eda_outputs/macro_vif.csv", index=False)

# -------------------------------------------
# 5) OPTIONAL: SIMPLE RULE-OF-THUMB INTERPRETATION
# -------------------------------------------
print("\nInterpretation guide:")
print("- |Correlation| > 0.8  -> strong pairwise association (risk of redundancy).")
print("- VIF > 10            -> serious multicollinearity.")
print("- VIF between 5 and 10 -> moderate multicollinearity; watch carefully.")

"""##1.3. ARIMA model"""

# ==========================================
#        ARIMA MODEL ON DAILY RETURNS
# ==========================================

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------------------------------------------------
# 0) OUTPUT FOLDER
# -------------------------------------------------------------------
if not os.path.exists("model_outputs"):
    os.makedirs("model_outputs")

# -------------------------------------------------------------------
# 1) BUILD DAILY LOG RETURNS SERIES
# -------------------------------------------------------------------
# Use last close of each day, then log-return
daily_close = data["Close"].resample("1D").last().dropna()
daily_ret = np.log(daily_close / daily_close.shift(1)).dropna()

y = daily_ret.copy()
y.name = "daily_log_ret"

print(f"Number of daily observations: {len(y)}")
print(f"From {y.index.min()} to {y.index.max()}")

# -------------------------------------------------------------------
# 2) TRAIN / TEST SPLIT (80 / 20)
# -------------------------------------------------------------------
split_idx = int(len(y) * 0.8)
y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

print("\nTrain length:", len(y_train), "Test length:", len(y_test))

# -------------------------------------------------------------------
# 3) SMALL GRID SEARCH FOR (p,d,q) USING AIC
# -------------------------------------------------------------------
candidate_orders = [
    (0, 0, 1),
    (1, 0, 0),
    (1, 0, 1),
    (2, 0, 1),
    (1, 1, 1),
]

search_results = []

for order in candidate_orders:
    try:
        model = ARIMA(y_train, order=order)
        res = model.fit()
        search_results.append({
            "order": order,
            "aic": res.aic,
            "bic": res.bic
        })
        print(f"Fitted ARIMA{order}  AIC={res.aic:.2f}  BIC={res.bic:.2f}")
    except Exception as e:
        print(f"ARIMA{order} failed: {e}")
        search_results.append({
            "order": order,
            "aic": np.inf,
            "bic": np.inf
        })

arima_search_df = pd.DataFrame(search_results).sort_values("aic")
arima_search_df.to_csv("model_outputs/arima_order_search.csv", index=False)

best_order = tuple(arima_search_df.iloc[0]["order"])
print("\nBest ARIMA order by AIC:", best_order)

# -------------------------------------------------------------------
# 4) FIT FINAL ARIMA MODEL ON TRAIN
# -------------------------------------------------------------------
best_model = ARIMA(y_train, order=best_order)
best_res = best_model.fit()

print("\nFinal ARIMA summary:")
print(best_res.summary())

# Save summary text
with open("model_outputs/arima_summary.txt", "w") as f:
    f.write(str(best_res.summary()))

# -------------------------------------------------------------------
# 5) FORECAST OVER TEST PERIOD
# -------------------------------------------------------------------
n_test = len(y_test)
fc = best_res.forecast(steps=n_test)
fc.index = y_test.index  # align index

# -------------------------------------------------------------------
# 6) EVALUATION METRICS
# -------------------------------------------------------------------
rmse = np.sqrt(mean_squared_error(y_test, fc))
mae  = mean_absolute_error(y_test, fc)

print(f"\nARIMA{best_order} performance on TEST:")
print(f"RMSE: {rmse:.6f}")
print(f"MAE : {mae:.6f}")

metrics_df = pd.DataFrame({
    "order": [str(best_order)],
    "rmse": [rmse],
    "mae": [mae],
})
metrics_df.to_csv("model_outputs/arima_metrics.csv", index=False)

# -------------------------------------------------------------------
# 7) PLOT: TRAIN, TEST, FORECAST
# -------------------------------------------------------------------
plt.figure(figsize=(12,5))
plt.plot(y_train.index, y_train, label="Train", linewidth=1)
plt.plot(y_test.index, y_test, label="Test", linewidth=1)
plt.plot(fc.index, fc, label=f"ARIMA{best_order} Forecast", linewidth=2)
plt.title("Daily Log Returns – ARIMA Forecast")
plt.xlabel("Date")
plt.ylabel("Daily log return")
plt.legend()
plt.tight_layout()
plt.savefig("model_outputs/arima_forecast.png")
plt.show()

# -------------------------------------------------------------------
# 8) OPTIONAL: STORE FORECAST IN A DAILY DATAFRAME
# -------------------------------------------------------------------
arima_daily = pd.DataFrame({
    "daily_log_ret": y,
    "arima_forecast": fc
})

arima_daily.to_csv("model_outputs/arima_daily_series.csv")

# ================================================
#   9) INTERPRETATION OF ARIMA RESULTS
# ================================================

print("\n" + "="*40)
print("         ARIMA MODEL INTERPRETATION")
print("="*40 + "\n")

# Unpack best order
p, d, q = best_order

print(f"Chosen ARIMA order (by AIC): ARIMA({p},{d},{q})")
print(f"Test RMSE: {rmse:.6f}")
print(f"Test MAE : {mae:.6f}")

# -------------------------
# Mean & bias interpretation
# -------------------------
mean_forecast = fc.mean()
mean_actual_test = y_test.mean()

print("\nMean behaviour:")
print(f"- Average actual test return   : {mean_actual_test:.6f}")
print(f"- Average ARIMA forecast return: {mean_forecast:.6f}")

if abs(mean_forecast) < 1e-4:
    print("  → Forecast mean is essentially zero (typical for FX returns).")
else:
    print("  → Forecast mean shows a small drift (model sees slight directional bias).")

# -------------------------
# Structure of the ARIMA
# -------------------------
print("\nModel structure:")

if p == 0:
    print("- No AR (autoregressive) term: model does not use past returns directly.")
else:
    print(f"- AR order p = {p}: past {p} return(s) help explain today's return.")

if d == 0:
    print("- No differencing (d = 0): model uses the series as-is (already stationary enough).")
else:
    print(f"- Differencing order d = {d}: long-term trend/drift removed before modeling.")

if q == 0:
    print("- No MA (moving average) term: shocks are not explicitly smoothed.")
else:
    print(f"- MA order q = {q}: past shocks/innovations influence current return.")

# Try to say something about AR coefficients (if any)
try:
    ar_params = getattr(best_res, "arparams", None)
    if ar_params is not None and len(ar_params) > 0:
        print("\nAR coefficients:")
        for i, coef in enumerate(ar_params, start=1):
            print(f"  AR({i}) = {coef:.4f}")
        if np.all(np.abs(ar_params) < 0.2):
            print("  → AR effects are small: little linear predictability in returns.")
        else:
            print("  → AR effects are sizeable: past returns have some predictive power.")
except Exception:
    pass

# -------------------------
# Forecast quality (shape)
# -------------------------
corr_fc = np.corrcoef(y_test.values, fc.values)[0, 1]

print("\nForecast vs actual shape:")
print(f"- Correlation between forecast and actual test returns: {corr_fc:.3f}")

if abs(corr_fc) < 0.1:
    print("  → Forecasts have very low linear correlation with realized returns.")
    print("    This is common for FX returns (hard to predict direction).")
else:
    print("  → Forecasts show some alignment with realized returns.")

print("\nFinancial interpretation:")
print("- ARIMA here models *linear dynamics* in daily FX returns.")
print("- Near-zero mean forecasts and modest AR coefficients are normal for FX.")
print("- The real edge often lies in volatility modeling (GARCH) and regime shifts (HMM, NN).")

print("\n" + "="*40 + "\n")

"""## 1.4. GARCH model"""

# ==============================================
#     ROBUST ROLLING GARCH(1,1) FORECAST
# ==============================================
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

if not os.path.exists("model_outputs"):
    os.makedirs("model_outputs")

# -----------------------------------------------------
# 1) DAILY LOG RETURNS
# -----------------------------------------------------
daily_close = data["Close"].resample("1D").last().dropna()
daily_ret = np.log(daily_close / daily_close.shift(1)).dropna()

# Train/test split (position-based)
split_idx = int(len(daily_ret) * 0.8)
y_train = daily_ret.iloc[:split_idx]
y_test  = daily_ret.iloc[split_idx:]

# Storage for forecasts
mean_forecasts = []
vol_forecasts  = []

# -----------------------------------------------------
# 2) ROLLING 1-STEP-AHEAD GARCH FORECASTING
# -----------------------------------------------------

for i in range(len(y_test)):
    # Expanding window: y[:split_idx + i]
    y_window = daily_ret.iloc[:split_idx + i]

    # Fit GARCH(1,1) with AR(1) mean
    am = arch_model(y_window, mean='AR', lags=1,
                    vol='GARCH', p=1, q=1, dist='normal')
    res = am.fit(disp="off")

    # 1-step ahead forecast
    f = res.forecast(horizon=1)

    # Extract mean forecast
    mean_fc = f.mean.iloc[-1, 0]
    # Extract volatility forecast (sigma)
    vol_fc = np.sqrt(f.variance.iloc[-1, 0])

    mean_forecasts.append(mean_fc)
    vol_forecasts.append(vol_fc)

# Convert to Series with correct index
mu_fc = pd.Series(mean_forecasts, index=y_test.index, name="garch_mean_fc")
sigma_fc = pd.Series(vol_forecasts, index=y_test.index, name="garch_sigma_fc")

# -----------------------------------------------------
# 3) EVALUATION
# -----------------------------------------------------
rmse = np.sqrt(mean_squared_error(y_test, mu_fc))
mae  = mean_absolute_error(y_test, mu_fc)

print(f"\nRolling GARCH(1,1) AR(1) Results:")
print(f"RMSE: {rmse:.6f}")
print(f"MAE : {mae:.6f}")

pd.DataFrame({
    "model": ["Rolling GARCH(1,1)-AR(1)"],
    "rmse": [rmse],
    "mae": [mae]
}).to_csv("model_outputs/garch_metrics.csv", index=False)


# -----------------------------------------------------
# 4) PLOT MEAN FORECAST VS RETURNS
# -----------------------------------------------------
plt.figure(figsize=(12,5))
plt.plot(y_train.index, y_train, label="Train")
plt.plot(y_test.index, y_test, label="Test")
plt.plot(mu_fc.index, mu_fc, label="GARCH Mean Forecast", linewidth=2)
plt.title("Rolling GARCH(1,1) – Mean Forecast")
plt.tight_layout()
plt.legend()
plt.savefig("model_outputs/garch_mean_forecast.png")
plt.show()


# -----------------------------------------------------
# 5) PLOT VOLATILITY FORECAST
# -----------------------------------------------------
realized_vol = daily_ret.rolling(20).std()
realized_vol_test = realized_vol.loc[y_test.index]

plt.figure(figsize=(12,5))
plt.plot(realized_vol_test, label="Realized Vol (20D)", linewidth=1)
plt.plot(sigma_fc, label="GARCH Forecast Vol", linewidth=2)
plt.title("Rolling GARCH Volatility Forecast")
plt.tight_layout()
plt.legend()
plt.savefig("model_outputs/garch_vol_forecast.png")
plt.show()


# -----------------------------------------------------
# 6) SAVE SERIES
# -----------------------------------------------------
out = pd.DataFrame({
    "daily_log_ret": daily_ret,
})
out.loc[y_test.index, "garch_mean_fc"] = mu_fc
out.loc[y_test.index, "garch_sigma_fc"] = sigma_fc

out.to_csv("model_outputs/garch_daily_series.csv")

# ================================================
#      INTERPRETATION OF GARCH RESULTS
# ================================================

print("\n====== GARCH INTERPRETATION ======\n")

print("Model: Rolling GARCH(1,1) with AR(1) Mean")
print(f"Test RMSE: {rmse:.6f}")
print(f"Test MAE : {mae:.6f}")

# Mean interpretation
if abs(mu_fc.mean()) < 1e-4:
    print("\nMean Forecast:")
    print("- The predicted return is essentially zero.")
    print("- This is normal: FX returns are typically unpredictable in mean.")
else:
    print(f"\nMean Forecast average: {mu_fc.mean():.6f}")
    print("- Small drift detected in short-term returns.")

# Volatility interpretation
avg_vol = sigma_fc.mean()

print("\nVolatility Forecast:")
print(f"- Average forecasted volatility (sigma): {avg_vol:.6f}")

if avg_vol > y_test.std():
    print("- GARCH predicts *higher* future volatility than realized.")
else:
    print("- GARCH predicts *lower or equal* future volatility relative to realized.")

print("\nVolatility Clustering:")
print("- GARCH(1,1) captures volatility clustering.")
print("- High-volatility periods are followed by high volatility, and vice-versa.")

print("\nFinancial Meaning:")
print("- GARCH does NOT forecast direction, it forecasts uncertainty.")
print("- Useful for risk modeling, VaR, volatility forecasting.")
print("- Mean returns remain noisy; volatility is more predictable.")

print("\n==================================\n")

"""# Chapter 2 : Define calculate_metrics"""

def calculate_metrics(nav, rf=0.02):
    """
    Parameters:

    nav: pd.Series — Net Asset Value curve (cumulative NAV)

    rf: Risk-free rate (annualized)

    Returns:

    dict — A dictionary of calculated performance metrics.
    """
    # Ensure NAV is sorted in ascending order by date
    nav = nav.sort_index()

    # -----------------------
    # 1️- Daily returns
    # -----------------------
    daily_ret = nav.pct_change().dropna()

    # -----------------------
    # 2️- Annualized return and volatility
    # -----------------------
    years = (nav.index[-1] - nav.index[0]).days / 365
    cumulative_return = nav.iloc[-1]/nav.iloc[0] - 1
    annual_return = (nav.iloc[-1]/nav.iloc[0])**(1/years) - 1
    annual_vol = daily_ret.std() * np.sqrt(252)

    # -----------------------
    # 3️- Maximum drawdown & Calmar ratio
    # -----------------------
    cum_max = nav.cummax()
    drawdown = (nav - cum_max) / cum_max
    max_dd = drawdown.min()
    calmar = annual_return / abs(max_dd) if max_dd != 0 else np.nan


    # -----------------------
    # 4️- Sharpe Ratio
    # -----------------------
    sharpe = (annual_return - rf) / annual_vol if annual_vol != 0 else np.nan

    # -----------------------
    # 5️- Omega Ratio
    # -----------------------
    threshold = rf/252
    omega = daily_ret[daily_ret>threshold].sum() / abs(daily_ret[daily_ret<=threshold].sum()) if abs(daily_ret[daily_ret<=threshold].sum())>0 else np.nan

    return {
        "Cumulative Return": cumulative_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Calmar Ratio": calmar,
        "Omega Ratio": omega,
        "Max Drawdown": max_dd
    }

"""# Chapter 3: Baseline Signal (Minute-level SMA + Volatility Normalization)"""

# Feature engineering
# ---------------------------
data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
data['volatility'] = data['log_ret'].rolling(window=60).std()

scaler = StandardScaler()
macro_features = ['US_GDP', 'Median_CPI', '10Y_Treasury_FF_Spread']
data[macro_features] = scaler.fit_transform(data[macro_features])

# ---------------------------
# Baseline Signal
# ---------------------------
fast_window = 60
slow_window = 240

data['SMA_fast'] = data['Close'].rolling(window=fast_window).mean()
data['SMA_slow'] = data['Close'].rolling(window=slow_window).mean()

data['baseline_signal'] = 0
valid_mask = (~data['SMA_fast'].isna()) & (~data['SMA_slow'].isna())
data.loc[valid_mask & (data['SMA_fast'] > data['SMA_slow']), 'baseline_signal'] = 1
data.loc[valid_mask & (data['SMA_fast'] < data['SMA_slow']), 'baseline_signal'] = -1

target_vol = 0.01
epsilon = 1e-6
data['baseline_signal_vol_adj'] = data['baseline_signal'] * target_vol / (data['volatility'] + epsilon)
data['baseline_signal_vol_adj'] = data['baseline_signal_vol_adj'].clip(-1, 1)

data['baseline_returns'] = data['baseline_signal_vol_adj'].shift(1) * data['log_ret']
data['baseline_returns'] = data['baseline_returns'].fillna(0)
data['baseline_cumret'] = (1 + data['baseline_returns']).cumprod()

# ---------------------------
# Plotting
# ---------------------------
first_signal_idx = data['baseline_signal_vol_adj'].ne(0).idxmax()
first_pos = data.index.get_loc(first_signal_idx)

pre_idx = data.index[:first_pos]

fig, ax = plt.subplots(figsize=(12,6))

# 1- SMA warming-up gray background
if len(pre_idx) > 0:
    ax.axvspan(pre_idx[0], pre_idx[-1], color='gray', alpha=0.2)

# 2️-SMA warming-up vertical red line at start
ax.axvline(first_signal_idx, color='red', linestyle='--', linewidth=2, label='SMA warming-up')

# 3️- Baseline cumulative return blue line
ax.plot(data.index, data['baseline_cumret'], color='blue', label='Baseline Cumulative Return')

# 4️-Legend
ax.set_title("Baseline Strategy (Minute-level SMA + Volatility Normalization)")
ax.set_xlabel("DateTime")
ax.set_ylabel("Cumulative Return")
ax.grid(True)
ax.legend(loc='best')

plt.show()

# -----------------------
# Printing of baseline strategy metrics
# -----------------------
baseline_nav = data['baseline_cumret'].dropna()

metrics = calculate_metrics(baseline_nav)

print("\n===== Baseline Strategy Metrics =====")
for k,v in metrics.items():
    print(f"{k}: {v:.4f}")

"""# Chapter 4: Single architecture (Minute-level)"""

# ================================
# Chapter 4: HMM Regime Detection (Minute-level)
# ================================

hmm_features = data[['log_ret', 'volatility']].dropna().values

n_states = 3
hmm_model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=1000, random_state=42)
hmm_model.fit(hmm_features)

hidden_states = hmm_model.predict(hmm_features)
data = data.iloc[-len(hidden_states):]  # Align
data['HMM_state'] = hidden_states

state_means = [hmm_features[hidden_states==i,0].mean() for i in range(n_states)]
high_mean_state = np.argmax(state_means)
low_mean_state = np.argmin(state_means)

data['HMM_signal'] = 0
data.loc[data['HMM_state'] == high_mean_state, 'HMM_signal'] = 1
data.loc[data['HMM_state'] == low_mean_state, 'HMM_signal'] = -1

data['HMM_returns'] = data['HMM_signal'].shift(1) * data['log_ret']
data['HMM_cumret'] = data['HMM_returns'].cumsum().apply(np.exp)
first_signal_idx = data.index[0]

plt.figure(figsize=(12,6))

# Cumulative return (blue line)
plt.plot(data.index, data['HMM_cumret'], color='blue', label='HMM Strategy')

# Vertical line: first signal time
plt.axvline(first_signal_idx, color='red', linestyle='--', linewidth=2, label='First Signal')

# Horizontal line: cumulative return = 1
plt.axhline(1, color='green', linestyle='--', linewidth=2, label='Initial Level')

plt.title("HMM Regime-Based Strategy Cumulative Return (Minute-level)")
plt.xlabel("DateTime")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()

baseline_nav = data['HMM_cumret'].dropna()

metrics = calculate_metrics(baseline_nav)

print("\n===== Baseline Strategy Metrics =====")
for k,v in metrics.items():
    print(f"{k}: {v:.4f}")

# ================================
# Complete HMM Strategy: Manual Multi-Init, Daily -> Monthly, Plot
# ================================

# ---------------------------
# 1️- Prepare features
# ---------------------------
hmm_features = data[['log_ret', 'volatility']].dropna().values

# Feature scaling
scaler = StandardScaler()
hmm_features_scaled = scaler.fit_transform(hmm_features)

# Optional: sample data for faster training
hmm_features_sampled = hmm_features_scaled[::5]

# ---------------------------
# 2️- Fit HMM with manual multi-init
# ---------------------------
n_states = 3
best_model = None
best_score = -np.inf

for seed in range(5):
    model = GaussianHMM(
        n_components=n_states,
        covariance_type='full',
        n_iter=2000,
        tol=1e-4,
        random_state=seed,
        verbose=False
    )
    model.fit(hmm_features_sampled)
    score = model.score(hmm_features_scaled)
    if score > best_score:
        best_score = score
        best_model = model

hmm_model = best_model

# ---------------------------
# 3️- Predict hidden states
# ---------------------------
hidden_states = hmm_model.predict(hmm_features_scaled)
data = data.iloc[-len(hidden_states):]  # Align
data['HMM_state'] = hidden_states

# ---------------------------
# 4️- Generate trading signals
# ---------------------------
state_means = [hmm_features[hidden_states==i,0].mean() for i in range(n_states)]
high_mean_state = np.argmax(state_means)
low_mean_state = np.argmin(state_means)

data['HMM_signal'] = 0
data.loc[data['HMM_state'] == high_mean_state, 'HMM_signal'] = 1
data.loc[data['HMM_state'] == low_mean_state, 'HMM_signal'] = -1

# ---------------------------
# 5️- Apply leverage and compute daily & monthly cumulative returns
# ---------------------------
leverage = 10

# Per-minute leveraged log return
data['leveraged_log_ret'] = leverage * data['HMM_signal'].shift(1) * data['log_ret']

# Daily log return (sum per day)
daily_log_ret = data['leveraged_log_ret'].resample('D').sum()

# Daily cumulative return
cumret_daily = np.exp(daily_log_ret.cumsum())

# Monthly cumulative return (take last day of each month)
cumret_monthly = cumret_daily.resample('ME').last()

# First effective signal
first_signal_idx = data[data['HMM_signal'] != 0].index[0]

# ---------------------------
# 6️- Plot
# ---------------------------
plt.figure(figsize=(12,6))

# Blue line: monthly cumulative return
plt.plot(cumret_monthly.index, cumret_monthly.values, color='blue',
         label=f'HMM Strategy ({leverage}x Leverage, Monthly Compounding)')

# Red vertical line: first effective signal
plt.axvline(first_signal_idx, color='red', linestyle='--', linewidth=2, label='First Signal')

# Green horizontal line: initial level
plt.axhline(1, color='green', linestyle='--', linewidth=2, label='Initial Level')

plt.title(f"HMM Regime-Based Strategy Cumulative Return (Monthly, {leverage}x Leverage)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()

# Daily equity curve as NAV series
nav = cumret_daily.dropna()

metrics = calculate_metrics(nav)

print("\n===== HMM Strategy Performance =====")
for k,v in metrics.items():
    print(f"{k}: {v:.4f}")

# ---------- Input parameters (you can modify) ----------
leverage = 5  # Current leverage (for display purposes only)
risk_free_rate_annual = 0.02  # Annualized risk-free rate for Sharpe ratio (adjustable)
# -------------------------------------------------------

if 'cumret_monthly' not in globals():
    # If daily cumulative net value (cumret_daily) exists, take the month-end values.
    if 'cumret_daily' in globals():
        cumret_monthly = cumret_daily.resample('ME').last()
    else:
        raise RuntimeError("cumret_monthly or cumret_daily not found. Please run the aggregation step first.")

# Ensure the index is a DatetimeIndex (month-end)
cumret_monthly = cumret_monthly.dropna()
cumret_monthly.index = pd.to_datetime(cumret_monthly.index)

# 1) Calculate key metrics
# Final value (last month’s net value)
final_value = float(cumret_monthly.iloc[-1])

# Monthly return rates (percentage)
monthly_returns = cumret_monthly.pct_change().dropna()

# Annualized return (CAGR) — using months
n_months = len(cumret_monthly)
if n_months > 1:
    years = n_months / 12.0
    cagr = (final_value ** (1.0 / years)) - 1.0
else:
    cagr = np.nan

# Annualized volatility (based on monthly returns)
annual_vol = monthly_returns.std() * np.sqrt(12) if len(monthly_returns) > 1 else np.nan

# Annualized mean monthly return × 12 = annual return (as a check)
annual_return_from_monthly = monthly_returns.mean() * 12 if len(monthly_returns) > 0 else np.nan

# Sharpe ratio (annualized)
if not np.isnan(annual_vol) and annual_vol > 0:
    sharpe = (cagr - risk_free_rate_annual) / annual_vol
else:
    sharpe = np.nan

# Maximum drawdown (based on monthly net value)
roll_max = cumret_monthly.cummax()
drawdown = cumret_monthly / roll_max - 1.0
max_drawdown = drawdown.min()

# Monthly win rate & average monthly return
win_rate = (monthly_returns > 0).sum() / len(monthly_returns) if len(monthly_returns) > 0 else np.nan
avg_monthly = monthly_returns.mean()
median_monthly = monthly_returns.median()

# Output metrics
print(f"Final value (net multiple): {final_value:.4f}")
print(f"Months: {n_months}, Years: {years:.2f}")
print(f"CAGR (annualized): {cagr:.2%}")
print(f"Annual vol (from monthly): {annual_vol:.2%}")
print(f"Sharpe (annual, rf={risk_free_rate_annual:.2%}): {sharpe:.2f}")
print(f"Annual return (mean*12): {annual_return_from_monthly:.2%}")
print(f"Max drawdown: {max_drawdown:.2%}")
print(f"Monthly win rate: {win_rate:.2%}")
print(f"Avg monthly return: {avg_monthly:.2%}, Median monthly: {median_monthly:.2%}")

# 2) Plotting: monthly NAV, monthly returns bar chart, drawdown
fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1, 1]})

# Net value curve
ax = axes[0]
ax.plot(cumret_monthly.index, cumret_monthly.values, color='blue', lw=2, label=f'Net (Monthly, {leverage}x)')
ax.axhline(1.0, color='gray', linestyle='--', lw=1)
ax.set_title('Monthly Cumulative Return (Net Value)')
ax.set_ylabel('Net multiple')
ax.legend()
ax.grid(True)

# Monthly returns bar chart
ax = axes[1]
ax.bar(monthly_returns.index, monthly_returns.values, color=np.where(monthly_returns > 0, '#2ca02c', '#d62728'))
ax.set_title('Monthly Returns')
ax.set_ylabel('Return')
ax.grid(True)

# Drawdown
ax = axes[2]
ax.plot(drawdown.index, drawdown.values, color='maroon', label='Drawdown')
ax.fill_between(drawdown.index, drawdown.values, 0, where=drawdown < 0, color='maroon', alpha=0.3)
ax.set_title('Drawdown (from peak)')
ax.set_ylabel('Drawdown')
ax.grid(True)

plt.tight_layout()
plt.show()

# 3) Suggestions: quick ideas to scale the final value (commented tips)
print("\nQuick tuning suggestions (try one at a time):")
print("  - Increase leverage (e.g. leverage = 8 or 10), but note: volatility and drawdown scale accordingly.")
print("  - Apply signal smoothing or thresholding: require state persistence before taking a position.")
print("  - Use daily_signal = data['HMM_signal'].resample('D').mean() instead of .last() to capture intra-day dynamics.")
print("  - Use smaller sampling intervals or longer HMM windows (adjust fast/slow) for more stable signals.")
print("  - Apply position sizing (e.g. volatility targeting) to reduce large drawdowns.")

# HMM strategy performance metrics
metrics = calculate_metrics(cumret_monthly)

print("\n===== HMM Strategy Performance Metrics =====")
print(pd.Series(metrics).round(4))

# ------- Global Parameters -------
# leverage_list = [1, 2, 5, 10, 15]   # Multiple leverage levels
leverage_list = [1, 2, 5, 10]         # Multiple leverage levels
risk_free_rate_annual = 0.02          # Annualized risk-free rate
results = []                          # To store performance metrics

# ------- Check Base Data -------
if 'data' not in globals():
    raise RuntimeError("Please run the HMM training section first so that the 'data' variable exists!")

if 'log_ret' not in data.columns or 'HMM_signal' not in data.columns:
    raise RuntimeError("The 'data' DataFrame must contain 'HMM_signal' and 'log_ret'. Please run the HMM strategy part first.")

# ------- Plot Preparation -------
plt.figure(figsize=(14,7))

color_map = plt.cm.tab10(np.linspace(0, 1, len(leverage_list)))

for i, lev in enumerate(leverage_list):

    # Calculate leveraged returns
    data[f'lev_ret_{lev}'] = lev * data['HMM_signal'].shift(1) * data['log_ret']

    # Aggregate log returns to daily level → daily NAV
    daily_log_ret = data[f'lev_ret_{lev}'].resample('D').sum()
    cumret_daily = np.exp(daily_log_ret.cumsum())

    # Month-end NAV (ME = month end)
    cumret_monthly = cumret_daily.resample('ME').last().dropna()

    # Monthly returns
    monthly_ret = cumret_monthly.pct_change().dropna()

    # Compute performance metrics
    final_val = cumret_monthly.iloc[-1]
    n_months = len(cumret_monthly)
    years = n_months / 12
    cagr = final_val ** (1/years) - 1
    annual_vol = monthly_ret.std() * np.sqrt(12)
    annual_ret_mean = monthly_ret.mean() * 12
    sharpe = (cagr - risk_free_rate_annual) / annual_vol if annual_vol > 0 else np.nan
    roll_max = cumret_monthly.cummax()
    max_dd = ((cumret_monthly / roll_max) - 1).min()
    win_rate = (monthly_ret > 0).mean()

    results.append([
        lev, final_val, cagr, annual_vol, sharpe, max_dd, win_rate, annual_ret_mean
    ])

    # Plot performance curve
    plt.plot(cumret_monthly.index, cumret_monthly.values,
             label=f'{lev}x leverage', color=color_map[i], linewidth=2)

# ------- Final Combined Plot -------
plt.axhline(1, color='gray', linestyle='--', linewidth=1)
plt.title("HMM Strategy — Multi-Leverage Net Value Curve (Monthly)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (Net Value)")
plt.legend()
plt.grid(True)
plt.show()

# ------- Output Performance Table -------
perf_df = pd.DataFrame(results, columns=[
    "Leverage",
    "Final Net Value",
    "CAGR",
    "Annual Vol",
    "Sharpe",
    "Max Drawdown",
    "Win Rate (Monthly)",
    "Annual Return (mean*12)"
])

# Format output
perf_df.style.format({
    "Final Net Value": "{:.2f}",
    "CAGR": "{:.2%}",
    "Annual Vol": "{:.2%}",
    "Sharpe": "{:.2f}",
    "Max Drawdown": "{:.2%}",
    "Win Rate (Monthly)": "{:.2%}",
    "Annual Return (mean*12)": "{:.2%}"
})

# ------- Global Parameters -------
# leverage_list = [1, 2, 5, 10]   # Multiple leverage levels
risk_free_rate_annual = 0.02       # Annualized risk-free rate
metrics_results = []               # To store results

# Check base data
if 'data' not in globals():
    raise RuntimeError("Please run the HMM training section first so that the 'data' variable exists!")

# Loop to compute metrics for each leverage level
for lev in leverage_list:
    # Daily leveraged strategy returns
    data[f'lev_ret_{lev}'] = lev * data['HMM_signal'].shift(1) * data['log_ret']

    # Daily net value
    daily_log_ret = data[f'lev_ret_{lev}'].resample('D').sum()
    cumret_daily = np.exp(daily_log_ret.cumsum())

    # Compute performance metrics
    metrics = calculate_metrics(cumret_daily, rf=risk_free_rate_annual)

    # Save leverage info
    metrics['Leverage'] = lev
    metrics_results.append(metrics)

# Generate DataFrame
metrics_df = pd.DataFrame(metrics_results).set_index('Leverage')

# Format output
metrics_df_formatted = metrics_df.style.format({
    "Cumulative Return": "{:.4f}",
    "Annual Return": "{:.4f}",
    "Annual Volatility": "{:.4f}",
    "Sharpe Ratio": "{:.4f}",
    "Calmar Ratio": "{:.4f}",
    "Omega Ratio": "{:.4f}",
    "Max Drawdown": "{:.4f}"
})

# Print
print("===== Strategy Performance Metrics =====")
metrics_df_formatted

"""# Chapter 5: Transformer (Minute-level)"""

# ================================
# Tiny Transformer: GPU + Downsampling + Monthly Backtest
# (Last 24 months of data)
# ---------------------------
# 1️- Set device (CPU / MPS / CUDA)
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")
print("Using device:", device)

# ---------------------------
# 2️- Recent 24 months of data
# ---------------------------
end_date = data.index.max()
# start_date = end_date - pd.DateOffset(months=3)
start_date = end_date - pd.DateOffset(months=24)  # Changed to 24 months; 36 months causes crash
data_recent = data.loc[start_date:end_date].copy()

# Downsample
sample_minute = 20
data_resampled = data_recent.resample(f"{sample_minute}min").last().dropna()

# ---------------------------
# 3️- Build training sequences
# ---------------------------
lookback = 30
feature_cols = ['log_ret', 'volatility', 'US_GDP', 'Median_CPI', '10Y_Treasury_FF_Spread']

X_seq, y_seq = [], []
for i in range(lookback, len(data_resampled)):
    X_seq.append(data_resampled[feature_cols].iloc[i-lookback:i].values)
    y_seq.append(data_resampled['log_ret'].iloc[i])

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)

# Check for NaN
assert not np.isnan(X_seq).any(), "X_seq contains NaN"
assert not np.isnan(y_seq).any(), "y_seq contains NaN"

# Convert to PyTorch tensors
X_seq = torch.tensor(X_seq, dtype=torch.float32)
y_seq = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1)

# ---------------------------
# 4️- Tiny Transformer model
# ---------------------------
class TinyTransformer(nn.Module):
    def __init__(self, input_dim, d_model=16, nhead=2, num_layers=1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # seq_len, batch, d_model
        x = self.transformer_encoder(x)
        x = x[-1, :, :]  # last time step
        out = self.fc_out(x)
        return out

model = TinyTransformer(input_dim=X_seq.shape[2]).to(device)

# ---------------------------
# 5️-  Loss function + optimizer
# ---------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# 6️-  Mini-batch training
# ---------------------------
n_epochs = 5
batch_size = 32
n_samples = X_seq.shape[0]

for epoch in range(n_epochs):
    permutation = torch.randperm(n_samples)
    epoch_loss = 0.0
    for start in range(0, n_samples, batch_size):
        idx = permutation[start:start+batch_size]
        batch_X = X_seq[idx].to(device)
        batch_y = y_seq[idx].to(device)

        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")

# ---------------------------
# 7️-  Batch prediction
# ---------------------------
y_pred_list = []
batch_size_pred = 4096
for start in range(0, n_samples, batch_size_pred):
    batch_X = X_seq[start:start+batch_size_pred].to(device)
    with torch.no_grad():
        y_pred_list.append(model(batch_X).cpu().numpy())

y_pred = np.concatenate(y_pred_list).flatten()

# Align predicted signals
data_signals = data_resampled.iloc[lookback:].copy()
data_signals['TinyTrans_signal'] = 0
data_signals.loc[y_pred > 0, 'TinyTrans_signal'] = 1
data_signals.loc[y_pred < 0, 'TinyTrans_signal'] = -1

# ---------------------------
# 8️-  Leveraged returns & cumulative NAV
# ---------------------------
leverage = 5
data_signals['leveraged_ret'] = leverage * data_signals['TinyTrans_signal'].shift(1) * data_signals['log_ret']

cumret_daily = data_signals['leveraged_ret'].resample('D').sum().cumsum().apply(np.exp)
cumret_monthly = cumret_daily.resample('M').last().dropna()

# ---------------------------
# 9️-  Output performance metrics
# ---------------------------
final_value = float(cumret_monthly.iloc[-1])
monthly_returns = cumret_monthly.pct_change().dropna()
years = len(cumret_monthly) / 12
cagr = final_value ** (1/years) - 1
annual_vol = monthly_returns.std() * np.sqrt(12)
sharpe = (cagr - 0.02) / annual_vol
drawdown = cumret_monthly / cumret_monthly.cummax() - 1
max_drawdown = drawdown.min()
win_rate = (monthly_returns > 0).sum() / len(monthly_returns)

print(f"Final Value: {final_value:.4f}")
print(f"CAGR: {cagr:.2%}, Sharpe: {sharpe:.2f}, Max DD: {max_drawdown:.2%}, Monthly Win Rate: {win_rate:.2%}, Annual Vol: {annual_vol:.2%}")

# ---------------------------
# 10️- Visualization
# ---------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1, 1]})

axes[0].plot(cumret_monthly.index, cumret_monthly.values, color='blue', lw=2, label=f'TinyTrans ({leverage}x)')
axes[0].axhline(1, color='gray', linestyle='--', lw=1)
axes[0].set_title("Monthly Cumulative Return (Net Value)")
axes[0].set_ylabel("Net Multiple")
axes[0].legend()
axes[0].grid(True)

axes[1].bar(monthly_returns.index, monthly_returns.values, color=np.where(monthly_returns > 0, '#2ca02c', '#d62728'))
axes[1].set_title("Monthly Returns")
axes[1].set_ylabel("Return")
axes[1].grid(True)

axes[2].plot(drawdown.index, drawdown.values, color='maroon', lw=1.5)
axes[2].fill_between(drawdown.index, drawdown.values, 0, where=drawdown < 0, color='maroon', alpha=0.3)
axes[2].set_title("Drawdown (from Peak)")
axes[2].set_ylabel("Drawdown")
axes[2].grid(True)

plt.tight_layout()
plt.show()

# ================================
# Compute Tiny Transformer Metrics Table
# ================================

# Assume cumret_daily has already been generated (from previous steps)
leverage_list = [5]  # Use your current leverage setting here, e.g., 5
metrics_results = []

for lev in leverage_list:
    # Use calculate_metrics to compute performance metrics
    metrics = calculate_metrics(cumret_daily, rf=0.02)
    metrics['Leverage'] = lev
    metrics_results.append(metrics)

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics_results).set_index('Leverage')

# Format display
metrics_df_formatted = metrics_df.style.format({
    "Cumulative Return": "{:.4f}",
    "Annual Return": "{:.4f}",
    "Annual Volatility": "{:.4f}",
    "Sharpe Ratio": "{:.4f}",
    "Calmar Ratio": "{:.4f}",
    "Omega Ratio": "{:.4f}",
    "Max Drawdown": "{:.4f}"
})

print("===== Tiny Transformer Strategy Performance Metrics =====")
metrics_df_formatted

# ================
# 1️- Data Processing
# ================

# Use the most recent 24 months
end = data.index.max()
start = end - pd.DateOffset(months=24)
df = data.loc[start:end].copy()

# Downsample to 20-minute frequency
df = df.resample("20min").last().dropna()

# Target: next-step return
df["target"] = df["log_ret"].shift(-1)
df.dropna(inplace=True)

# Feature columns
feat_cols = ["log_ret", "volatility", "US_GDP", "Median_CPI", "10Y_Treasury_FF_Spread"]
X = df[feat_cols].values
y = df["target"].values

lookback = 30

class FXDataset(Dataset):
    def __init__(self, X, y, lookback):
        self.X = X
        self.y = y
        self.lookback = lookback
    def __len__(self):
        return len(self.X) - self.lookback
    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i:i+self.lookback], dtype=torch.float32),
            torch.tensor(self.y[i+self.lookback], dtype=torch.float32)
        )

dataset = FXDataset(X, y, lookback)
train_size = int(len(dataset) * 0.8)
trainset, testset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ================
# 2️- Transformer Model
# ================
class FXTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # last time step
        return self.fc(x).squeeze()

model = FXTransformer(input_dim=len(feat_cols)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# ================
# 3️- Training
# ================
epochs = 8
for e in range(epochs):
    model.train()
    losses = []
    for xb, yb in trainloader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"Epoch {e+1}/{epochs}, Loss: {np.mean(losses):.6f}")

# ================
# 4️- Prediction Signals
# ================
model.eval()
preds = []
with torch.no_grad():
    for xb, _ in DataLoader(dataset, batch_size=1):
        xb = xb.to(device)
        preds.append(float(model(xb)))

df = df.iloc[len(df) - len(preds):].copy()
df["pred"] = preds
df["signal"] = np.where(df["pred"] > 0, 1, -1)

# ================
# 5️- Backtesting
# ================
leverage_list = [1, 2, 5, 10]
results = {}

for lev in leverage_list:
    df[f"ret_{lev}x"] = lev * df["signal"].shift(1) * df["log_ret"]
    df[f"nav_{lev}x"] = df[f"ret_{lev}x"].cumsum().apply(np.exp)

# ================
# 6️- Plot Results
# ================
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))

for lev in leverage_list:
    plt.plot(df.index, df[f"nav_{lev}x"], label=f"{lev}x")

plt.title("Transformer FX Strategy (20min, 24M window, lookback=30)")
plt.axhline(1, color="gray", linestyle="--")
plt.legend()
plt.grid(True)
plt.show()

# ================================
# Compute Transformer FX Strategy Performance Table
# ================================
# leverage_list = [1, 2, 5, 10]
metrics_results = []

for lev in leverage_list:
    nav_series = df[f"nav_{lev}x"]
    metrics = calculate_metrics(nav_series, rf=0.02)  # Annual risk-free rate = 2%
    metrics['Leverage'] = lev
    metrics_results.append(metrics)

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics_results).set_index('Leverage')

# Format for display
metrics_df_formatted = metrics_df.style.format({
    "Cumulative Return": "{:.4f}",
    "Annual Return": "{:.4f}",
    "Annual Volatility": "{:.4f}",
    "Sharpe Ratio": "{:.4f}",
    "Calmar Ratio": "{:.4f}",
    "Omega Ratio": "{:.4f}",
    "Max Drawdown": "{:.4f}"
})

print("===== Transformer FX Strategy Performance Metrics =====")
metrics_df_formatted

# =============================
# 1️- Data Processing
# =============================
end = data.index.max()
start = end - pd.DateOffset(months=24)
df = data.loc[start:end].copy()
df = df.resample("20min").last().dropna()

lookback = 30
features = ["log_ret", "volatility", "US_GDP", "Median_CPI", "10Y_Treasury_FF_Spread"]
df["target"] = df["log_ret"].shift(-1)
df.dropna(inplace=True)

X = df[features].values
y = df["target"].values

class FXDataset(Dataset):
    def __init__(self, X, y, lookback):
        self.X = X
        self.y = y
        self.lookback = lookback
    def __len__(self):
        return len(self.X) - self.lookback
    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i:i+self.lookback], dtype=torch.float32),
            torch.tensor(self.y[i+self.lookback], dtype=torch.float32)
        )

dataset = FXDataset(X, y, lookback)

# =============================
# 2️- Transformer Model
# =============================
class FXTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # last time step
        return self.fc(x).squeeze()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = FXTransformer(input_dim=len(features)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# =============================
# 3️- Training
# =============================
train_size = int(len(dataset)*0.8)
trainset, testset = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

epochs = 8
for e in range(epochs):
    model.train()
    losses = []
    for xb, yb in trainloader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"Epoch {e+1}/{epochs}, Loss: {np.mean(losses):.6f}")

# =============================
# 4️- Prediction Signals
# =============================
model.eval()
preds = []
with torch.no_grad():
    for i in range(len(dataset)):
        xb, _ = dataset[i]
        xb = xb.unsqueeze(0).to(device)
        preds.append(float(model(xb)))
preds = np.array(preds)

df_sig = df.iloc[len(df)-len(preds):].copy()
df_sig["pred"] = preds
df_sig["signal"] = np.where(df_sig["pred"]>0, 1, -1)

# =============================
# 5️- Multi-Leverage Cumulative Net Value
# =============================
leverage_list = [1,2,5,10]
cumret_dict = {}

for lev in leverage_list:
    df_sig[f"ret_{lev}x"] = lev * df_sig["signal"].shift(1) * df_sig["log_ret"]
    df_sig[f"ret_{lev}x"].fillna(0, inplace=True)
    df_sig[f"nav_{lev}x"] = df_sig[f"ret_{lev}x"].cumsum().apply(np.exp)
    cumret_dict[lev] = df_sig[f"nav_{lev}x"]

# =============================
# 6️-  Compute Performance Metrics
# =============================
metrics_results = []
for lev, nav in cumret_dict.items():
    metrics = calculate_metrics(nav, rf=0.02)
    metrics['Leverage'] = lev
    metrics_results.append(metrics)

metrics_df = pd.DataFrame(metrics_results)
metrics_df.set_index('Leverage', inplace=True)

metrics_df_formatted = metrics_df.style.format({
    "Cumulative Return": "{:.4f}",
    "Annual Return": "{:.4f}",
    "Annual Volatility": "{:.4f}",
    "Sharpe Ratio": "{:.4f}",
    "Calmar Ratio": "{:.4f}",
    "Omega Ratio": "{:.4f}",
    "Max Drawdown": "{:.4f}"
})

print("===== Transformer FX Strategy Performance Metrics =====")
metrics_df_formatted

# =============================
# 7️- Visualization: Net Value, Monthly Returns, Drawdown
# =============================
cumret_monthly = {lev: nav.resample('ME').last() for lev, nav in cumret_dict.items()}

plt.figure(figsize=(14,8))
for lev, nav in cumret_monthly.items():
    plt.plot(nav.index, nav.values, label=f"{lev}x", lw=2)
plt.axhline(1, color='gray', linestyle='--')
plt.title("Transformer FX Strategy - Cumulative Net Value")
plt.xlabel("Date")
plt.ylabel("Net Multiple")
plt.legend()
plt.grid(True)
plt.show()

metrics_df_formatted

# TFT Transfer FX Strategy

class TFT(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_heads=4):
        super().__init__()

        # variable selection network
        self.var_proj = nn.Linear(input_dim, hidden_dim)

        # GRN for variable selection gating
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        # LSTM encoder-decoder (TFT backbone)
        self.lstm_enc = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm_dec = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Multi-head attention for temporal fusion
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # Feed-forward network
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, features]

        # variable gating
        gate = self.gate(x[:, -1, :])  # gating uses last step macro regime
        x = x * gate.unsqueeze(1)

        # projection
        x = self.var_proj(x)

        # encoder-decoder
        enc_out, (h, c) = self.lstm_enc(x)
        dec_out, _ = self.lstm_dec(x, (h, c))

        # attention fusion
        attn_out, _ = self.attn(dec_out, enc_out, enc_out)

        # prediction at last timestep
        out = self.fc(attn_out[:, -1, :])

        return out


end = data.index.max()
start = end - pd.DateOffset(months=24)

df = data.loc[start:end].copy()
df = df.resample("20min").last().dropna()

lookback = 30
features = ['log_ret','volatility','US_GDP','Median_CPI','10Y_Treasury_FF_Spread']

X, y = [], []
for i in range(lookback, len(df)):
    X.append(df[features].iloc[i-lookback:i].values)
    y.append(df['log_ret'].iloc[i])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

X = torch.tensor(X)
y = torch.sign(torch.tensor(y)).float().unsqueeze(1)  # ↑1 ↓-1


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = TFT(input_dim=X.shape[2]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

batch = 64
epochs = 5

for e in range(epochs):
    perm = torch.randperm(len(X))
    total = 0
    for i in range(0, len(X), batch):
        idx = perm[i:i+batch]
        xb = X[idx].to(device)
        yb = y[idx].to(device)

        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {e+1}, Loss={total:.4f}")


# prediction
with torch.no_grad():
    y_pred = []
    for i in range(0, len(X), 2048):
        y_pred.extend(model(X[i:i+2048].to(device)).cpu().flatten().numpy())
y_pred = np.array(y_pred)

sig = np.where(y_pred>0,1,-1)

df_sig = df.iloc[lookback:].copy()
df_sig["sig"] = sig
df_sig["ret"] = df_sig["sig"].shift()*df_sig["log_ret"]

# leverage performance
levs=[1,2,5,10]
results={}
for L in levs:
    nav = (L*df_sig["ret"]).resample("D").sum().cumsum().apply(np.exp)
    results[L]=nav


import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.figure(figsize=(10,6))

for L, nav in results.items():
    plt.plot(nav.index, nav, label=f'{L}x', linewidth=2)

plt.axhline(1, color='gray', linestyle='--', linewidth=1)

plt.title("TFT FX Strategy - Cumulative Net Value\n(20min data, 24 months, lookback=30)", fontsize=13)
plt.xlabel("Date", fontsize=11)
plt.ylabel("Net Value (cumulative)", fontsize=11)

# format y axis
plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

plt.legend(title="Leverage", fontsize=10)
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()

# -----------------------
# Strategy Performance Metrics
# -----------------------
metrics_results = {}

# Loop through each leverage level and calculate performance metrics
for L, nav in results.items():
    metrics_results[L] = calculate_metrics(nav)

# Convert results to DataFrame for readability
metrics_df = pd.DataFrame(metrics_results).T
metrics_df.index.name = "Leverage"

print("\n===== Strategy Performance Metrics =====")
metrics_df.round(4)

# ================================
# TFT + Regime Filter Backtest
# ================================
# ---------------------------
# 1️- Set Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")
print("Using device:", device)

# ---------------------------
# 2️-  Prepare Data
# ---------------------------
# Restrict to the most recent 24 months
end_date = data.index.max()
start_date = end_date - pd.DateOffset(months=24)
data_recent = data.loc[start_date:end_date].copy()

# Downsample to 20-minute frequency
sample_minute = 20
data_resampled = data_recent.resample(f"{sample_minute}min").last().dropna()

# Features and target
feature_cols = ['log_ret', 'volatility', 'US_GDP', 'Median_CPI', '10Y_Treasury_FF_Spread']
target_col = 'log_ret'

# ---------------------------
# 3️-  Build Input Sequences
# ---------------------------
lookback = 30
X_seq, y_seq = [], []
for i in range(lookback, len(data_resampled)):
    X_seq.append(data_resampled[feature_cols].iloc[i-lookback:i].values)
    y_seq.append(data_resampled[target_col].iloc[i])

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)

# Convert to PyTorch tensors
X_seq = torch.tensor(X_seq, dtype=torch.float32)
y_seq = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1)

# ---------------------------
# 4️- Tiny Transformer Model
# ---------------------------
class TinyTransformer(nn.Module):
    def __init__(self, input_dim, d_model=16, nhead=2, num_layers=1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # seq_len, batch, d_model
        x = self.transformer_encoder(x)
        x = x[-1, :, :]  # final time step
        out = self.fc_out(x)
        return out

model = TinyTransformer(input_dim=X_seq.shape[2]).to(device)

# ---------------------------
# 5️- Loss Function & Optimizer
# ---------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# 6️- Training
# ---------------------------
n_epochs = 5
batch_size = 32
n_samples = X_seq.shape[0]

for epoch in range(n_epochs):
    permutation = torch.randperm(n_samples)
    epoch_loss = 0.0
    for start in range(0, n_samples, batch_size):
        idx = permutation[start:start+batch_size]
        batch_X = X_seq[idx].to(device)
        batch_y = y_seq[idx].to(device)

        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")

# ---------------------------
# 7️- Prediction
# ---------------------------
y_pred_list = []
batch_size_pred = 4096
for start in range(0, n_samples, batch_size_pred):
    batch_X = X_seq[start:start+batch_size_pred].to(device)
    with torch.no_grad():
        y_pred_list.append(model(batch_X).cpu().numpy())

final_pred = pd.Series(
    np.concatenate(y_pred_list).flatten(),
    index=data_resampled.iloc[lookback:].index,
    name='pred'
)

# ---------------------------
# 8️- Regime Filter
# ---------------------------
def regime_filter_positions(pred_series, data, window_vol=24, window_mean=24):
    """
    Simple regime filter:
    - Compute rolling volatility and rolling mean of predictions
    - Close positions during extreme volatility or unstable signals
    """
    vol = data['log_ret'].rolling(window=window_vol, min_periods=1).std()
    mean = pred_series.rolling(window=window_mean, min_periods=1).mean()
    pos = pd.Series(0, index=pred_series.index)
    pos[mean > 0] = 1
    pos[mean < 0] = -1
    # Close position when volatility exceeds 95th percentile
    pos[vol > vol.quantile(0.95)] = 0
    return pos

pos_filtered = regime_filter_positions(final_pred, data_resampled)

# ---------------------------
# 9️- Backtest
# ---------------------------
leverages = [1, 2, 5, 10]
bt = data_resampled.iloc[lookback:].copy()
bt['pos'] = pos_filtered
bt['ret_next'] = bt['log_ret'].shift(-1)
bt = bt.dropna(subset=['ret_next'])

results = {}
for lev in leverages:
    bt[f'ret_{lev}x'] = lev * bt['pos'] * bt['ret_next']
    cum_daily = bt[f'ret_{lev}x'].resample('D').sum().cumsum().apply(np.exp)
    results[lev] = cum_daily

# ---------------------------
# 10️- Visualization
# ---------------------------
plt.figure(figsize=(12,6))
for lev, cum in results.items():
    plt.plot(cum.index, cum.values, label=f'{lev}x')
plt.axhline(1, color='gray', linestyle='--')
plt.title("TFT + Regime Filter Strategy NAV")
plt.xlabel("Date")
plt.ylabel("Net Value")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# -----------------------
# Strategy Performance Metrics
# -----------------------
metrics_results = {}
for L, nav in results.items():
    metrics_results[L] = calculate_metrics(nav)

metrics_df = pd.DataFrame(metrics_results).T
metrics_df.index.name = "Leverage"
print("\n===== Strategy Performance Metrics =====")
metrics_df.round(4)

# ================================
# Chapter 5: Bayesian Network — CPU-based, Downsampled, Monthly Backtest
# ================================
# ---------------------------
# 1️- Recent Data (limit memory load)
# ---------------------------
end_date = data.index.max()
start_date = end_date - pd.DateOffset(months=96)
data_recent = data.loc[start_date:end_date].copy()

# Downsample to reduce noise
sample_minute = 10
data_resampled = data_recent.resample(f"{sample_minute}min").last().dropna()

# ---------------------------
# 2️- Feature Discretization
# ---------------------------
feature_cols = ['log_ret', 'volatility', 'US_GDP', 'Median_CPI', '10Y_Treasury_FF_Spread']
data_disc = data_resampled[feature_cols].copy()

# Discretize each feature into 4 quantile bins
kb = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
data_disc[:] = kb.fit_transform(data_disc)

# ---------------------------
# 3️-Learn Bayesian Network Structure
# ---------------------------
hc = HillClimbSearch(data_disc)
best_model = hc.estimate(scoring_method="bdeu")  # use string to avoid import issues

bn_model = DiscreteBayesianNetwork(best_model.edges())
bn_model.fit(data_disc)

print("Learned BN edges:")
print(bn_model.edges())

# ---------------------------
# 4️- Predict Return Signals
# ---------------------------
# Probability-based strategy:
# if P(high log_ret bin) > 0.5 → Long, else Short
target_col = 'log_ret'
y_pred = []

for idx in range(len(data_disc)):
    evidence = data_disc.iloc[idx].to_dict()
    # Remove target column from evidence
    evidence.pop(target_col)
    try:
        probs = bn_model.predict_probability(evidence)
        # Probability of the highest return bin
        prob_highest = probs[target_col].iloc[0, -1]
    except Exception:
        prob_highest = 0.5  # neutral fallback
    y_pred.append(prob_highest)

y_pred = np.array(y_pred)

# Trading signal: +1 (long) or -1 (short)
data_signals = data_resampled.copy()
data_signals['BN_signal'] = 0
data_signals.loc[y_pred > 0.5, 'BN_signal'] = 1
data_signals.loc[y_pred <= 0.5, 'BN_signal'] = -1

# ---------------------------
# 5️- Leveraged Returns & Cumulative NAV
# ---------------------------
leverage = 5
data_signals['leveraged_ret'] = leverage * data_signals['BN_signal'].shift(1) * data_signals['log_ret']

cumret_daily = data_signals['leveraged_ret'].resample('D').sum().cumsum().apply(np.exp)
cumret_monthly = cumret_daily.resample('ME').last().dropna()

# ---------------------------
# 6️- Performance Metrics
# ---------------------------
final_value = float(cumret_monthly.iloc[-1])
monthly_returns = cumret_monthly.pct_change().dropna()
years = len(cumret_monthly) / 12
cagr = final_value ** (1/years) - 1
annual_vol = monthly_returns.std() * np.sqrt(12)
sharpe = (cagr - 0.02) / annual_vol
drawdown = cumret_monthly / cumret_monthly.cummax() - 1
max_drawdown = drawdown.min()
win_rate = (monthly_returns > 0).sum() / len(monthly_returns)

print(f"Final Value: {final_value:.4f}")
print(f"CAGR: {cagr:.2%}, Sharpe: {sharpe:.2f}, Max DD: {max_drawdown:.2%}, "
      f"Monthly Win Rate: {win_rate:.2%}, Annual Vol: {annual_vol:.2%}")

# ---------------------------
# 7️⃣ Visualization
# ---------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1, 1]})

# Net Asset Value
axes[0].plot(cumret_monthly.index, cumret_monthly.values, color='blue', lw=2, label=f'BN ({leverage}x)')
axes[0].axhline(1, color='gray', linestyle='--', lw=1)
axes[0].set_title("Monthly Cumulative Return (Net Value)")
axes[0].set_ylabel("Net Multiple")
axes[0].legend()
axes[0].grid(True)

# Monthly Returns
axes[1].bar(monthly_returns.index, monthly_returns.values,
            color=np.where(monthly_returns > 0, '#2ca02c', '#d62728'))
axes[1].set_title("Monthly Returns")
axes[1].set_ylabel("Return")
axes[1].grid(True)

# Drawdown
axes[2].plot(drawdown.index, drawdown.values, color='maroon', lw=1.5)
axes[2].fill_between(drawdown.index, drawdown.values, 0, where=drawdown < 0, color='maroon', alpha=0.3)
axes[2].set_title("Drawdown (from Peak)")
axes[2].set_ylabel("Drawdown")
axes[2].grid(True)

plt.tight_layout()
plt.show()

# ================================
# Multi-Leverage Performance Evaluation — Bayesian Network Strategy
# ================================

# Leverage list
leverage_list = [5]
results = {}  # Store daily NAV for each leverage level

# Generate daily NAV series for each leverage
for lev in leverage_list:
    leveraged_ret = lev * data_signals['BN_signal'].shift(1) * data_signals['log_ret']
    daily_log_ret = leveraged_ret.resample("D").sum()  # Aggregate log returns daily
    nav = np.exp(daily_log_ret.cumsum())               # Daily cumulative NAV
    results[lev] = nav

# Compute performance metrics
metrics_results = []

for L, nav in results.items():
    if len(nav) == 0:
        print(f"Leverage {L} has empty NAV, skipping.")
        continue
    metrics = calculate_metrics(nav, rf=0.02)  # Annualized risk-free rate = 2%
    metrics['Leverage'] = L
    metrics_results.append(metrics)

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics_results)
metrics_df.set_index('Leverage', inplace=True)

# Format for clean display
metrics_df_formatted = metrics_df.style.format({
    "Cumulative Return": "{:.4f}",
    "Annual Return": "{:.4f}",
    "Annual Volatility": "{:.4f}",
    "Sharpe Ratio": "{:.4f}",
    "Calmar Ratio": "{:.4f}",
    "Omega Ratio": "{:.4f}",
    "Max Drawdown": "{:.4f}"
})

print("===== Bayesian Network FX Strategy Performance Metrics =====")
metrics_df_formatted

# ================================
# Bayesian Network: Full Dataset + Multiple Leverages + Monthly Backtest
# ================================
# ---------------------------
# 1️- Use Full Dataset
# ---------------------------
data_full = data.copy()

# Downsample to 10-minute frequency
sample_minute = 10
data_resampled = data_full.resample(f"{sample_minute}min").last().dropna()

# ---------------------------
# 2️- Feature Discretization
# ---------------------------
feature_cols = ['log_ret', 'volatility', 'US_GDP', 'Median_CPI', '10Y_Treasury_FF_Spread']
data_disc = data_resampled[feature_cols].copy()

# Discretize each feature into 4 quantile bins
kb = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
data_disc[:] = kb.fit_transform(data_disc)

# ---------------------------
# 3️- Learn Bayesian Network Structure
# ---------------------------
hc = HillClimbSearch(data_disc)
best_model = hc.estimate(scoring_method="bdeu")

bn_model = DiscreteBayesianNetwork(best_model.edges())
bn_model.fit(data_disc)

print("Learned BN edges:")
print(bn_model.edges())

# ---------------------------
# 4️- Predict Return Signals
# ---------------------------
target_col = 'log_ret'
y_pred = []

for idx in range(len(data_disc)):
    evidence = data_disc.iloc[idx].to_dict()
    evidence.pop(target_col)
    try:
        probs = bn_model.predict_probability(evidence)
        # Probability that log_ret falls into the highest bin
        prob_highest = probs[target_col].iloc[0, -1]
    except Exception:
        prob_highest = 0.5  # neutral default
    y_pred.append(prob_highest)

y_pred = np.array(y_pred)

# Generate binary signals
data_signals = data_resampled.copy()
data_signals['BN_signal'] = 0
data_signals.loc[y_pred > 0.5, 'BN_signal'] = 1
data_signals.loc[y_pred <= 0.5, 'BN_signal'] = -1

# ---------------------------
# 5️- Multi-Leverage Cumulative NAV Calculation
# ---------------------------
leverages = [1, 2, 5, 10]
cumret_dict = {}

for lev in leverages:
    data_signals[f'leveraged_ret_{lev}x'] = lev * data_signals['BN_signal'].shift(1) * data_signals['log_ret']
    cumret_daily = data_signals[f'leveraged_ret_{lev}x'].resample('D').sum().cumsum().apply(np.exp)
    cumret_dict[lev] = cumret_daily

# ---------------------------
# 6️- Plot Comparison of NAVs
# ---------------------------
plt.figure(figsize=(12,6))
for lev in leverages:
    plt.plot(cumret_dict[lev].index, cumret_dict[lev].values, lw=2, label=f'{lev}x')
plt.axhline(1, color='gray', linestyle='--', lw=1)
plt.title("Bayesian Network Strategy — Cumulative Return Across Leverages")
plt.xlabel("Date")
plt.ylabel("Net Multiple")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# 7️- Performance Metrics
# ---------------------------
for lev in leverages:
    cumret_monthly = cumret_dict[lev].resample('ME').last().dropna()
    final_value = float(cumret_monthly.iloc[-1])
    monthly_returns = cumret_monthly.pct_change().dropna()
    years = len(cumret_monthly) / 12
    cagr = final_value ** (1/years) - 1
    annual_vol = monthly_returns.std() * np.sqrt(12)
    sharpe = (cagr - 0.02) / annual_vol
    drawdown = cumret_monthly / cumret_monthly.cummax() - 1
    max_drawdown = drawdown.min()
    win_rate = (monthly_returns > 0).sum() / len(monthly_returns)

    print(f"\nLeverage {lev}x:")
    print(f"Final Value: {final_value:.4f}")
    print(f"CAGR: {cagr:.2%}, Sharpe: {sharpe:.2f}, Max DD: {max_drawdown:.2%}, "
          f"Monthly Win Rate: {win_rate:.2%}, Annual Vol: {annual_vol:.2%}")

# ================================
# Bayesian Network — Multi-Leverage Performance Metrics
# ================================

# 1️⃣ Define leverage levels and container
leverages = [1, 2, 5, 10]
cumret_dict = {}

# 2️⃣ Generate daily cumulative NAVs
for lev in leverages:
    leveraged_ret = lev * data_signals['BN_signal'].shift(1) * data_signals['log_ret']
    daily_log_ret = leveraged_ret.resample("D").sum()  # aggregate to daily log returns
    nav = np.exp(daily_log_ret.cumsum())               # convert to cumulative NAV
    cumret_dict[lev] = nav

# 3️⃣ Compute performance metrics for each leverage
metrics_results = []

for lev, nav in cumret_dict.items():
    if len(nav) == 0:
        print(f"Leverage {lev}x NAV is empty, skipping.")
        continue
    metrics = calculate_metrics(nav, rf=0.02)  # annualized risk-free rate = 2%
    metrics['Leverage'] = lev
    metrics_results.append(metrics)

# 4️⃣ Convert to DataFrame
metrics_df = pd.DataFrame(metrics_results)
metrics_df.set_index('Leverage', inplace=True)

# 5️⃣ Format for clean display
metrics_df_formatted = metrics_df.style.format({
    "Cumulative Return": "{:.4f}",
    "Annual Return": "{:.4f}",
    "Annual Volatility": "{:.4f}",
    "Sharpe Ratio": "{:.4f}",
    "Calmar Ratio": "{:.4f}",
    "Omega Ratio": "{:.4f}",
    "Max Drawdown": "{:.4f}"
})

print("===== Bayesian Network FX Strategy Performance Metrics =====")
metrics_df_formatted

"""# Chapter 6: Hybrid architecture (Minute-level)

## 6-1 : Hybrid BN + HMM FX Strategy
"""

# ================================
# Hybrid BN + HMM FX Strategy
# ================================
# ---------------------------
# 1️- Select last 24 months and downsample
# ---------------------------
end_date = data.index.max()
start_date = end_date - pd.DateOffset(months=24)
data_recent = data.loc[start_date:end_date].copy()

sample_minute = 20
data_resampled = data_recent.resample(f"{sample_minute}min").last().dropna()

# ---------------------------
# 2️- Bayesian Network: Feature Discretization
# ---------------------------
feature_cols = ['log_ret', 'volatility', 'US_GDP', 'Median_CPI', '10Y_Treasury_FF_Spread']
data_disc = data_resampled[feature_cols].copy()

kb = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
data_disc[:] = kb.fit_transform(data_disc)

# ---------------------------
# 3️- Bayesian Network: Structure Learning
# ---------------------------
hc = HillClimbSearch(data_disc)
best_model = hc.estimate(scoring_method="bdeu")  # use string to avoid import issues

bn_model = DiscreteBayesianNetwork(best_model.edges())
bn_model.fit(data_disc)
print("BN edges:", bn_model.edges())

# ---------------------------
# 4️- Bayesian Network: Predictive Signal
# ---------------------------
target_col = 'log_ret'
y_pred_bn = []

for idx in range(len(data_disc)):
    evidence = data_disc.iloc[idx].to_dict()
    evidence.pop(target_col)
    try:
        probs = bn_model.predict_probability(evidence)
        prob_highest = probs[target_col].iloc[0, -1]  # probability of highest bin
    except Exception:
        prob_highest = 0.5  # neutral default
    y_pred_bn.append(prob_highest)

y_pred_bn = np.array(y_pred_bn)
bn_signal = np.zeros_like(y_pred_bn)
bn_signal[y_pred_bn > 0.5] = 1
bn_signal[y_pred_bn <= 0.5] = -1

# ---------------------------
# 5️- Hidden Markov Model (HMM): Regime Detection
# ---------------------------
hmm_features = data_resampled[['log_ret', 'volatility']].fillna(0).values
hmm_model = GaussianHMM(n_components=2, covariance_type='full', n_iter=200)
hmm_model.fit(hmm_features)
hmm_states = hmm_model.predict(hmm_features)

# Assume state 1 is the "bull" (high-return) regime
hmm_signal = np.where(hmm_states == 1, 1, -1)

# ---------------------------
# 6️- Hybrid Signal: Combine BN and HMM
# ---------------------------
final_signal = bn_signal * hmm_signal
data_signals = data_resampled.copy()
data_signals['Hybrid_signal'] = final_signal

# ---------------------------
# 7️- Backtest with Multiple Leverages
# ---------------------------
leverages = [1, 2, 5, 10]
cumrets_dict = {}

for lev in leverages:
    col_ret = f'leveraged_ret_{lev}x'
    data_signals[col_ret] = lev * data_signals['Hybrid_signal'].shift(1) * data_signals['log_ret']
    cumret_daily = data_signals[col_ret].resample('D').sum().cumsum().apply(np.exp)
    cumrets_dict[lev] = cumret_daily

# ---------------------------
# 8️- Performance Metrics
# ---------------------------
for lev in leverages:
    monthly = cumrets_dict[lev].resample('ME').last().pct_change().dropna()
    final_val = cumrets_dict[lev].iloc[-1]
    years = len(monthly) / 12
    cagr = final_val ** (1 / years) - 1
    ann_vol = monthly.std() * np.sqrt(12)
    sharpe = (cagr - 0.02) / ann_vol
    max_dd = (cumrets_dict[lev] / cumrets_dict[lev].cummax() - 1).min()
    win_rate = (monthly > 0).sum() / len(monthly)
    print(f"Leverage {lev}x -> Final: {final_val:.4f}, CAGR: {cagr:.2%}, "
          f"Sharpe: {sharpe:.2f}, MaxDD: {max_dd:.2%}, Monthly WinRate: {win_rate:.2%}")

# ---------------------------
# 9️- Plot Results
# ---------------------------
fig, ax = plt.subplots(figsize=(12, 6))
for lev in leverages:
    ax.plot(cumrets_dict[lev].index, cumrets_dict[lev].values, lw=2, label=f'{lev}x leverage')
ax.set_title("Hybrid BN + HMM FX Strategy — Cumulative Net Value\n(20min data, 24 months)", fontsize=13)
ax.set_ylabel("Net Multiple")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# 1️- Compute Multi-Leverage Performance Table
# ---------------------------

metrics_results = []

# Loop over each leverage level and compute key metrics
for lev, nav in cumrets_dict.items():
    metrics = calculate_metrics(nav, rf=0.02)  # annual risk-free rate = 2%
    metrics['Leverage'] = lev
    metrics_results.append(metrics)

# Combine all results into a single DataFrame
metrics_df = pd.DataFrame(metrics_results)
metrics_df.set_index('Leverage', inplace=True)

# Format display for clean presentation
metrics_df_formatted = metrics_df.style.format({
    "Cumulative Return": "{:.4f}",
    "Annual Return": "{:.4f}",
    "Annual Volatility": "{:.4f}",
    "Sharpe Ratio": "{:.4f}",
    "Calmar Ratio": "{:.4f}",
    "Omega Ratio": "{:.4f}",
    "Max Drawdown": "{:.4f}"
})

print("===== Hybrid BN + HMM FX Strategy Performance Metrics =====")
metrics_df_formatted

# ================================
# Hybrid TFT + HMM FX Strategy
# ================================

# ---------------------------
# 1️- Data Preprocessing — Last 24 Months
# ---------------------------
end = data.index.max()
start = end - pd.DateOffset(months=24)
df = data.loc[start:end].copy()
df = df.resample("20min").last().dropna()

lookback = 30
features = ['log_ret', 'volatility', 'US_GDP', 'Median_CPI', '10Y_Treasury_FF_Spread']

# ---------------------------
# 2️- Build TFT Input Sequences
# ---------------------------
X, y = [], []
for i in range(lookback, len(df)):
    X.append(df[features].iloc[i - lookback:i].values)
    y.append(df['log_ret'].iloc[i])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

X_tft = torch.tensor(X)
y_tft = torch.sign(torch.tensor(y)).float().unsqueeze(1)  # +1 for up, -1 for down

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------------------
# 3️- Define Temporal Fusion Transformer (TFT)
# ---------------------------
class TFT(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_heads=4):
        super().__init__()
        self.var_proj = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.lstm_enc = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm_dec = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        gate = self.gate(x[:, -1, :])
        x = x * gate.unsqueeze(1)
        x = self.var_proj(x)
        enc_out, (h, c) = self.lstm_enc(x)
        dec_out, _ = self.lstm_dec(x, (h, c))
        attn_out, _ = self.attn(dec_out, enc_out, enc_out)
        out = self.fc(attn_out[:, -1, :])
        return out

# ---------------------------
# 4️- Train TFT
# ---------------------------
model = TFT(input_dim=X.shape[2]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
batch_size = 64
epochs = 5

for e in range(epochs):
    perm = torch.randperm(len(X_tft))
    total_loss = 0
    for i in range(0, len(X_tft), batch_size):
        idx = perm[i:i+batch_size]
        xb = X_tft[idx].to(device)
        yb = y_tft[idx].to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {e+1}, Loss={total_loss:.4f}")

# ---------------------------
# 5️- TFT Prediction
# ---------------------------
with torch.no_grad():
    y_pred_tft = []
    for i in range(0, len(X_tft), 2048):
        y_pred_tft.extend(model(X_tft[i:i+2048].to(device)).cpu().flatten().numpy())
y_pred_tft = np.array(y_pred_tft)

tft_signal = np.where(y_pred_tft > 0, 1, -1)
df_sig = df.iloc[lookback:].copy()
df_sig["tft_sig"] = tft_signal

# ---------------------------
# 6️- HMM Regime Filter
# ---------------------------
hmm_features = df_sig[["log_ret", "volatility"]].fillna(0).values
# Add minimal noise to prevent covariance singularities
hmm_features += np.random.normal(0, 1e-6, hmm_features.shape)

hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
hmm_states = hmm_model.fit(hmm_features).predict(hmm_features)
df_sig["hmm_state"] = hmm_states

# Only trade in regime=1 (presumed bull state)
df_sig["hybrid_sig"] = df_sig["tft_sig"]
df_sig.loc[df_sig["hmm_state"] == 0, "hybrid_sig"] = 0

# ---------------------------
# 7️-Leverage Returns & Cumulative NAV
# ---------------------------
levs = [1, 2, 5, 10]
results = {}

df_sig["hybrid_ret"] = df_sig["hybrid_sig"].shift() * df_sig["log_ret"]

for L in levs:
    nav = (L * df_sig["hybrid_ret"]).resample("D").sum().cumsum().apply(np.exp)
    results[L] = nav

# ---------------------------
# 8️- Plot Cumulative Returns
# ---------------------------
plt.figure(figsize=(10, 6))
for L, nav in results.items():
    plt.plot(nav.index, nav, label=f'{L}x', linewidth=2)
plt.axhline(1, color='gray', linestyle='--', linewidth=1)
plt.title("Hybrid TFT + HMM FX Strategy — Cumulative Net Value\n(20min data, 24 months, lookback=30)", fontsize=13)
plt.xlabel("Date", fontsize=11)
plt.ylabel("Net Value (cumulative)", fontsize=11)
plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.legend(title="Leverage", fontsize=10)
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()

# ---------------------------
#  Compute Multi-Leverage Performance Table
# ---------------------------

metrics_results = []

# Loop over each leverage level to compute performance metrics
for lev, nav in results.items():
    metrics = calculate_metrics(nav, rf=0.02)  # annual risk-free rate = 2%
    metrics['Leverage'] = lev
    metrics_results.append(metrics)

# Combine all leverage results into one DataFrame
metrics_df = pd.DataFrame(metrics_results)
metrics_df.set_index('Leverage', inplace=True)

# Format table for presentation
metrics_df_formatted = metrics_df.style.format({
    "Cumulative Return": "{:.4f}",
    "Annual Return": "{:.4f}",
    "Annual Volatility": "{:.4f}",
    "Sharpe Ratio": "{:.4f}",
    "Calmar Ratio": "{:.4f}",
    "Omega Ratio": "{:.4f}",
    "Max Drawdown": "{:.4f}"
})

print("===== Hybrid TFT + HMM FX Strategy Performance Metrics =====")
metrics_df_formatted

"""## 6-2: Hybrid TFT + Bayesian Network FX Strategy"""

# ================================
# Hybrid TFT + Bayesian Network FX Strategy
# ================================
# ---------------------------
# 1️- Data selection & preprocessing
# ---------------------------
end = data.index.max()
start = end - pd.DateOffset(months=24)
df = data.loc[start:end].copy()
df_resampled = df.resample("20min").last().dropna()

lookback = 30
features = ['log_ret','volatility','US_GDP','Median_CPI','10Y_Treasury_FF_Spread']

# ---------------------------
# 2️- Bayesian Network
# ---------------------------
data_disc = df_resampled[features].copy()
kb = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
data_disc[:] = kb.fit_transform(data_disc)

hc = HillClimbSearch(data_disc)
best_model = hc.estimate(scoring_method="bdeu")
bn_model = DiscreteBayesianNetwork(best_model.edges())
bn_model.fit(data_disc)

# BN prediction
bn_signal = []
target_col = 'log_ret'
for idx in range(len(data_disc)):
    evidence = data_disc.iloc[idx].to_dict()
    evidence.pop(target_col)
    try:
        probs = bn_model.predict_probability(evidence)
        prob_highest = probs[target_col].iloc[0, -1]
    except Exception:
        prob_highest = 0.5
    bn_signal.append(1 if prob_highest>0.5 else -1)
bn_signal = np.array(bn_signal)

# ---------------------------
# 3️-  TFT Model
# ---------------------------
class TFT(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_heads=4):
        super().__init__()
        self.var_proj = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.lstm_enc = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm_dec = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        gate = self.gate(x[:, -1, :])
        x = x * gate.unsqueeze(1)
        x = self.var_proj(x)
        enc_out, (h, c) = self.lstm_enc(x)
        dec_out, _ = self.lstm_dec(x, (h, c))
        attn_out, _ = self.attn(dec_out, enc_out, enc_out)
        out = self.fc(attn_out[:, -1, :])
        return out

# Prepare TFT data
X, y = [], []
for i in range(lookback, len(df_resampled)):
    X.append(df_resampled[features].iloc[i-lookback:i].values)
    y.append(df_resampled['log_ret'].iloc[i])
X = np.array(X, dtype=np.float32)
y = np.sign(np.array(y, dtype=np.float32))  # ±1

X = torch.tensor(X)
y = torch.tensor(y).float().unsqueeze(1)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = TFT(input_dim=X.shape[2]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
batch = 64
epochs = 5

for e in range(epochs):
    perm = torch.randperm(len(X))
    total = 0
    for i in range(0, len(X), batch):
        idx = perm[i:i+batch]
        xb = X[idx].to(device)
        yb = y[idx].to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {e+1}, Loss={total:.4f}")

# TFT prediction
with torch.no_grad():
    y_pred_tft = []
    for i in range(0, len(X), 2048):
        y_pred_tft.extend(model(X[i:i+2048].to(device)).cpu().flatten().numpy())
y_pred_tft = np.array(y_pred_tft)
y_pred_tft_sig = np.sign(y_pred_tft)

# ---------------------------
# 4️- Hybrid Signal
# ---------------------------
# Align BN signal
bn_signal_aligned = bn_signal[lookback:]
# Weighted hybrid (TFT 70%, BN 30%)
hybrid_signal = 0.7*y_pred_tft_sig + 0.3*bn_signal_aligned
hybrid_signal = np.sign(hybrid_signal)

df_sig = df_resampled.iloc[lookback:].copy()
df_sig['hybrid_sig'] = hybrid_signal
df_sig['ret'] = df_sig['hybrid_sig'].shift() * df_sig['log_ret']

# ---------------------------
# 5️-  Leverage & Cumulative Net Value
# ---------------------------
levs=[1,2,5,10]
results={}
for L in levs:
    nav = (L*df_sig['ret']).resample("D").sum().cumsum().apply(np.exp)
    results[L] = nav

# ---------------------------
# 6️-  Plot
# ---------------------------
plt.figure(figsize=(10,6))
for L, nav in results.items():
    plt.plot(nav.index, nav, label=f'{L}x', linewidth=2)
plt.axhline(1, color='gray', linestyle='--', linewidth=1)
plt.title("Hybrid TFT + BN FX Strategy - Cumulative Net Value\n(20min data, 24 months, lookback=30)", fontsize=13)
plt.xlabel("Date", fontsize=11)
plt.ylabel("Net Value (cumulative)", fontsize=11)
plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.legend(title="Leverage", fontsize=10)
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()

# ---------------------------
# Compute Multi-Leverage Performance Metrics Table
# ---------------------------

metrics_results = []

# Loop over each leverage level and compute performance metrics
for lev, nav in results.items():
    metrics = calculate_metrics(nav, rf=0.02)  # annual risk-free rate = 2%
    metrics['Leverage'] = lev
    metrics_results.append(metrics)

# Combine all leverage-level metrics into a single DataFrame
metrics_df = pd.DataFrame(metrics_results)
metrics_df.set_index('Leverage', inplace=True)

# Format table for clear presentation
metrics_df_formatted = metrics_df.style.format({
    "Cumulative Return": "{:.4f}",
    "Annual Return": "{:.4f}",
    "Annual Volatility": "{:.4f}",
    "Sharpe Ratio": "{:.4f}",
    "Calmar Ratio": "{:.4f}",
    "Omega Ratio": "{:.4f}",
    "Max Drawdown": "{:.4f}"
})

print("===== Hybrid TFT + BN FX Strategy Performance Metrics =====")
metrics_df_formatted

"""## 6-3: Hybrid TFT + HMM + BN FX Strategy"""

# ============================
# Hybrid TFT + BN + HMM FX Strategy
# ============================
# ---------------------------
# 1️- Data Preparation
# ---------------------------
end_date = data.index.max()
start_date = end_date - pd.DateOffset(months=24)  # last 24 months
df = data.loc[start_date:end_date].copy()
df = df.resample("20min").last().dropna()

lookback = 30
features = ['log_ret', 'volatility', 'US_GDP', 'Median_CPI', '10Y_Treasury_FF_Spread']

# ---------------------------
# 2️- TFT Model — Transformer Forecast Signal
# ---------------------------
class TFT(nn.Module):
    """A compact Temporal Fusion Transformer with LSTM encoder-decoder and multihead attention."""
    def __init__(self, input_dim, hidden_dim=32, num_heads=4):
        super().__init__()
        self.var_proj = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.lstm_enc = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm_dec = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        gate = self.gate(x[:, -1, :])
        x = x * gate.unsqueeze(1)
        x = self.var_proj(x)
        enc_out, (h, c) = self.lstm_enc(x)
        dec_out, _ = self.lstm_dec(x, (h, c))
        attn_out, _ = self.attn(dec_out, enc_out, enc_out)
        out = self.fc(attn_out[:, -1, :])
        return out

# Build training data
X, y = [], []
for i in range(lookback, len(df)):
    X.append(df[features].iloc[i - lookback:i].values)
    y.append(df['log_ret'].iloc[i])
X = torch.tensor(np.array(X, dtype=np.float32))
y = torch.sign(torch.tensor(np.array(y, dtype=np.float32))).float().unsqueeze(1)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = TFT(input_dim=X.shape[2]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
batch, epochs = 64, 5

for e in range(epochs):
    perm = torch.randperm(len(X))
    total = 0
    for i in range(0, len(X), batch):
        idx = perm[i:i + batch]
        xb = X[idx].to(device)
        yb = y[idx].to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {e+1}, Loss={total:.4f}")

with torch.no_grad():
    y_pred = []
    for i in range(0, len(X), 2048):
        y_pred.extend(model(X[i:i + 2048].to(device)).cpu().flatten().numpy())
y_pred = np.array(y_pred)
tft_signal = np.where(y_pred > 0, 1, -1)

# ---------------------------
# 3️- Bayesian Network — Probabilistic Macro Dependency Model
# ---------------------------
data_disc = df[features].copy()
kb = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
data_disc[:] = kb.fit_transform(data_disc)

hc = HillClimbSearch(data_disc)
best_model = hc.estimate(scoring_method="bdeu")
bn_model = DiscreteBayesianNetwork(best_model.edges())
bn_model.fit(data_disc)

bn_signal = []
target_col = 'log_ret'
for idx in range(len(data_disc)):
    evidence = data_disc.iloc[idx].to_dict()
    evidence.pop(target_col)
    try:
        probs = bn_model.predict_probability(evidence)
        prob_highest = probs[target_col].iloc[0, -1]
    except:
        prob_highest = 0.5
    bn_signal.append(1 if prob_highest > 0.5 else -1)
bn_signal = np.array(bn_signal)

# ---------------------------
# 4️- Hidden Markov Model — Regime Detection
# ---------------------------
returns = df['log_ret'].values.reshape(-1, 1)
hmm_model = GaussianHMM(n_components=2, covariance_type='full', n_iter=200)
hmm_model.fit(returns)
hmm_states = hmm_model.predict(returns)
hmm_signal = np.where(hmm_states == hmm_states.max(), 1, -1)

# ---------------------------
# 5️- Align and Combine All Signals
# ---------------------------
min_len = min(len(tft_signal), len(bn_signal), len(hmm_signal))
tft_signal = tft_signal[-min_len:]
bn_signal = bn_signal[-min_len:]
hmm_signal = hmm_signal[-min_len:]
df_aligned = df.iloc[-min_len:].copy()

# Weighted ensemble — Transformer (50%), Bayesian Network (30%), HMM (20%)
hybrid_signal = 0.5 * tft_signal + 0.3 * bn_signal + 0.2 * hmm_signal
hybrid_signal = np.sign(hybrid_signal)
df_aligned['signal'] = hybrid_signal
df_aligned['ret'] = df_aligned['signal'].shift() * df_aligned['log_ret']

# ---------------------------
# 6️- Backtest Under Multiple Leverage Levels
# ---------------------------
levs = [1, 2, 5, 10]
results = {}
for L in levs:
    nav = (L * df_aligned['ret']).resample('D').sum().cumsum().apply(np.exp)
    results[L] = nav

# ---------------------------
# 7️- Visualization
# ---------------------------
plt.figure(figsize=(12, 6))
for L, nav in results.items():
    plt.plot(nav.index, nav, label=f'{L}x', linewidth=2)

plt.axhline(1, color='gray', linestyle='--', linewidth=1)
plt.title("Hybrid TFT + BN + HMM FX Strategy\n(20min data, 24 months, lookback=30)", fontsize=13)
plt.xlabel("Date", fontsize=11)
plt.ylabel("Net Value (cumulative)", fontsize=11)
plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.grid(alpha=0.4)
plt.legend(title="Leverage", fontsize=10)
plt.tight_layout()
plt.show()

# ============================
# Performance Metrics Calculation
# ============================

def calculate_metrics(nav_series, rf=0.02):
    """
    Parameters:
    ----------
    nav_series : pd.Series
        Daily cumulative net asset value (NAV).
    rf : float, default 0.02
        Annualized risk-free rate.

    Returns:
    -------
    dict
        Dictionary containing key performance metrics.
    """
    # Daily returns
    daily_ret = nav_series.pct_change().dropna()

    # Annualized return and volatility
    years = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
    cum_ret = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    annual_ret = (1 + cum_ret) ** (1 / years) - 1
    annual_vol = daily_ret.std() * np.sqrt(252)

    # Sharpe Ratio
    sharpe = (annual_ret - rf) / annual_vol if annual_vol > 0 else np.nan

    # Maximum drawdown
    rolling_max = nav_series.cummax()
    drawdown = nav_series / rolling_max - 1
    max_dd = drawdown.min()

    # Calmar Ratio
    calmar = annual_ret / abs(max_dd) if max_dd < 0 else np.nan

    # Omega Ratio (threshold = dailyized rf)
    threshold = (1 + rf) ** (1 / 252) - 1
    gains = daily_ret[daily_ret > threshold] - threshold
    losses = threshold - daily_ret[daily_ret <= threshold]
    omega = gains.sum() / losses.sum() if losses.sum() > 0 else np.nan

    return {
        "Cumulative Return": cum_ret,
        "Annual Return": annual_ret,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Calmar Ratio": calmar,
        "Omega Ratio": omega,
        "Max Drawdown": max_dd
    }

# ---------------------------
# Compute and Display Metrics for Each Leverage Level
# ---------------------------
metrics_results = {}
for L, nav in results.items():
    metrics_results[L] = calculate_metrics(nav)

# Convert to DataFrame for comparison
metrics_df = pd.DataFrame(metrics_results).T
metrics_df.index.name = "Leverage"

print("\n===== Hybrid TFT + BN + HMM FX Strategy Performance Metrics =====")
print(metrics_df.round(4))

def calculate_metrics(nav, rf=0.02):
    """
    Compute strategy performance metrics.
    nav : pd.Series, cumulative NAV series (daily)
    rf : float, annualized risk-free rate (default 2%)

    Returns a dict with:
    - Cumulative Return
    - Annual Return
    - Annual Volatility
    - Sharpe Ratio
    - Calmar Ratio
    - Omega Ratio
    - Max Drawdown
    """
    # Daily returns
    daily_ret = nav.pct_change().dropna()

    # Cumulative return
    cum_return = nav.iloc[-1] / nav.iloc[0] - 1

    # Annualized return and volatility
    periods_per_year = 252  # assume daily
    ann_ret = (1 + cum_return) ** (periods_per_year / len(daily_ret)) - 1
    ann_vol = daily_ret.std() * np.sqrt(periods_per_year)

    # Sharpe Ratio
    sharpe = (ann_ret - rf) / ann_vol if ann_vol != 0 else np.nan

    # Maximum drawdown
    drawdown = nav / nav.cummax() - 1
    max_dd = drawdown.min()

    # Calmar Ratio
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    # Omega Ratio (threshold = 0)
    gains = daily_ret[daily_ret > 0].sum()
    losses = -daily_ret[daily_ret < 0].sum()  # make positive
    omega = gains / losses if losses != 0 else np.nan

    return {
        "Cumulative Return": cum_return,
        "Annual Return": ann_ret,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Calmar Ratio": calmar,
        "Omega Ratio": omega,
        "Max Drawdown": max_dd
    }

# ===== Example: compute metrics for different leverages =====
metrics_results = {}
for L, nav in results.items():  # results is your dict {1x: nav1, 2x: nav2, ...}
    metrics_results[L] = calculate_metrics(nav)

# Output DataFrame
metrics_df = pd.DataFrame(metrics_results).T
metrics_df.index.name = "Leverage"
print("\n===== Strategy Performance Metrics =====")
print(metrics_df.round(4))

def calculate_metrics(nav, rf=0.02):
    """
    Compute strategy metrics
    nav: pd.Series, NAV curve (cumulative NAV)
    rf: Annualized risk-free rate
    Returns: dict
    """
    # Ensure nav is in ascending date order
    nav = nav.sort_index()

    # -----------------------
    # 1️- Daily returns
    # -----------------------
    daily_ret = nav.pct_change().dropna()

    # -----------------------
    # 2️-Annualized return & volatility
    # -----------------------
    years = (nav.index[-1] - nav.index[0]).days / 365
    cumulative_return = nav.iloc[-1]/nav.iloc[0] - 1
    annual_return = (nav.iloc[-1]/nav.iloc[0])**(1/years) - 1
    annual_vol = daily_ret.std() * np.sqrt(252)

    # -----------------------
    # 3️- Max drawdown & Calmar Ratio
    # -----------------------
    cum_max = nav.cummax()
    drawdown = (nav - cum_max) / cum_max
    max_dd = drawdown.min()
    calmar = annual_return / abs(max_dd) if max_dd != 0 else np.nan

    # -----------------------
    # 4️- Sharpe Ratio
    # -----------------------
    sharpe = (annual_return - rf) / annual_vol if annual_vol != 0 else np.nan

    # -----------------------
    # 5️- Omega Ratio
    # -----------------------
    threshold = rf/252
    omega = daily_ret[daily_ret>threshold].sum() / abs(daily_ret[daily_ret<=threshold].sum()) if abs(daily_ret[daily_ret<=threshold].sum())>0 else np.nan

    return {
        "Cumulative Return": cumulative_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Calmar Ratio": calmar,
        "Omega Ratio": omega,
        "Max Drawdown": max_dd
    }

metrics_results = {}
for L, nav in results.items():   # results is your dict of leveraged NAV series
    metrics_results[L] = calculate_metrics(nav)

metrics_df = pd.DataFrame(metrics_results).T
metrics_df.index.name = "Leverage"
print("\n===== Strategy Performance Metrics =====")
metrics_df.round(4)

"""### ------ End of script---------"""
