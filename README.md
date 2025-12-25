# Trading-of-Euro-and-USD-with-hybrrid-models
Forecasting EUR/USD using individual and hybrid ML + probabilistic models on high-frequency data (2018–2025). Walk-forward validation under 1x–10x leverage. Results show BN+HMM hybrids deliver the best risk-adjusted returns, reduced drawdowns, and improved stability versus ARIMA, GARCH and standalone ML.

## EUR/USD Hybrid Machine Learning Trading Models

This project evaluates individual and hybrid machine learning and probabilistic models for forecasting EUR/USD price direction and returns under leveraged swing-trading conditions using high-frequency data from 2018–2025.

The goal is to identify models that maximize risk-adjusted performance while maintaining stability in volatile FX markets.

### Models Implemented

Baseline Econometric Models

- ARIMA

- GARCH

Used to confirm non-stationarity and nonlinear dynamics of FX prices.

## Machine Learning Models

- Temporal Fusion Transformer (TFT)

- Other ML architectures for regression & classification

## Probabilistic Models

- Hidden Markov Models (HMM – GaussianHMM)

- Bayesian Networks (BN – pgmpy)

## Hybrid Architectures

- BN + HMM

- BH + HMM

- All-in-one stacked hybrids combining ML + probabilistic layers

## Trading Framework

EUR/USD high-frequency price data

Macroeconomic indicators

Walk-forward validation

Leverage scenarios: 1x, 2x, 5x, 10x

## Performance Metrics

Each model is evaluated using:

- Annualized Return

- Annualized Volatility

- Sharpe Ratio

- Calmar Ratio

- Omega Ratio

- Maximum Drawdown

These metrics measure **profitability**, **stability**, and **downside risk**.

## Key Findings
Model Type	Result
ARIMA / GARCH	Fail to predict direction or price reliably
Standalone ML	Predicts price & direction but unstable under leverage
Hybrid Models	Maintain forecast accuracy while reducing drawdown
BN + HMM	Best overall balance of profitability & stability
BH + HMM	Highest risk-adjusted returns

Hybrid probabilistic-deep learning architectures significantly improve robustness in volatile FX environments.


## Conclusion

This study demonstrates that integrating probabilistic models (Bayesian Networks + Hidden Markov Models) into ML trading systems significantly improves risk-adjusted returns and drawdown control in leveraged EUR/USD swing trading.

The BN + HMM hybrid model is the most reliable candidate for real-world deployment.
