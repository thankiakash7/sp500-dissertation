# Final Dissertation Results Summary

> **⚠️ Methodology note — read before using these numbers.** The tables
> below were generated from a run in which the classical models (mean
> baseline, ARMA, AR(1)-GARCH-t) were evaluated walk-forward (retrained
> every 21 trading days) but the deep learning models (LSTM, GRU, TFT)
> used a **static train/test split** — fit once on data through 2018 and
> evaluated straight through 2020-2024, not retrained. This is a
> methodological inconsistency, not a footnote: the deep learning models
> had an easier task (no requirement to adapt to unseen regimes) than the
> classical models did. The pipeline has since been fixed so all six
> models are genuinely walk-forward — see `CHANGELOG.md` for what changed.
> **The numbers in this document need to be regenerated from a corrected
> run before they go in the dissertation.** They're kept here as a
> reference point for what the corrected numbers should roughly resemble
> in shape, not as final results. Any sentence below that says all six
> models used "an identical walk-forward evaluation framework" describes
> what was intended, not what this particular run actually did.

# 1. Dataset

## CRSP Dataset

The study uses daily stock return data obtained from the CRSP database covering S&P 500 constituent companies between 2010 and 2024.

Dataset characteristics:

* **Source:** CRSP daily stock return database
* **Period:** 4 January 2010 – 31 December 2024
* **Number of observations:** 1,711,517 daily stock observations
* **Number of unique stocks:** 740 PERMNO identifiers
* **Market universe:** S&P 500 constituent companies
* **Training period:** 2010–2018
* **Validation period:** 2019
* **Out-of-sample testing period:** 2020–2024

Membership filtering and data quality checks were applied before modelling.

Data audit results:

| Check                                         | Result |
| --------------------------------------------- | -----: |
| Duplicate PERMNO-date observations            |      0 |
| Rows outside valid S&P 500 membership periods |      0 |
| Negative market capitalisation values         |      0 |
| Returns below -100%                           |      0 |
| Missing combined returns                      |     60 |
| Delisting return observations (DLRET)         |    126 |
| Unmatched delisting records                   |      0 |

CRSP delisting returns were incorporated where available to ensure that extreme events and delisted securities were appropriately represented.

---

# 2. Exploratory Data Analysis

Exploratory analysis was conducted to examine return characteristics, volatility behaviour, and statistical properties of stock returns.

Key observations:

* Daily stock returns demonstrated very low linear autocorrelation, with lag-1 autocorrelation values generally close to zero across examined securities.
* Squared returns exhibited significant autocorrelation, indicating volatility clustering and supporting the application of conditional volatility models such as GARCH.
* Return distributions displayed substantial excess kurtosis, with Jarque-Bera tests strongly rejecting normality for all examined stocks.
* The presence of heavy tails and volatility clustering highlights the difficulty of forecasting daily equity returns using traditional statistical assumptions.

These characteristics motivated the comparison between classical time-series approaches and nonlinear deep learning architectures.

---

# 3. Forecasting Model Results

The study evaluates six forecasting approaches over the 2020–2024 testing
period. The classical models (mean baseline, ARMA, AR(1)-GARCH-t) were
evaluated using a walk-forward methodology (retrained every 21 trading
days). **In this run, the deep learning models (LSTM, GRU, TFT) instead
used a static train/test split** (fit once on data through 2018-12-31,
evaluated without retraining through 2020-2024) — see the methodology
note at the top of this document.

## Classical Benchmarks

The classical forecasting models include:

* Mean return benchmark
* ARMA model
* AR(1)-GARCH-t model

## Deep Learning Models

The neural forecasting models include:

* Long Short-Term Memory Network (LSTM)
* Gated Recurrent Unit Network (GRU)
* Temporal Fusion Transformer (TFT)

All models were evaluated using out-of-sample daily return forecasts.
The classical models' forecasts came from a walk-forward methodology; the
deep learning models' forecasts in this run came from a static split
(see the methodology note at the top of this document) rather than the
walk-forward methodology used elsewhere in this summary.

---

# Forecast Accuracy Comparison

| Model         |     RMSE |      MAE | Directional Accuracy |
| ------------- | -------: | -------: | -------------------: |
| LSTM          | 0.020519 | 0.013539 |               51.75% |
| GRU           | 0.020547 | 0.013547 |               51.51% |
| TFT           | 0.020554 | 0.013554 |               51.67% |
| Mean Baseline | 0.020652 | 0.013572 |               51.70% |
| ARMA          | 0.020677 | 0.013609 |               51.10% |
| AR(1)-GARCH-t | 0.020677 | 0.013609 |               51.10% |

---

## Interpretation

Deep learning models achieved marginally lower forecasting errors compared with classical benchmarks.

The LSTM model produced the lowest RMSE and MAE values, followed closely by GRU and TFT. However, the differences between models were small:

* Best RMSE (LSTM): 0.020519
* Worst RMSE (ARMA/GARCH): 0.020677

The difference between the strongest and weakest models was approximately 0.76%.

Directional accuracy remained close to 50% for all approaches, suggesting that although nonlinear models captured additional patterns in return behaviour, the predictive signal remained weak.

Overall, the results indicate that increased model complexity provides only limited improvements when forecasting daily stock returns.

---

# 4. Model Ranking Summary

Based on forecasting accuracy:

1. LSTM achieved the strongest predictive performance.
2. GRU and TFT produced similar forecasting accuracy.
3. The mean baseline remained competitive despite its simplicity.
4. ARMA and AR(1)-GARCH-t produced the weakest forecasting performance.

The relatively small differences between models highlight the difficulty of extracting consistent predictive information from daily equity returns.

---

# 5. Statistical Significance Tests

## Diebold-Mariano Test

The Diebold-Mariano (DM) test was applied to determine whether forecasting performance differences between each model and the mean return benchmark were statistically significant.

A Newey-West HAC correction was applied to account for possible autocorrelation in forecast errors.

| Model         | DM Statistic | p-value | Result                    |
| ------------- | -----------: | ------: | ------------------------- |
| AR(1)-GARCH-t |       0.0506 |  0.9597 | No significant difference |
| ARMA          |       0.0513 |  0.9592 | No significant difference |
| GRU           |       1.5602 |  0.1202 | No significant difference |
| LSTM          |       1.1654 |  0.2452 | No significant difference |
| TFT           |       1.6747 |  0.0955 | No significant difference |

---

## Interpretation

Although deep learning models achieved slightly lower forecasting errors, the Diebold-Mariano tests indicate that these improvements are not statistically significant at the 5% significance level.

The TFT model produced the strongest statistical evidence among the neural models with a p-value of 0.0955, suggesting weak evidence of improved predictive performance at the 10% level.

However, the null hypothesis of equal predictive accuracy cannot be rejected.

Overall, the findings suggest that nonlinear models may capture additional return patterns, but these improvements are small relative to daily market noise.

---

# Model Confidence Set Test

The Model Confidence Set procedure was used to identify whether any forecasting models could be statistically eliminated from the superior model set.

| Model         | p-value |
| ------------- | ------: |
| Mean Baseline |  0.4014 |
| AR(1)-GARCH-t |  0.4014 |
| ARMA          |  0.4014 |
| GRU           |  0.6638 |
| TFT           |  0.6638 |
| LSTM          |  1.0000 |

## Interpretation

All evaluated models remained within the superior model confidence set.

This indicates that no model demonstrated statistically significant superiority over the alternatives under the chosen confidence level.

---

# 6. Portfolio Evaluation

## Portfolio Construction

To evaluate whether forecasting improvements translated into economic value, predicted returns were converted into long-short investment portfolios.

Portfolio construction procedure:

1. Stocks were ranked daily according to model-generated return forecasts.
2. The highest-ranked stocks formed the long portfolio.
3. The lowest-ranked stocks formed the short portfolio.
4. Portfolio returns were calculated using subsequent realised returns.
5. Performance was evaluated under transaction costs of 0, 5, 10 and 25 basis points.

Performance metrics included:

* Annualised return
* Annualised volatility
* Sharpe ratio
* Maximum drawdown
* Portfolio turnover

---

# Gross Portfolio Performance (0 bps Transaction Costs)

| Model         | Annual Return | Volatility | Sharpe Ratio | Maximum Drawdown |
| ------------- | ------------: | ---------: | -----------: | ---------------: |
| LSTM          |        16.68% |     21.70% |        0.769 |          -40.64% |
| GRU           |        18.18% |     24.55% |        0.740 |          -23.97% |
| TFT           |        14.63% |     24.24% |        0.604 |          -37.36% |
| Mean Baseline |        13.64% |     28.28% |        0.482 |          -55.29% |
| ARMA          |         6.29% |     17.85% |        0.352 |          -23.74% |
| AR(1)-GARCH-t |         5.81% |     17.86% |        0.325 |          -23.74% |

---

# Transaction Cost Impact

After applying realistic transaction costs, deep learning portfolio performance deteriorated substantially.

At 5 basis points:

| Model         | Annual Return | Sharpe Ratio |
| ------------- | ------------: | -----------: |
| Mean Baseline |        13.45% |        0.476 |
| ARMA          |         4.31% |        0.241 |
| AR(1)-GARCH-t |         3.83% |        0.214 |
| GRU           |        -5.46% |       -0.222 |
| LSTM          |       -15.44% |       -0.713 |
| TFT           |       -15.84% |       -0.654 |

---

## Interpretation

Although deep learning models generated stronger gross portfolio returns before transaction costs, their higher turnover reduced economic profitability.

The results demonstrate that improved forecasting accuracy does not necessarily translate into superior investment performance.

The simple mean return benchmark remained the strongest strategy after transaction costs, highlighting the importance of trading costs and portfolio implementation constraints.

---

# 7. Fama-French-Carhart Alpha Analysis

Risk-adjusted performance was evaluated using Fama-French-Carhart four-factor regression.

The regression model estimated abnormal portfolio returns after controlling for:

* Market risk
* Size factor
* Value factor
* Momentum factor

---

## Results (5 bps Transaction Costs)

| Model         | Annualised Alpha | Alpha p-value |
| ------------- | ---------------: | ------------: |
| Mean Baseline |            5.30% |         0.451 |
| ARMA          |           -0.16% |         0.982 |
| AR(1)-GARCH-t |           -0.65% |         0.928 |
| GRU           |          -12.00% |         0.191 |
| LSTM          |          -19.76% |         0.025 |
| TFT           |          -21.98% |         0.010 |

---

## Interpretation

The factor regression results provide limited evidence of persistent abnormal returns.

The mean baseline generated positive but statistically insignificant alpha.

Deep learning strategies generated negative risk-adjusted alpha after transaction costs, suggesting that additional portfolio turnover outweighed any predictive advantage.

Therefore, improved forecasting accuracy did not consistently translate into economically significant investment performance.

---

# 8. Overall Conclusion

## Main Research Finding

Deep learning architectures achieved marginal improvements in daily S&P 500 return forecasting accuracy compared with classical time-series models. However, these improvements were not statistically significant and did not consistently translate into superior portfolio performance after transaction costs.

The findings indicate that while nonlinear models such as LSTM, GRU and TFT can capture additional patterns within financial time-series data, the weak predictability of daily equity returns limits their practical advantage over simpler benchmarks.

The results highlight the distinction between statistical forecasting performance and economic usefulness: a model may achieve improved predictive metrics while failing to generate sustainable investment returns once realistic market constraints are considered.
