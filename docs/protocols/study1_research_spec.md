# Research Specification — Locked Pilot Design

## 1. Research question

**Do observable top-of-book and transaction-flow imbalances in BTCUSDT USD-M perpetual futures contain stable out-of-sample information about future mid-price changes?**

The pilot study is deliberately narrow. It prioritizes defensible methodology over model complexity.

## 2. Instrument

- Venue: Binance USD-M futures
- Symbol: BTCUSDT
- Contract type: perpetual futures
- Initial scope: one instrument only

Cross-asset effects are deferred until the single-asset pipeline is validated.

## 3. Information set

At observation time `t`, features may use only information timestamped at or before `t`.

Core top-of-book variables:

- best bid price `b_t`
- best ask price `a_t`
- best bid quantity `q^b_t`
- best ask quantity `q^a_t`
- mid-price `m_t = (a_t + b_t)/2`
- spread `s_t = a_t - b_t`
- relative spread `s_t / m_t`
- book imbalance
- microprice displacement

Transaction variables:

- signed trade quantity
- signed trade notional
- buy/sell trade-flow imbalance
- rolling trade intensity

## 4. Primary features

### Book imbalance

\[
I_t = \frac{q^b_t - q^a_t}{q^b_t + q^a_t}
\]

Range: `[-1, 1]`.

### Microprice

\[
\mu_t =
\frac{a_t q^b_t + b_t q^a_t}
{q^b_t + q^a_t}
\]

The normalized microprice displacement is:

\[
D_t = \frac{\mu_t - m_t}{m_t}
\]

### Signed trades

If `isBuyerMaker = True`, the aggressive side is treated as seller-initiated; otherwise buyer-initiated.

\[
x_i =
\begin{cases}
+q_i, & \text{buyer initiated}\\
-q_i, & \text{seller initiated}
\end{cases}
\]

Rolling trade-flow imbalance will be computed only from past trades.

## 5. Targets

For a horizon `h`:

\[
r_{t,t+h} = \log(m_{t+h}) - \log(m_t)
\]

Primary horizons for the historical pilot:

- 30 seconds
- 60 seconds
- 300 seconds

If event-level archived `bookTicker` data are sufficiently dense and timestamp quality is verified, shorter horizons may be added as a secondary experiment. They are **not pre-committed primary endpoints**.

Direction target:

\[
y_t = 1[r_{t,t+h} > \theta]
\]

where `theta` may initially be zero for research diagnostics. Any trading threshold must be selected using training/validation data only.

## 6. Pre-specified hypotheses

### H1
Book imbalance has statistically detectable predictive association with future return.

### H2
Microprice displacement adds information beyond raw imbalance.

### H3
Trade-flow imbalance adds information beyond top-of-book state.

### H4
Predictive effect weakens as the forecast horizon lengthens.

### H5
Regularized/nonlinear models outperform a simple linear/logistic baseline only modestly, if at all.

### H6
Net economic performance deteriorates materially after spread, fees, slippage, and execution delay.

### H7
Predictive coefficients and/or feature importance differ between volatility/liquidity regimes.

## 7. Baselines

The first models must be:

1. unconditional/no-change baseline;
2. one-feature linear/logistic model;
3. multivariate linear/logistic model;
4. regularized linear/logistic model.

Only after these are established:

5. tree-based gradient boosting;
6. optional small neural network.

## 8. Validation

Forbidden:

- random shuffle split;
- fitting scalers on the entire dataset;
- using future rows in rolling features;
- hyperparameter selection on final test data.

Required:

- chronological train/validation/test ordering;
- walk-forward evaluation;
- explicit gap/embargo where overlapping forward targets require it;
- final untouched test block.

Initial split convention:

- Train: first 60%
- Validation: next 20%
- Final test: last 20%

This is a pilot convention and may be replaced by calendar-based windows once real-data coverage is known.

## 9. Statistical reporting

For each horizon/model:

- sample count;
- class balance;
- Pearson/Spearman relationship where appropriate;
- coefficient sign and stability;
- accuracy and balanced accuracy for direction classification;
- ROC-AUC where meaningful;
- log loss/Brier score for probabilistic models;
- regression MAE/RMSE for return prediction;
- performance by chronological fold.

We care more about **stability** than one headline metric.

## 10. Economic evaluation

First backtest: conservative market/taker execution.

Assumptions must be explicit:

- decision timestamp;
- execution delay;
- entry price;
- exit price;
- bid/ask crossing;
- fees;
- slippage.

Report:

- gross and net PnL;
- turnover;
- average PnL/trade;
- hit rate;
- Sharpe-like statistic with caveats;
- maximum drawdown;
- transaction-cost sensitivity.

Passive maker simulation is deferred until fill/queue assumptions can be defended.

## 11. Robustness

Required before finalizing conclusions:

- horizon sensitivity;
- transaction-cost sensitivity;
- volatility regimes;
- spread/liquidity regimes;
- feature ablation;
- coefficient/feature-importance stability;
- subperiod performance;
- threshold sensitivity.

## 12. Definition of success

Success is **not** "profitable strategy found."

The pilot succeeds if it produces a reproducible, well-tested answer to:

1. Is the signal statistically present?
2. Does it survive out of sample?
3. Does it survive reasonable costs?
4. When does it fail?
5. Can the mechanism be explained?
