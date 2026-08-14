# Dynamic Aggressive Trade-Flow Extension — Protocol v0.8

## Why this is a new experiment

v0.7 established:

1. static L1 imbalance had statistically detectable predictive information;
2. the unchanged baseline model generalized weakly out of sample;
3. the baseline failed conservative economic validation.

v0.8 does **not** tune that baseline after failure.

Instead, it asks a new, pre-specified question:

> Does dynamic aggressive trade flow provide incremental predictive information beyond static L1 book state?

## Important terminology

This project calls the new variable **trade-flow imbalance (TFI)**.

It is **not** full Cont-style order-flow imbalance (OFI), because 1-minute Binance klines do not contain individual limit-order additions and cancellations.

The futures kline archive provides:

- total base volume;
- total quote volume;
- number of trades;
- taker-buy base volume;
- taker-buy quote volume.

From this we derive aggressive buyer/seller flow.

## Data

### Static L1 book state
Existing 1-minute BTCUSDT perpetual book-ticker feature dataset.

### Dynamic aggressive flow
Official Binance USD-M futures **1-minute klines**, monthly files:

May 2023 through March 2024.

Every ZIP is SHA-256 verified against Binance's `.CHECKSUM`.

## Trade-flow definitions

Let:

- `V_t` = total base volume in minute t
- `B_t` = taker-buy base volume

Then seller-initiated base volume is:

`S_t = V_t - B_t`

Signed aggressive flow:

`F_t = B_t - S_t = 2B_t - V_t`

One-minute trade-flow imbalance:

`TFI_1m = F_t / V_t`

Five- and ten-minute TFI are causal volume-weighted rolling ratios:

`TFI_w = sum(F) / sum(V)` over the past w completed minutes.

## Pre-registered incremental features

Only these six are added to the existing baseline:

1. `tfi_1m`
2. `tfi_5m`
3. `tfi_10m`
4. `log_quote_volume`
5. `log_num_trades`
6. `tfi5_x_book_imbalance`

No technical indicators are added.

## Hypotheses

### H8
Positive aggressive trade-flow imbalance predicts positive future mid-price return.

### H9
Five-minute aggregated TFI is more stable than single-minute TFI.

### H10
The augmented model improves out-of-sample ranking relative to the static L1 baseline.

### H11
Agreement between trade flow and book imbalance contains incremental information.

## Model

Both models are fixed logistic regressions with the same scaling and `C=1.0`.

### Baseline
Exactly the v0.5/v0.7 static L1 feature set.

### Augmented
Baseline + the six pre-registered trade-flow features.

No hyperparameter tuning.

## Chronology

- Development/model fit: May 16–Oct 31 2023
- Incremental validation: November 2023
- Incremental confirmation: December 2023
- Extension OOS: Jan–Mar 2024

The extension OOS is not called "untouched" because the same target period was previously inspected for the baseline model. However, the **new flow feature set and v0.8 gate are frozen before any flow-enhanced Jan–Mar result is inspected**.

## Pre-OOS extension gate

Jan–Mar flow-enhanced predictions may be evaluated only if all are true:

1. November ΔAUC >= +0.002
2. December ΔAUC >= +0.001
3. December augmented log loss is no worse than baseline
4. Development standardized coefficient on `tfi_5m` is positive
5. Pre-OOS monthly TFI_5m return beta is positive in >=70% of months

If the gate fails, the extension OOS remains closed.

## Incremental uncertainty

For November, December, and any permitted extension OOS:

- baseline and augmented predictions are compared on identical rows;
- ΔAUC is bootstrapped by **calendar day blocks**;
- the paired design preserves the comparison between models.

## Next stage

If v0.8 passes and the extension OOS is incrementally positive, a later economic study may test whether flow features improve the previously failed v0.7 execution result.

If v0.8 fails, the correct conclusion is:

> Aggressive trade flow did not add sufficiently stable incremental information at this 1-minute horizon under the pre-registered specification.

No post-hoc XGBoost rescue is permitted.
