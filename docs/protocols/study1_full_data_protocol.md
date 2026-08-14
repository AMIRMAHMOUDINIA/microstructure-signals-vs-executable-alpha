# Full-Data Protocol — v0.5

## Data

Dataset: `Mindbyte-89/btcusdt_perp_bookticker_features_1m_05_2023_to_03_2024`

Public card reports:

- 460,265 rows
- 1-minute cadence
- 2023-05-16 11:49 UTC → 2024-03-31 23:59 UTC
- 99.726% completeness

Expected Parquet SHA-256:

`274eb8e87c7d7185a0162271144b30a0e387ae496fe657c6af83833448f08624`

## Primary response

Exact-clock future **mid-price log return**, not future microprice return.

Mid is reconstructed from the dataset's own definitions:

`mid = microprice / (1 + microprice_premium)`.

## Primary descriptive horizons

- 1 minute
- 5 minutes
- 10 minutes
- 30 minutes

## Primary descriptive features

- close imbalance
- TWAP imbalance
- microprice premium (composite proxy; not independent)

## Inference

For each feature/horizon:

- Pearson and Spearman effect;
- OLS effect size in bps per unit feature;
- HAC/Newey–West covariance;
- Benjamini–Hochberg FDR across the primary family;
- non-overlapping estimates over every possible clock offset;
- moving-block bootstrap;
- monthly coefficient stability.

## Primary model features

- close imbalance
- TWAP imbalance
- close spread
- TWAP spread
- log quote-update rate
- log close depth
- close-minus-TWAP imbalance

Microprice premium is excluded from the primary model because it is essentially the close spread × close imbalance interaction.

## Locked periods

- Development: 2023-05-16 11:49 → 2023-10-31 23:59
- Validation: 2023-11-01 → 2023-12-31 23:59
- Test: 2024-01-01 → 2024-03-31 23:59

Labels crossing a period boundary are purged.

## Validation gate before test unlock

For the pre-specified 10-minute primary path:

- development close-imbalance HAC FDR q <= 0.05;
- non-overlap sign agreement >= 75%;
- monthly close-imbalance beta positive in >= 70% of development+validation months;
- validation AUC >= 0.505;
- validation MCC > 0.

This is an internal escalation rule, not a universal scientific law.

If the gate fails, we stop complexity escalation and write the fragile/null result.

If the gate passes, the untouched test may be evaluated once.

## Only after the test

- transaction-cost-aware decision threshold;
- conservative taker backtest;
- regime robustness;
- gradient boosting only if linear evidence justifies it.
