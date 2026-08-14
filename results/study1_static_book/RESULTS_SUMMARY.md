# Full-Data Research Result Summary

## Dataset integrity

- Observed rows: **460,265**
- Grid completeness: **99.726%**
- Missing grid rows: **1266**
- Premium identity correlation: **1.000000**

## Primary descriptive path — 10-minute close imbalance

| Sample | N | Beta (bps/unit) | HAC p | BH q | Non-overlap positive fraction |
|---|---:|---:|---:|---:|---:|
| Development | 241,489 | 0.5255 | 0.00000 | 0.00000 | 1.000 |
| Validation | 87,810 | 0.6647 | 0.00000 | 0.00000 | 1.000 |

- Positive monthly beta fraction across pre-test months: **1.000**

## Logistic baseline

| Sample | N | AUC | Balanced accuracy | MCC | Log loss | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Development | 241,489 | 0.5232 | 0.5174 | 0.0348 | 0.6923 | 0.2496 |
| Validation | 87,810 | 0.5133 | 0.5095 | 0.0196 | 0.6930 | 0.2499 |

## Standardized logistic coefficients

| Feature | Coefficient |
|---|---:|
| `bt_imbalance_close` | 0.04120 |
| `imbalance_close_minus_twap` | 0.03539 |
| `log_depth_close` | -0.02475 |
| `bt_imbalance_twap` | 0.01439 |
| `bt_spread_bps_close` | -0.00248 |
| `bt_spread_bps_twap` | 0.00166 |
| `log_update_rate` | -0.00163 |

## Pre-specified validation gate

**Gate result: PASS**

- PASS — `development_hac_fdr_q_le_005`
- PASS — `nonoverlap_sign_agreement_ge_075`
- PASS — `monthly_positive_fraction_ge_070`
- PASS — `validation_auc_ge_0505`
- PASS — `validation_mcc_positive`

The pre-specified gate passed. The untouched January–March 2024 test may now be evaluated **once**.

## Untouched test (opened after gate)

| N | AUC | Balanced accuracy | MCC | Log loss | Brier |
|---:|---:|---:|---:|---:|---:|
| 130,851 | 0.5103 | 0.5065 | 0.0141 | 0.6932 | 0.2500 |
