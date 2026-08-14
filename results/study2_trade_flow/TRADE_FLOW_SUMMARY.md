# Dynamic Trade-Flow Extension — v0.8

## Data QA

- Flow rows: **461,531**
- Flow completeness on book grid: **100.000%**
- Median same-minute kline-close vs reconstructed-mid difference: **0.016 bps**

## Pre-OOS incremental comparison

| Period | Baseline AUC | Augmented AUC | ΔAUC | ΔLogLoss | AUC bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| november | 0.5109 | 0.5151 | 0.0042 | 0.000052 | [-0.0035, 0.0108] |
| december | 0.5160 | 0.5220 | 0.0060 | -0.000318 | [0.0005, 0.0116] |

## Primary TFI_5m monthly stability

| Month | Beta (bps/unit) | HAC p |
|---|---:|---:|
| 2023-05 | -0.7441 | 0.32950 |
| 2023-06 | -0.9861 | 0.15040 |
| 2023-07 | -0.9937 | 0.00785 |
| 2023-08 | -0.4137 | 0.18325 |
| 2023-09 | -0.6684 | 0.06933 |
| 2023-10 | -0.5937 | 0.32181 |
| 2023-11 | -1.0249 | 0.08470 |
| 2023-12 | -0.1109 | 0.87700 |

## Largest augmented standardized coefficients

| Feature | Coefficient |
|---|---:|
| `bt_imbalance_close` | 0.04924 |
| `tfi_10m` | -0.03995 |
| `bt_imbalance_twap` | 0.03867 |
| `log_quote_volume` | -0.03789 |
| `log_num_trades` | 0.03487 |
| `tfi_1m` | -0.03135 |
| `tfi_5m` | -0.03005 |
| `imbalance_close_minus_twap` | 0.02973 |
| `log_depth_close` | -0.02267 |
| `tfi5_x_book_imbalance` | 0.00499 |

## Pre-specified extension gate

**FAIL**

- PASS — `november_delta_auc_ge_0_002`
- PASS — `december_delta_auc_ge_0_001`
- PASS — `december_log_loss_not_worse`
- FAIL — `development_tfi5_coefficient_positive`
- FAIL — `pre_oos_monthly_tfi5_positive_fraction_ge_0_70`

**Jan–Mar extension OOS was not opened because the pre-OOS gate failed.**
