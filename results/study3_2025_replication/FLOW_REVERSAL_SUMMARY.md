# Study III v0.9.1 — 2025 Trade-Flow Reversal Replication

## Data QA

- Observed minutes: **525,600**
- Completeness: **100.000%**

## Aggregate TFI_5m → future 10m close return

| Period | N | Beta (bps/unit) | HAC p | Spearman rho |
|---|---:|---:|---:|---:|
| confirmation | 129,577 | -1.9028 | 0.00003 | -0.03912 |
| replication | 131,030 | -1.2810 | 0.00014 | -0.02909 |

## Incremental control for past 5m return

| Period | TFI beta (bps/unit) | HAC p | Past-return beta |
|---|---:|---:|---:|
| confirmation | -1.1822 | 0.09175 | -0.0202 |
| replication | -0.7815 | 0.22443 | -0.0186 |

## Parameter-free directional ranking

| Period | -TFI5 AUC | -PastRet5 AUC | ΔAUC | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| confirmation | 0.5210 | 0.5176 | 0.0034 | [-0.0007, 0.0076] |
| replication | 0.5151 | 0.5142 | 0.0008 | [-0.0033, 0.0055] |

## Gate

**FAIL**

- PASS — `confirmation_tfi_beta_negative`
- PASS — `replication_tfi_beta_negative`
- PASS — `h1_2025_monthly_negative_fraction_ge_0_67`
- PASS — `confirmation_flow_reversal_auc_ge_0_505`
- PASS — `replication_flow_reversal_auc_ge_0_505`
- PASS — `replication_partial_tfi_beta_negative`
- FAIL — `replication_partial_tfi_hac_p_le_0_10`
- FAIL — `replication_flow_auc_beats_price_reversal_by_0_001`

**Jul–Dec 2025 final OOS was not opened.**
