# CV Bullets

## Recommended 2-bullet version

- Built a **Python market-microstructure research pipeline** for BTCUSDT perpetual futures, engineering L1 book-imbalance and aggressive trade-flow signals with exact-horizon labels, chronological validation, HAC/FDR inference, block bootstrap and automated tests; a frozen logistic baseline retained weak untouched-test ranking power (**AUC 0.510**).
- Designed a **cost-aware execution and hypothesis-gating framework** that separated statistical predictability from executable alpha: the static-book signal failed independent economic confirmation, while an unexpected aggressive-flow reversal effect replicated on fresh 2025 data but was not robustly incremental to simple price mean reversion.

## More technical 3-bullet version

- Researched short-horizon BTCUSDT microstructure using **460k+ one-minute L1 observations**, finding a stable positive 10-minute book-imbalance coefficient across all pre-test months and validating with HAC/Newey-West, Benjamini-Hochberg FDR, non-overlapping samples and block bootstrap.
- Implemented leakage-aware logistic baselines and a **locked train/validation/test protocol**; directional ranking generalized to Jan-Mar 2024 (**AUC 0.5103**) but a conservative BBO-crossing backtest failed an independent December economic gate, preventing post-hoc PnL optimization.
- Extended the study to Binance taker-flow imbalance; rejected the pre-registered continuation hypothesis and independently replicated a **contrarian TFI-return relationship** in 2025 (AUC **0.5210 / 0.5151** in confirmation/replication), while preserving the null result that incremental value versus simple price reversal was not robust.
