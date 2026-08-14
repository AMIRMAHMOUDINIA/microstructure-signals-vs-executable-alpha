# Reproducibility Guide

The final release is script-first and does not commit raw market data.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

## Study I - static L1 book state

The original workflow uses the public one-minute BTCUSDT book-ticker feature Parquet documented in the project, verifies data integrity, reconstructs mid-price, creates exact clock targets, then runs:

```bash
python scripts/full_descriptive_analysis_v05.py <feature_dataset.parquet>
python scripts/chronological_logistic_baseline_v05.py <feature_dataset.parquet> --horizon 10
```

The test set is protected by the validation-gate mechanism in the original scripts.

## Study I-B - economic validation

```bash
python scripts/economic_backtest_v07.py <feature_dataset.parquet> \
  --prior-gate prior_results/validation_gate.json \
  --output-dir results/economic_v07
```

Thresholds are selected in November and confirmed in December before any economic OOS is permitted.

## Study II - aggressive trade flow

Download official Binance USD-M 1-minute klines:

```bash
python scripts/download_binance_klines_v08.py \
  --start 2023-05 --end 2024-03 \
  --output data/raw/binance_klines
```

Then run the pre-registered extension:

```bash
PYTHONPATH="scripts:$PYTHONPATH" python scripts/trade_flow_extension_v08.py \
  <feature_dataset.parquet> \
  --kline-dir data/raw/binance_klines
```

## Study III - fresh 2025 reversal replication

```bash
python scripts/download_binance_klines_v08.py \
  --start 2025-01 --end 2025-12 \
  --output data/raw/binance_klines_2025

PYTHONPATH="scripts:$PYTHONPATH" python scripts/flow_reversal_replication_v091.py \
  --kline-dir data/raw/binance_klines_2025
```

The Jul-Dec 2025 final OOS remains gated if the Jan-Jun replication gate fails.

## Integrity

The Binance downloader verifies the `.CHECKSUM` SHA-256 companion files. Results in `results/` are frozen outputs from the executed research sequence.
