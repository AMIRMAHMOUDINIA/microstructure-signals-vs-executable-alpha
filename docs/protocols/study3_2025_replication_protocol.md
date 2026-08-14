# Study III v0.9.1 — 2025 Trade-Flow Reversal Replication

## Why v0.9 was abandoned

The attempted post-March-2024 Binance `bookTicker` source was incomplete.
The v0.9 result bundle contained only 407 minutes (2024-04-01 00:00–06:46 UTC)
and never reached its confirmation/replication stage.

That branch produces **no scientific conclusion**.

## New Study III question

Study II unexpectedly found:

> positive aggressive trade-flow imbalance was followed by lower, not higher,
> 10-minute future prices.

v0.9.1 asks whether that reversal pattern replicates on a completely newer
calendar-year sample.

## Fresh data

Official Binance USD-M BTCUSDT 1-minute klines:

**2025-01 through 2025-12**

Each monthly ZIP is checksum-verified.

No 2025 outcome has been inspected before this protocol is frozen.

## Target

Future 10-minute **close-to-close log return**.

This is a new replication target and is not labelled as identical to the
earlier mid-price target.

Missing exact future minutes yield missing targets; no forward filling.

## Primary signal

`flow_reversal_score = -TFI_5m`

where:

`TFI_5m = sum(2*taker_buy_base - total_volume) / sum(total_volume)`

over the current and previous four completed 1-minute bars.

## Control signal

`price_reversal_score = -past_5m_close_return`

This tests whether TFI contains information beyond ordinary short-horizon
price mean reversion.

## Chronology

- Jan–Mar 2025: confirmation
- Apr–Jun 2025: replication
- Jul–Dec 2025: final OOS, gated

## Gate

All conditions must pass before Jul–Dec is opened:

1. Jan–Mar aggregate TFI_5m beta < 0
2. Apr–Jun aggregate TFI_5m beta < 0
3. At least 2/3 of Jan–Jun monthly TFI betas are negative
4. Jan–Mar `-TFI_5m` AUC >= 0.505
5. Apr–Jun `-TFI_5m` AUC >= 0.505
6. Apr–Jun TFI coefficient remains negative after controlling for past 5m return
7. That partial TFI HAC p-value <= 0.10
8. Apr–Jun flow-reversal AUC exceeds simple price-reversal AUC by >= 0.001

If the gate fails, Jul–Dec remains closed.

## No economic claim

This is a **statistical replication study only**.

No PnL, fees, spread, or execution claims are made from kline closes.

Economic validation would require a later, separately frozen execution study.
