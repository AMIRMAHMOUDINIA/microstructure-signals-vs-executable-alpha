# Economic Validation Protocol — v0.7

## Status entering v0.7

The statistical gate passed before the model test was opened.

Previously observed model performance:

- Development AUC: 0.5232
- Validation AUC: 0.5133
- Jan–Mar 2024 model OOS AUC: 0.5103

v0.7 does **not** change the model, horizon, or feature set.

## Core question

> Does the weak but reproducible directional signal survive conservative execution and additional transaction costs?

## Chronology

### Model training
May 16–October 31, 2023.

### Economic threshold selection
**November 2023 only.**

### Economic confirmation
**December 2023 only.**

### Economic OOS
**January–March 2024.**

The economic OOS period is not used to select a threshold, cost assumption, model, or horizon.

## Signal threshold family

A fixed family of target coverages is declared before economic results:

- 100%
- 75%
- 50%
- 25%
- 10%
- 5%

For each coverage, November determines a numeric cutoff in `abs(p - 0.5)`.

For each additional-cost scenario, the November cutoff with the highest net total bps is selected subject to at least 100 trades.

No threshold is chosen from December or test PnL.

## Execution

At decision minute `t`:

1. The minute-t feature vector is complete.
2. Direction is taken from the unchanged logistic probability.
3. Entry occurs at the **next minute's close BBO (t+1)**.
4. Exit occurs at the **original 10-minute forecast endpoint (t+10)**.
5. Long: buy ask at entry, sell bid at exit.
6. Short: sell bid at entry, buy ask at exit.
7. Positions do not overlap.

This deliberately imposes one minute of execution delay. The realized primary holding interval is therefore 9 minutes while the signal remains the locked t→t+10 forecast.

## Spread

The dataset contains minute-close spread in basis points and a reconstructable mid.

We reconstruct:

- `bid = mid - spread/2`
- `ask = mid + spread/2`

The backtest crosses the spread on both entry and exit.

## Additional round-trip cost sensitivity

After spread, a pre-specified fee+slippage sensitivity grid is subtracted:

`0, 1, 2, 4, 6, 8, 10, 12 bps round trip`

This is intentionally a sensitivity analysis rather than a hardcoded claim about one live Binance tier.

## December economic gate

For each cost scenario, its November-selected threshold advances to economic OOS only if December has:

- at least 100 trades;
- positive mean net bps/trade;
- positive total net bps;
- positive net PnL in at least half of calendar weeks.

If a cost scenario fails, its Jan–Mar economic PnL is not evaluated.

## Reported outputs

- number of trades;
- long fraction;
- gross mean/total bps after spread;
- net mean/median/std bps;
- net total bps;
- hit rate;
- fixed-notional maximum drawdown;
- break-even additional round-trip cost;
- HAC t/p for mean trade PnL;
- monthly OOS stability.

## What this backtest does NOT claim

It is not a production simulator.

It does not model:

- queue priority;
- passive fills;
- funding payments;
- market impact/capacity;
- exchange outages;
- sub-minute latency;
- liquidation/leverage;
- variable fee tiers.

Those become explicit limitations in the final research report.
