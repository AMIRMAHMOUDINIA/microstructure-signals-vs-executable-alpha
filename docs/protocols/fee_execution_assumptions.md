# Fee and Execution Assumptions — v0.7

## Why the code does not hardcode a single "current Binance fee"

Futures fees can vary with VIP status, BNB discounts, programs, product, and promotions. The economically honest approach for this research project is therefore to show a **cost sensitivity curve**, not to build the conclusion around one favorable tier.

The v0.7 additional round-trip fee+slippage grid is:

`0 / 1 / 2 / 4 / 6 / 8 / 10 / 12 bps`

This is charged **after** the bid/ask spread already paid by the conservative BBO execution model.

The upper part of the grid is deliberately severe enough to cover retail-taker-like conditions plus modest slippage. The final GitHub report should always tell readers to check the exchange's live fee page before interpreting one specific scenario as operationally current.

## Spread

Spread is not approximated as an arbitrary fixed cost.

It is reconstructed minute by minute from the dataset's:

- mid price;
- `bt_spread_bps_close`.

Every long enters at ask and exits at bid. Every short enters at bid and exits at ask.

## Execution delay

Signal is computed from minute `t`, but entry uses the BBO at minute `t+1`.

This avoids the unrealistic claim that we can observe the completed minute-t feature vector and simultaneously fill against the exact same closing quote.

## Funding

Not modeled in v0.7.

Positions are short-lived and only some would cross a funding timestamp, but ignoring funding is still a limitation. If the signal survives ordinary spread/fee/slippage costs, funding can be added as a secondary refinement.
