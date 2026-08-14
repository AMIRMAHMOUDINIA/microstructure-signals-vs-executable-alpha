# v0.9 Data-Source Postmortem

v0.9 attempted to construct a fresh post-March-2024 L1 book-state sample from
Binance USD-M `bookTicker` monthly archives.

The returned v0.9 result bundle showed:

- only 407 book minutes;
- start: 2024-04-01 00:00 UTC;
- end: 2024-04-01 06:46 UTC;
- no confirmation/replication result files.

Therefore v0.9 never reached its statistical gate and **must not be interpreted
as evidence for or against the state-flow hypothesis**.

The branch is abandoned rather than repaired around incomplete bookTicker data.

Study III v0.9.1 instead performs a genuinely fresh 2025 replication of the
trade-flow-reversal discovery using the consistently available official Binance
USD-M 1-minute kline archive.
