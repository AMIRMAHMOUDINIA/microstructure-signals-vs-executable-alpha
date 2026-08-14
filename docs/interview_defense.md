# Interview Defense Notes

## 1. Why is AUC 0.510 meaningful at all?

Because the sample is large and the research question is short-horizon ranking, even a small AUC can represent real information. But I do not equate 0.510 with profitable alpha. The project explicitly tests that second claim separately, and the economic gate fails.

## 2. Why did you not optimize until the backtest became profitable?

That would convert a research result into a data-mined portfolio demo. Thresholds, chronology and cost grids were frozen before the relevant evaluation periods. When the economic gate failed, I preserved the failure.

## 3. What is the difference between book imbalance, TFI and OFI?

- **Book imbalance** is a state variable: relative displayed quantity at the best bid versus ask.
- **Trade-flow imbalance (TFI)** is derived from aggressive buyer- versus seller-initiated traded volume.
- **Order-flow imbalance (OFI)** is broader and includes changes caused by limit-order additions, cancellations and market orders. I deliberately did not call the kline-derived signal OFI.

## 4. Why HAC/Newey-West?

Forward returns overlap. A 10-minute target sampled every minute induces serial dependence in regression residuals. Naive iid standard errors can therefore create false confidence. HAC covariance is one of several checks I used alongside non-overlapping samples and block bootstrap.

## 5. Why reconstruct future mid-price instead of using microprice?

Microprice embeds contemporaneous queue imbalance. Predicting future microprice with current imbalance risks making the response mechanically related to the predictor. Mid-price is a cleaner primary response.

## 6. Why did you create a complete minute grid?

A row-based `shift(-10)` is not necessarily a 10-minute target when rows are missing. Reindexing to the full clock grid lets missing exact endpoints remain missing instead of silently skipping forward.

## 7. Why was the final economic OOS not opened?

The December economic confirmation gate failed for every pre-specified cost scenario. Looking at later PnL after that failure would only invite test-driven threshold selection.

## 8. What did the economic study teach you?

The strongest conclusion was that statistical information was too small or unstable to monetize under the frozen taker design. At the high-confidence December threshold, gross break-even extra round-trip cost was only about 0.91 bps.

## 9. What surprised you most?

The trade-flow sign. I expected aggressive buying to predict continuation, but TFI coefficients were negative across every pre-OOS 2023 month. That led to a new hypothesis rather than rewriting the old one.

## 10. Did that reversal replicate?

Yes. On fresh 2025 data, TFI_5m beta was -1.90 bps/unit in Jan-Mar and -1.28 bps/unit in Apr-Jun, with all six monthly betas negative.

## 11. Then why did Study III still fail its gate?

Because the stronger incremental claim did not replicate. Once I controlled for previous 5-minute price return, the Apr-Jun TFI coefficient had HAC p=0.224, and flow-reversal AUC beat simple price reversal by only +0.0008, below the pre-registered +0.001 criterion.

## 12. Why not use XGBoost?

A nonlinear model should answer a research question, not rescue a failed gate. The linear relationships and execution economics were the central questions. After those results, adding complexity without new data would increase overfitting risk and weaken interpretability.

## 13. What would you do next in a real prop environment?

I would move to event-level data with queue-aware L2/L3 dynamics, estimate true OFI including additions/cancellations, separate passive and aggressive execution, and evaluate whether state-flow disagreement predicts temporary impact versus information-driven flow. I would pre-register the feature family and use a new out-of-time period.

## 14. How would you improve execution realism?

Model exchange fees by tier, funding when positions cross funding timestamps, latency distributions, market impact/capacity, queue position for maker strategies, outage handling and position sizing. The current economic layer is intentionally a conservative small-notional taker model.

## 15. What is the most important lesson?

**Predictability, calibration and monetizability are different claims.** The project is designed to show the evidence required for each one rather than collapse them into a single backtest statistic.
