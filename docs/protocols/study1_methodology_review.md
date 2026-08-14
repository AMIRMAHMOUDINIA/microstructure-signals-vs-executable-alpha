# Methodology Review — v0.5

This version corrects four issues that could otherwise create a misleading quant-research result.

## Correction 1 — exact clock horizons, not row shifts

The source dataset reports 99.726% completeness on a 1-minute grid. A naïve:

```python
price.shift(-10)
```

on the **observed-row table** does not always mean 10 minutes; around a missing minute it can become 11+ clock minutes.

v0.5 first reindexes to a complete UTC 1-minute grid. Missing minutes remain missing. The target at `t+h` is valid only when both `t` and the exact `t+h` row were actually observed.

## Correction 2 — future mid-price, not future microprice

The earlier pilot used future microprice as the response. But microprice is itself:

\[
\mu_t = \frac{a_t q^b_t + b_t q^a_t}{q^b_t + q^a_t}
\]

and therefore embeds queue imbalance. That makes it a poor primary response for testing whether imbalance forecasts the underlying quoted price.

The dataset defines:

\[
premium_t = \frac{\mu_t - m_t}{m_t}
\]

so v0.5 reconstructs:

\[
m_t = \frac{\mu_t}{1 + premium_t}
\]

and predicts **future mid-price log return**.

## Correction 3 — microprice premium is not an independent primitive feature

At level 1:

\[
premium_t =
\frac{spread_{bps,t} \times imbalance_t}{20{,}000}
\]

up to source rounding.

Therefore `microprice_premium_close` is essentially a deterministic interaction between close spread and close imbalance. It remains in descriptive analysis as a composite pressure proxy, but it is excluded from the primary logistic feature matrix to avoid pretending it provides independent information.

The QA report explicitly measures this identity.

## Correction 4 — purge period boundaries

A 10-minute development label constructed at 23:55 immediately before a 00:00 validation boundary can consume validation-period prices.

v0.5 requires each label endpoint to remain strictly inside its own period. This purges boundary-crossing labels.

## Additional controls

- Benjamini–Hochberg FDR across feature/horizon HAC tests.
- Non-overlapping inference across **all possible clock offsets**, not one arbitrary offset.
- Fast moving-block-bootstrap confidence intervals.
- No missing-feature imputation in the primary baseline.
- Test set remains locked until a pre-specified validation gate passes.
- Standardized coefficients are exported for interpretation.
- The model predicts direction of exact-horizon **mid-price** returns.

## Interpretation rule

Statistical significance alone is insufficient.

A result can advance only if it shows:

1. meaningful effect size;
2. HAC/FDR support;
3. non-overlap sign stability;
4. monthly directional stability;
5. validation performance above the fixed gate.

Only after that do we open the test set. Only after the test set do we begin execution-cost modelling.

This is deliberately stricter than a typical portfolio project.
