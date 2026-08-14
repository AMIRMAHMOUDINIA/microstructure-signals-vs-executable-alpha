# Untouched Test-Set Policy

The locked test period is:

**2024-01-01 00:00 UTC through 2024-03-31 23:59 UTC**

The primary baseline script does **not** evaluate it by default.

The test may be opened only after `evaluate_validation_gate.py` writes a JSON result with:

```json
{"gate_pass": true}
```

After the test is opened:

- no feature may be added because it improves test performance;
- no horizon may be changed because it improves test performance;
- no threshold may be tuned on test;
- no model family may be selected on test.

Any later change must be evaluated on a genuinely new out-of-time dataset and labelled accordingly.

This policy exists to make the final GitHub project interview-defensible.
