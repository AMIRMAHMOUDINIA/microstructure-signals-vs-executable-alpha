import numpy as np
import pandas as pd

from microalpha.data.normalize import infer_epoch_unit, normalize_bookticker
from microalpha.data.grid import last_quote_grid
from microalpha.data.qa import analyze_bookticker_raw


def sample_raw():
    return pd.DataFrame({
        "update_id": [1, 2, 3, 4],
        "best_bid_price": [99.0, 99.0, 100.0, 100.0],
        "best_bid_qty": [10.0, 12.0, 9.0, 11.0],
        "best_ask_price": [101.0, 101.0, 102.0, 102.0],
        "best_ask_qty": [8.0, 7.0, 6.0, 5.0],
        "transaction_time": [1700000000000, 1700000000000, 1700000001000, 1700000002000],
        "event_time": [1700000000000, 1700000000000, 1700000001000, 1700000002000],
    })


def test_infer_ms():
    s = pd.Series([1700000000000, 1700000001000])
    assert infer_epoch_unit(s) == "ms"


def test_same_timestamp_events_are_preserved():
    df = normalize_bookticker(sample_raw())
    assert len(df) == 4
    assert df["timestamp"].duplicated(keep=False).sum() == 2


def test_grid_explicitly_takes_last_state():
    df = normalize_bookticker(sample_raw())
    grid = last_quote_grid(df, "1s")
    # At first timestamp update_id=2 is the last known state.
    assert np.isclose(grid.iloc[0]["bid_qty"], 12.0)


def test_qa_reports_same_timestamp_without_calling_exact_duplicate():
    _, report = analyze_bookticker_raw(sample_raw())
    assert report.same_timestamp_rows == 2
    assert report.exact_duplicate_rows == 0
    assert report.crossed_rows == 0
