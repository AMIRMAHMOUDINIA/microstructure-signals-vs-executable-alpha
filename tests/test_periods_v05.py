import pandas as pd

from microalpha.validation.periods import Period, select_purged_period


def test_boundary_crossing_label_is_purged():
    ts = pd.to_datetime([
        "2023-10-31T23:48:00Z",
        "2023-10-31T23:50:00Z",
        "2023-10-31T23:59:00Z",
    ])
    df = pd.DataFrame({"timestamp": ts})
    df["target_timestamp_10m"] = df["timestamp"] + pd.Timedelta(minutes=10)

    p = Period("dev", "2023-10-01T00:00:00Z", "2023-11-01T00:00:00Z")
    out = select_purged_period(df, p, 10)

    # 23:50 target is exactly 00:00 => excluded because endpoint must be < end.
    assert out["timestamp"].tolist() == [pd.Timestamp("2023-10-31T23:48:00Z")]
