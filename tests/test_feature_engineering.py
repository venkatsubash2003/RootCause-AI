
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from rootcause.detection.feature_engineering import (
    build_features
)


def test_feature_engineering():
    records = pd.DataFrame({
        "started_at_utc": pd.to_datetime(
            [
                "2026-11-18T10:00:01Z",
                "2026-11-18T10:00:05Z",
                "2026-11-18T10:00:11Z",
                "2026-11-18T10:00:35Z"
            ],
            utc=True
        ),
        "duration_ms": [
            1.0, 2.0, 9.0, 3.0
        ],
        "is_error": [
            0, 0, 1, 0
        ]
    })

    features = build_features(
        records,
        window_size="30s"
    )

    # Two nonempty windows
    assert len(features) == 2

    first = features.iloc[0]
    second = features.iloc[1]

    assert first["span_count"] == 3
    assert first["median_latency_ms"] == 2.0
    assert first["error_count"] == 1
    assert first["error_fraction"] == 1 / 3

    assert second["span_count"] == 1
    assert second["median_latency_ms"] == 3.0
    assert second["error_fraction"] == 0.0
