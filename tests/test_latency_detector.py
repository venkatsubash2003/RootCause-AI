
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

import pytest

from rootcause.detection.latency_detector import (
    LatencyAnomalyDetector
)


@pytest.fixture
def detector():
    healthy = [
        98, 102, 101, 99, 100,
        103, 97, 101, 102, 100,
        99, 98, 104, 100, 101,
        103, 99, 102, 100, 98,
    ]

    return LatencyAnomalyDetector().fit(healthy)


def test_normal_latency(detector):
    result = detector.predict(103)

    assert result["is_anomaly"] is False


def test_abnormally_high_latency(detector):
    result = detector.predict(450)

    assert result["is_anomaly"] is True
    assert result["anomaly_score"] > 3.5


def test_requires_baseline():
    model = LatencyAnomalyDetector()

    with pytest.raises(RuntimeError):
        model.predict(120)


def test_insufficient_baseline():
    model = LatencyAnomalyDetector()

    with pytest.raises(ValueError):
        model.fit([100, 105, 110])
