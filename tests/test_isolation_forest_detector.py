
import sys
from pathlib import Path

import numpy as np
import pytest

from sklearn.exceptions import NotFittedError

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from rootcause.detection.isolation_forest_detector import (
    IsolationForestDetector
)


@pytest.fixture
def detector():
    # Simulated healthy latencies.
    healthy = np.linspace(0.5, 1.5, 50)

    return IsolationForestDetector().fit(healthy)


def test_training(detector):
    assert detector.is_fitted is True


def test_prediction_structure(detector):
    result = detector.predict(1.0)

    assert "duration_ms" in result
    assert "anomaly_score" in result
    assert "is_anomaly" in result

    assert isinstance(result["is_anomaly"], bool)


def test_extreme_outlier(detector):
    normal_score = detector.score(1.0)
    extreme_score = detector.score(100.0)

    assert extreme_score > normal_score


def test_requires_training():
    model = IsolationForestDetector()

    with pytest.raises(NotFittedError):
        model.predict(1.0)


def test_insufficient_samples():
    model = IsolationForestDetector()

    with pytest.raises(ValueError):
        model.fit([0.5, 1.0, 1.5])


def test_invalid_latency(detector):
    with pytest.raises(ValueError):
        detector.predict(-10)


def test_reproducibility():
    healthy = np.linspace(0.5, 1.5, 50)

    model_a = IsolationForestDetector().fit(healthy)
    model_b = IsolationForestDetector().fit(healthy)

    assert model_a.score(2.0) == pytest.approx(
        model_b.score(2.0)
    )
