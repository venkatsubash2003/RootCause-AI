
import sys
from pathlib import Path

import numpy as np
import pytest

from sklearn.exceptions import NotFittedError

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from rootcause.detection.multivariate_detector import (
    MultivariateAnomalyDetector
)


@pytest.fixture
def healthy_features():
    rng = np.random.default_rng(42)

    span_count = rng.integers(5, 20, 50)
    median_latency = rng.uniform(0.5, 1.5, 50)

    p95_latency = (
        median_latency
        + rng.uniform(0.5, 2.0, 50)
    )

    error_fraction = np.zeros(50)

    return np.column_stack([
        span_count,
        median_latency,
        p95_latency,
        error_fraction
    ])


def test_training(healthy_features):
    model = MultivariateAnomalyDetector()
    model.fit(healthy_features)

    assert model.is_fitted


def test_prediction_shape(healthy_features):
    model = MultivariateAnomalyDetector()
    model.fit(healthy_features)

    result = model.predict(healthy_features[:5])

    assert len(result["anomaly_scores"]) == 5
    assert len(result["is_anomaly"]) == 5


def test_insufficient_samples(healthy_features):
    model = MultivariateAnomalyDetector()

    with pytest.raises(ValueError):
        model.fit(healthy_features[:5])


def test_incorrect_feature_count():
    model = MultivariateAnomalyDetector()

    with pytest.raises(ValueError):
        model.fit(np.ones((30, 3)))


def test_invalid_error_fraction():
    model = MultivariateAnomalyDetector()

    invalid = np.ones((30, 4))
    invalid[:, -1] = 1.5

    with pytest.raises(ValueError):
        model.fit(invalid)


def test_requires_training(healthy_features):
    model = MultivariateAnomalyDetector()

    with pytest.raises(NotFittedError):
        model.predict(healthy_features[:2])
