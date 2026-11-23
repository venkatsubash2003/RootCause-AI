
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError


FEATURE_COLUMNS = [
    "span_count",
    "median_latency_ms",
    "p95_latency_ms",
    "error_fraction",
]


class MultivariateAnomalyDetector:
    def __init__(
        self,
        contamination=0.05,
        n_estimators=200,
        min_samples=20,
        random_state=42
    ):
        self.min_samples = min_samples
        self.feature_names = FEATURE_COLUMNS.copy()

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )

        self.is_fitted = False

    def _validate_features(self, features):
        """
        Validate feature matrix shape and values.
        """

        values = np.asarray(
            features,
            dtype=float
        )

        if values.ndim != 2:
            raise ValueError(
                "Features must be a 2D matrix"
            )

        if values.shape[1] != len(self.feature_names):
            raise ValueError(
                "Incorrect number of features"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Features contain NaN or infinity"
            )

        if np.any(values < 0):
            raise ValueError(
                "Telemetry features cannot be negative"
            )

        if np.any(values[:, -1] > 1):
            raise ValueError(
                "Error fraction must be between 0 and 1"
            )

        return values

    def fit(self, features):
        """
        Train exclusively on healthy telemetry windows.
        """

        values = self._validate_features(features)

        if len(values) < self.min_samples:
            raise ValueError(
                "Insufficient healthy training windows"
            )

        self.model.fit(values)

        self.is_fitted = True

        return self

    def score(self, features):
        """
        Return signed anomaly scores for a feature matrix.

        Positive values indicate observations classified
        as anomalous by the fitted Isolation Forest.
        """

        if not self.is_fitted:
            raise NotFittedError(
                "Fit the model before scoring"
            )

        values = self._validate_features(features)

        return -self.model.decision_function(values)

    def predict(self, features):
        """
        Return anomaly predictions for a feature matrix.
        """

        scores = self.score(features)

        return {
            "anomaly_scores": scores,
            "is_anomaly": scores > 0
        }
