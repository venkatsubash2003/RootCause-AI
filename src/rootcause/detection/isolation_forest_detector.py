
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError


class IsolationForestDetector:
    def __init__(
        self,
        contamination=0.05,
        n_estimators=200,
        min_samples=20,
        random_state=42
    ):
        self.min_samples = min_samples

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )

        self.is_fitted = False

    def fit(self, durations):
        """
        Train Isolation Forest on healthy latency.
        """

        values = np.asarray(
            durations, dtype=float
        ).reshape(-1)

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Training durations must be finite"
            )

        if np.any(values < 0):
            raise ValueError(
                "Durations cannot be negative"
            )

        if len(values) < self.min_samples:
            raise ValueError(
                "Insufficient healthy training samples"
            )

        # Isolation Forest expects:
        # (number_of_samples, number_of_features)
        features = values.reshape(-1, 1)

        self.model.fit(features)
        self.is_fitted = True

        return self

    def score(self, duration):
        """
        Return a signed anomaly score.

        Positive score: classified as anomalous.
        Negative score: classified as normal.
        """

        if not self.is_fitted:
            raise NotFittedError(
                "Fit Isolation Forest before scoring"
            )

        value = float(duration)

        if not np.isfinite(value) or value < 0:
            raise ValueError("Invalid duration")

        features = np.array([[value]])

        decision = self.model.decision_function(
            features
        )[0]

        return -float(decision)

    def predict(self, duration):
        """
        Predict anomaly status for one observation.
        """

        anomaly_score = self.score(duration)

        return {
            "duration_ms": float(duration),
            "anomaly_score": round(
                anomaly_score, 6
            ),
            "is_anomaly": anomaly_score > 0
        }
