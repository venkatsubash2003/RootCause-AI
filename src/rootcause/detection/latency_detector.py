
import numpy as np


class LatencyAnomalyDetector:
    def __init__(self, threshold=3.5, min_samples=20):
        self.threshold = threshold
        self.min_samples = min_samples
        self.median = None
        self.scale = None

    def fit(self, durations):
        """
        Learn latency statistics from a healthy
        reference period.
        """
        values = np.asarray(durations, dtype=float)

        values = values[
            np.isfinite(values) & (values >= 0)
        ]

        if len(values) < self.min_samples:
            raise ValueError(
                "Insufficient healthy baseline samples"
            )

        self.median = float(np.median(values))

        mad = np.median(
            np.abs(values - self.median)
        )

        # Robust scale estimate.
        # The floor avoids division by zero.
        # It must be tuned for each metric.
        self.scale = max(
            float(1.4826 * mad),
            1.0
        )

        return self

    def score(self, duration):
        """
        Calculate the latency anomaly score.
        """
        if self.median is None:
            raise RuntimeError("Fit the detector first")

        if not np.isfinite(duration) or duration < 0:
            raise ValueError("Invalid duration")

        return (
            float(duration) - self.median
        ) / self.scale

    def predict(self, duration):
        """
        Detect unusually high latency.
        """
        anomaly_score = self.score(duration)

        return {
            "duration_ms": float(duration),
            "anomaly_score": round(
                anomaly_score, 3
            ),
            "is_anomaly": (
                anomaly_score > self.threshold
            )
        }
