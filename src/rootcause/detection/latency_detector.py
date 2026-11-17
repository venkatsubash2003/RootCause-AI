
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


    def train_detector(healthy_df):
        split_index = int(len(healthy_df) * 0.7)

        train_df = healthy_df.iloc[:split_index]
        test_df = healthy_df.iloc[split_index:]

        if len(train_df) < 20 or len(test_df) < 5:
            raise ValueError(
                "Need at least 20 training samples "
                "and 5 held-out healthy samples"
            )

        model = LatencyAnomalyDetector(
            threshold=3.5,
            min_samples=20
        )

        model.fit(
            train_df["duration_ms"].to_numpy()
        )

        print("\nHealthy baseline learned")
        print("------------------------")
        print("Training samples:", len(train_df))
        print("Healthy test samples:", len(test_df))
        print("Baseline median:", model.median)
        print("Baseline scale:", model.scale)

        return model, test_df
    
def evaluate(model, df, dataset_name):
    results = df.copy()

    predictions = [
        model.predict(duration)
        for duration in results["duration_ms"]
    ]

    results["anomaly_score"] = [
        p["anomaly_score"] for p in predictions
    ]

    results["is_anomaly"] = [
        p["is_anomaly"] for p in predictions
    ]

    total = len(results)
    anomalies = int(results["is_anomaly"].sum())
    anomaly_rate = anomalies / total if total else 0

    print(f"\nDataset: {dataset_name}")
    print("-" * 35)
    print("Total observations:", total)
    print("Detected anomalies:", anomalies)
    print(f"Anomaly rate: {anomaly_rate:.2%}")
    print(
        "Median latency:",
        round(results["duration_ms"].median(), 2),
        "ms"
    )
    print(
        "p95 latency:",
        round(
            results["duration_ms"].quantile(0.95),
            2
        ),
        "ms"
    )

    return results

