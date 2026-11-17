
import argparse

import pandas as pd

from rootcause.detection.latency_detector import (
    LatencyAnomalyDetector
)


REQUIRED_COLUMNS = {
    "service",
    "operation",
    "duration_ms",
    "started_at_utc",
    "trace_id",
    "span_id",
}


def load_dataset(path, service, operation):
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[
        (df["service"] == service)
        & (df["operation"] == operation)
    ].copy()

    df["duration_ms"] = pd.to_numeric(
        df["duration_ms"],
        errors="coerce"
    )

    df["started_at_utc"] = pd.to_datetime(
        df["started_at_utc"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "duration_ms",
            "started_at_utc",
            "trace_id",
            "span_id"
        ]
    )

    df = df[
        (df["duration_ms"] >= 0)
        & df["trace_id"].astype(str).str.len().gt(0)
        & df["span_id"].astype(str).str.len().gt(0)
    ]

    df = df.drop_duplicates(
        subset=["trace_id", "span_id"]
    )

    return df.sort_values(
        "started_at_utc"
    ).reset_index(drop=True)

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




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--healthy", required=True)
    parser.add_argument("--fault", required=True)
    parser.add_argument("--service", default="ad")
    parser.add_argument("--operation", required=True)

    args = parser.parse_args()

    healthy = load_dataset(
        args.healthy,
        args.service,
        args.operation
    )

    faulty = load_dataset(
        args.fault,
        args.service,
        args.operation
    )

    if healthy.empty or faulty.empty:
        raise ValueError(
            "One or both datasets contain no valid "
            "samples for the selected operation"
        )

    # Prevent reusing identical observations.
    healthy_ids = set(
        zip(healthy["trace_id"], healthy["span_id"])
    )
    faulty_ids = set(
        zip(faulty["trace_id"], faulty["span_id"])
    )

    if healthy_ids & faulty_ids:
        raise ValueError(
            "Healthy and fault datasets share spans. "
            "Collect or filter disjoint windows."
        )

    # Require the healthy window to precede
    # the fault evaluation window.
    if (
        healthy["started_at_utc"].max()
        >= faulty["started_at_utc"].min()
    ):
        raise ValueError(
            "The datasets overlap in time or are "
            "not chronologically separated."
        )

    model, healthy_test = train_detector(healthy)

    healthy_results = evaluate(
        model,
        healthy_test,
        "Held-out healthy"
    )

    faulty_results = evaluate(
        model,
        faulty,
        "Fault-period observations"
    )

    print("\nComparison")
    print("=" * 40)

    normal_rate = healthy_results["is_anomaly"].mean()
    fault_rate = faulty_results["is_anomaly"].mean()

    print(f"Healthy anomaly rate: {normal_rate:.2%}")
    print(f"Fault anomaly rate: {fault_rate:.2%}")
    print(f"Difference: {fault_rate - normal_rate:+.2%}")


if __name__ == "__main__":
    main()
