
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


# --------------------------------------------------
# STEP 1: Load and validate telemetry
# --------------------------------------------------

def load_dataset(path, service, operation):
    """
    Load normalized Jaeger telemetry for one
    service-operation pair.
    """

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in {path}: "
            f"{sorted(missing)}"
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


# --------------------------------------------------
# STEP 2: Train the healthy latency baseline
# --------------------------------------------------

def train_detector(healthy_df):
    """
    Train on the earliest 70% of healthy samples.
    Reserve the final 30% for healthy evaluation.
    """

    split_index = int(len(healthy_df) * 0.7)

    train_df = healthy_df.iloc[:split_index]
    test_df = healthy_df.iloc[split_index:]

    if len(train_df) < 20 or len(test_df) < 5:
        raise ValueError(
            "Need at least 20 training samples "
            "and 5 held-out healthy samples. "
            f"Found {len(train_df)} training and "
            f"{len(test_df)} testing observations."
        )

    model = LatencyAnomalyDetector(
        threshold=3.5,
        min_samples=20
    )

    model.fit(
        train_df["duration_ms"].to_numpy()
    )

    print("\nHealthy baseline learned")
    print("-" * 40)
    print("Training samples:", len(train_df))
    print("Healthy test samples:", len(test_df))
    print("Baseline median:", model.median)
    print("Baseline scale:", model.scale)
    print("Anomaly threshold:", model.threshold)

    return model, test_df


# --------------------------------------------------
# STEP 3: Evaluate a dataset
# --------------------------------------------------

def evaluate(model, df, dataset_name):
    """
    Score observations using an already-trained
    latency anomaly detector.
    """

    if df.empty:
        raise ValueError(
            f"{dataset_name} contains no observations"
        )

    results = df.copy()

    predictions = [
        model.predict(duration)
        for duration in results["duration_ms"]
    ]

    results["anomaly_score"] = [
        prediction["anomaly_score"]
        for prediction in predictions
    ]

    results["is_anomaly"] = [
        prediction["is_anomaly"]
        for prediction in predictions
    ]

    total = len(results)
    anomalies = int(results["is_anomaly"].sum())
    anomaly_rate = anomalies / total

    median_latency = results["duration_ms"].median()
    p95_latency = results["duration_ms"].quantile(0.95)
    max_latency = results["duration_ms"].max()

    print(f"\nDataset: {dataset_name}")
    print("-" * 40)

    print("Total observations:", total)
    print("Detected anomalies:", anomalies)
    print(f"Anomaly rate: {anomaly_rate:.2%}")
    print(f"Median latency: {median_latency:.2f} ms")
    print(f"p95 latency: {p95_latency:.2f} ms")
    print(f"Maximum latency: {max_latency:.2f} ms")

    return results


# --------------------------------------------------
# STEP 4: Validate dataset separation
# --------------------------------------------------

def validate_datasets(datasets):
    """
    Ensure all experiment stages use distinct spans
    and occur in the expected chronological order.
    """

    for name, df in datasets:
        if df.empty:
            raise ValueError(
                f"{name} dataset contains no valid observations"
            )

    # Check every pair for duplicate span identities.
    for i in range(len(datasets)):
        name_a, df_a = datasets[i]

        ids_a = set(zip(
            df_a["trace_id"],
            df_a["span_id"]
        ))

        for j in range(i + 1, len(datasets)):
            name_b, df_b = datasets[j]

            ids_b = set(zip(
                df_b["trace_id"],
                df_b["span_id"]
            ))

            shared = ids_a & ids_b

            if shared:
                raise ValueError(
                    f"{name_a} and {name_b} share "
                    f"{len(shared)} span observations."
                )

    # Check the stage order.
    for (name_a, df_a), (name_b, df_b) in zip(
        datasets, datasets[1:]
    ):
        latest_a = df_a["started_at_utc"].max()
        earliest_b = df_b["started_at_utc"].min()

        if latest_a >= earliest_b:
            raise ValueError(
                f"{name_a} and {name_b} overlap in time "
                "or are not chronologically separated."
            )

    print("\nDataset validation passed")
    print("-" * 40)

    for name, df in datasets:
        print(
            f"{name}: {len(df)} observations, "
            f"{df['started_at_utc'].min()} to "
            f"{df['started_at_utc'].max()}"
        )


# --------------------------------------------------
# STEP 5: Run the complete experiment
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "RootCause AI - Latency Anomaly Evaluation"
        )
    )

    parser.add_argument(
        "--healthy",
        required=True,
        help="Path to healthy telemetry CSV"
    )

    parser.add_argument(
        "--fault",
        required=True,
        help="Path to fault-period telemetry CSV"
    )

    parser.add_argument(
        "--recovery",
        required=False,
        help="Optional recovery telemetry CSV"
    )

    parser.add_argument(
        "--service",
        default="ad",
        help="Target microservice"
    )

    parser.add_argument(
        "--operation",
        required=True,
        help="Exact operation name"
    )

    args = parser.parse_args()

    # 1. Load healthy observations.
    healthy = load_dataset(
        args.healthy,
        args.service,
        args.operation
    )

    # 2. Load fault-period observations.
    faulty = load_dataset(
        args.fault,
        args.service,
        args.operation
    )

    # 3. Optionally load recovery observations.
    recovery = None

    if args.recovery:
        recovery = load_dataset(
            args.recovery,
            args.service,
            args.operation
        )

    # 4. Validate all dataset boundaries.
    datasets = [
        ("Healthy", healthy),
        ("Fault", faulty)
    ]

    if recovery is not None:
        datasets.append(
            ("Recovery", recovery)
        )

    validate_datasets(datasets)

    # 5. Train the model ONLY on healthy samples.
    model, healthy_test = train_detector(healthy)

    # 6. Score held-out healthy observations.
    healthy_results = evaluate(
        model,
        healthy_test,
        "Held-out healthy"
    )

    # 7. Score fault-period observations.
    faulty_results = evaluate(
        model,
        faulty,
        "Fault-period observations"
    )

    # 8. Calculate comparison.
    healthy_rate = healthy_results[
        "is_anomaly"
    ].mean()

    fault_rate = faulty_results[
        "is_anomaly"
    ].mean()

    print("\nHealthy vs. Fault Comparison")
    print("=" * 40)

    print(
        f"Healthy anomaly rate: {healthy_rate:.2%}"
    )

    print(
        f"Fault anomaly rate: {fault_rate:.2%}"
    )

    print(
        f"Difference: {fault_rate - healthy_rate:+.2%}"
    )

    # 9. Score recovery with the SAME model.
    if recovery is not None:
        recovery_results = evaluate(
            model,
            recovery,
            "Recovery observations"
        )

        recovery_rate = recovery_results[
            "is_anomaly"
        ].mean()

        print("\nThree-Stage Experiment Summary")
        print("=" * 40)

        print(
            f"Healthy:  {healthy_rate:.2%}"
        )

        print(
            f"Fault:    {fault_rate:.2%}"
        )

        print(
            f"Recovery: {recovery_rate:.2%}"
        )

        print(
            f"Fault vs. Recovery difference: "
            f"{fault_rate - recovery_rate:+.2%}"
        )


if __name__ == "__main__":
    main()
