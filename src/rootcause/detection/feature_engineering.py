
import pandas as pd
from pathlib import Path


DATA_DIR = Path("data/processed")

SERVICE = "ad"
OPERATION = "oteldemo.AdService/GetAds"

WINDOW_SIZE = "30s"


def load_spans(file_path):
    """
    Load and validate Jaeger span records.
    """

    df = pd.read_csv(file_path)

    required = {
        "service",
        "operation",
        "started_at_utc",
        "duration_ms",
        "status",
        "trace_id",
        "span_id",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing telemetry columns: {missing}"
        )

    df = df[
        (df["service"] == SERVICE)
        & (df["operation"] == OPERATION)
    ].copy()

    df["started_at_utc"] = pd.to_datetime(
        df["started_at_utc"],
        utc=True,
        errors="coerce"
    )

    df["duration_ms"] = pd.to_numeric(
        df["duration_ms"],
        errors="coerce"
    )

    df["status"] = pd.to_numeric(
        df["status"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "started_at_utc",
            "duration_ms",
            "status",
            "trace_id",
            "span_id",
        ]
    )

    df = df[
        (df["duration_ms"] >= 0)
        & df["status"].isin([0, 1, 2])
    ]

    df = df.drop_duplicates(
        subset=["trace_id", "span_id"]
    )

    # OpenTelemetry status code 2 means ERROR.
    # Code 0 means UNSET, not confirmed success.
    df["is_error"] = (
        df["status"] == 2
    ).astype(int)

    return df.sort_values("started_at_utc")


def build_features(df, window_size=WINDOW_SIZE):
    """
    Aggregate spans into time-window features.
    """

    columns = [
        "window_start",
        "span_count",
        "median_latency_ms",
        "p95_latency_ms",
        "mean_latency_ms",
        "error_count",
        "error_fraction",
    ]

    if df.empty:
        return pd.DataFrame(columns=columns)

    grouped = df.groupby(
        pd.Grouper(
            key="started_at_utc",
            freq=window_size
        )
    )

    features = grouped.agg(
        span_count=("duration_ms", "size"),
        median_latency_ms=(
            "duration_ms", "median"
        ),
        p95_latency_ms=(
            "duration_ms",
            lambda values: values.quantile(0.95)
        ),
        mean_latency_ms=(
            "duration_ms", "mean"
        ),
        error_count=("is_error", "sum")
    ).reset_index()

    features = features.rename(
        columns={"started_at_utc": "window_start"}
    )

    # Do not treat empty intervals as observations.
    features = features[
        features["span_count"] > 0
    ].copy()

    features["error_fraction"] = (
        features["error_count"]
        / features["span_count"]
    )

    return features[columns]


def main():
    stages = ["healthy", "faulty", "recovery"]

    datasets = []

    for stage in stages:
        path = DATA_DIR / f"ad_{stage}.csv"

        print(f"\nProcessing {stage} telemetry")

        spans = load_spans(path)

        features = build_features(spans)

        features["stage"] = stage

        print("Input spans:", len(spans))
        print("Feature windows:", len(features))

        if not features.empty:
            print(
                features.head().to_string(
                    index=False
                )
            )

        datasets.append(features)

    combined = pd.concat(
        datasets,
        ignore_index=True
    )

    output = DATA_DIR / "ad_window_features.csv"

    combined.to_csv(output, index=False)

    print("\nFeature engineering completed")
    print("Output:", output)

    print("\nWindow counts by stage:")
    print(
        combined.groupby("stage")
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
