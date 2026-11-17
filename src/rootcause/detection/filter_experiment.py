
import argparse
from pathlib import Path

import pandas as pd


def filter_experiment(input_path, output_path, start, end):
    """
    Filter Jaeger span records by an experiment's
    UTC time window and remove duplicate spans.
    """

    start_time = pd.to_datetime(start, utc=True)
    end_time = pd.to_datetime(end, utc=True)

    if pd.isna(start_time) or pd.isna(end_time):
        raise ValueError("Invalid experiment timestamps")

    if start_time >= end_time:
        raise ValueError(
            "Experiment start must be before end"
        )

    df = pd.read_csv(input_path)

    required_columns = {
        "trace_id",
        "span_id",
        "service",
        "operation",
        "duration_ms",
        "started_at_utc",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required CSV columns: {missing}"
        )

    # Convert timestamps into UTC
    df["started_at_utc"] = pd.to_datetime(
        df["started_at_utc"],
        utc=True,
        errors="coerce"
    )

    # Retain only observations inside the
    # selected experiment window.
    filtered = df[
        (df["started_at_utc"] >= start_time)
        & (df["started_at_utc"] < end_time)
    ].copy()

    # Remove repeated trace/span records.
    filtered = filtered.drop_duplicates(
        subset=["trace_id", "span_id"]
    )

    # Sort observations chronologically.
    filtered = filtered.sort_values(
        "started_at_utc"
    )

    # Create the output directory if necessary.
    output_file = Path(output_path)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Export filtered telemetry.
    filtered.to_csv(output_file, index=False)

    print("\nExperiment Filtering Complete")
    print("=" * 45)

    print("Input file:", input_path)
    print("Output file:", output_path)
    print("Start UTC:", start_time)
    print("End UTC:", end_time)

    print("\nOriginal spans:", len(df))
    print("Filtered spans:", len(filtered))

    ad_spans = filtered[
        filtered["service"] == "ad"
    ]

    print("Ad service spans:", len(ad_spans))

    if not ad_spans.empty:
        print("\nAd operations:")
        print(
            ad_spans["operation"]
            .value_counts()
            .to_string()
        )

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Filter RootCause AI experiment telemetry"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Original telemetry CSV"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Filtered experiment CSV"
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Experiment start timestamp in UTC"
    )

    parser.add_argument(
        "--end",
        required=True,
        help="Experiment end timestamp in UTC"
    )

    args = parser.parse_args()

    filter_experiment(
        input_path=args.input,
        output_path=args.output,
        start=args.start,
        end=args.end
    )


if __name__ == "__main__":
    main()
