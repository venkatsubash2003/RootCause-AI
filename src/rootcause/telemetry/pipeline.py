
import argparse
import csv
import json

from datetime import datetime, timezone
from pathlib import Path

from jaeger_client import get_traces
from trace_normalizer import normalize_traces


def parse_utc(value):
    """
    Convert ISO 8601 timestamp to a UTC datetime.
    """
    dt = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    if dt.tzinfo is None:
        raise ValueError(
            "Timestamp must include timezone information"
        )

    return dt.astimezone(timezone.utc)


def collect_telemetry(service_name, start_time, end_time):
    print(f"Collecting traces for: {service_name}")
    print("Start UTC:", start_time.isoformat())
    print("End UTC:", end_time.isoformat())

    raw_traces = get_traces(
        service_name=service_name,
        start_time=start_time,
        end_time=end_time
    )

    if not isinstance(raw_traces, dict):
        raise ValueError(
            "Unexpected Jaeger response type"
        )

    result = raw_traces.get("result", raw_traces)

    if not isinstance(result, dict):
        raise ValueError(
            "Unexpected Jaeger response structure"
        )

    if "resourceSpans" not in result:
        raise ValueError(
            "Jaeger response does not contain resourceSpans"
        )

    return raw_traces


def run_pipeline(service_name, start_time, end_time, stage):
    raw_traces = collect_telemetry(
        service_name,
        start_time,
        end_time
    )

    records = normalize_traces(raw_traces)

    # Keep spans belonging to the requested interval.
    filtered_records = []

    for record in records:
        started_at = record.get("started_at_utc")
        duration_ms = record.get("duration_ms")

        if not started_at or duration_ms is None:
            continue

        started_at = parse_utc(started_at)

        if duration_ms < 0:
            continue

        finished_at = started_at + __import__(
            "datetime"
        ).timedelta(milliseconds=duration_ms)

        if start_time <= started_at and finished_at < end_time:
            filtered_records.append(record)

    # Remove duplicate spans.
    unique_records = {}

    for record in filtered_records:
        key = (
            record.get("trace_id"),
            record.get("span_id")
        )

        if not all(key):
            continue

        unique_records[key] = record

    records = list(unique_records.values())

    if not records:
        raise ValueError(
            "No valid spans in the requested time window"
        )

    output_dir = Path("data")
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save the original response.
    raw_file = raw_dir / f"{service_name}_{stage}.json"

    with raw_file.open("w") as file:
        json.dump(raw_traces, file, indent=2)

    # Save the normalized records.
    csv_file = (
        processed_dir / f"{service_name}_{stage}.csv"
    )

    with csv_file.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )

        writer.writeheader()
        writer.writerows(records)

    print("\nPipeline completed")
    print("Normalized spans:", len(records))
    print("Raw JSON:", raw_file)
    print("Processed CSV:", csv_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--service", default="ad")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)

    parser.add_argument(
        "--stage",
        required=True,
        choices=["healthy", "faulty", "recovery"]
    )

    args = parser.parse_args()

    start_time = parse_utc(args.start)
    end_time = parse_utc(args.end)

    if start_time >= end_time:
        raise ValueError("Start must be before end")

    run_pipeline(
        service_name=args.service,
        start_time=start_time,
        end_time=end_time,
        stage=args.stage
    )


if __name__ == "__main__":
    main()
