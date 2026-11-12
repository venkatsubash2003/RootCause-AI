
import csv
import json
from pathlib import Path

from jaeger_client import get_traces
from trace_normalizer import normalize_traces



def collect_telemetry(service_name):
    print(f"Collecting traces for: {service_name}")

    raw_traces = get_traces(service_name)

    if not isinstance(raw_traces, dict):
        raise ValueError(
            "Unexpected Jaeger response type"
        )

    print("Response keys:", list(raw_traces.keys()))

    result = raw_traces.get("result", raw_traces)

    if not isinstance(result, dict):
        raise ValueError(
            "Unexpected Jaeger result structure"
        )

    if "resourceSpans" not in result:
        raise ValueError(
            "Jaeger response does not contain "
            "resourceSpans. Inspect the API response."
        )

    return raw_traces


def run_pipeline(service_name):
    raw_traces = collect_telemetry(service_name)

    output_dir = Path("data")
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the original API response
    raw_file = raw_dir / f"{service_name}_traces.json"

    with raw_file.open("w") as file:
        json.dump(raw_traces, file, indent=2)

    # Transform nested traces into records
    records = normalize_traces(raw_traces)

    if not records:
        raise ValueError(
            "No spans were returned for normalization"
        )

    csv_file = (
        processed_dir / f"{service_name}_spans.csv"
    )

    with csv_file.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"\nNormalized spans: {len(records)}")
    print(f"Raw traces saved: {raw_file}")
    print(f"Structured data saved: {csv_file}")


if __name__ == "__main__":
    run_pipeline("checkout")
