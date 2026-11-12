import csv
from pathlib import Path


def save_records(records, output_path):
    if not records:
        raise ValueError("No span records to save")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )

        writer.writeheader()
        writer.writerows(records)

