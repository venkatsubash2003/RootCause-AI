
import pandas as pd

CSV_PATH = "data/processed/ad_spans.csv"

df = pd.read_csv(CSV_PATH)

df["duration_ms"] = pd.to_numeric(
    df["duration_ms"],
    errors="coerce"
)

df = df.dropna(
    subset=["service", "operation", "duration_ms"]
)

df = df[df["duration_ms"] >= 0]

summary = (
    df.groupby(["service", "operation"])
      .agg(
          samples=("duration_ms", "size"),
          median_ms=("duration_ms", "median"),
          p95_ms=(
              "duration_ms",
              lambda x: x.quantile(0.95)
          )
      )
      .sort_values("samples", ascending=False)
)

print(summary.head(20).to_string())
