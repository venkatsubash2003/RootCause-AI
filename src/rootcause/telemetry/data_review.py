import pandas as pd

df = pd.read_csv("../../../data/processed/ad_spans.csv")

summary = (
    df.groupby(["service", "operation"])
    ["duration_ms"]
    .agg(["count", "median", "mean", "max"])
)

print(summary.sort_values("count", ascending=False).head(15))