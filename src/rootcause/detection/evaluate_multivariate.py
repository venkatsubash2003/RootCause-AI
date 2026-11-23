
from pathlib import Path

import pandas as pd

from rootcause.detection.multivariate_detector import (
    FEATURE_COLUMNS,
    MultivariateAnomalyDetector
)


DATA_PATH = Path(
    "data/processed/ad_window_features.csv"
)


def main():
    df = pd.read_csv(DATA_PATH)

    df["window_start"] = pd.to_datetime(
        df["window_start"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=["window_start"] + FEATURE_COLUMNS
    )

    datasets = {
        stage: df[df["stage"] == stage]
        .sort_values("window_start")
        .copy()
        for stage in ["healthy", "faulty", "recovery"]
    }

    healthy = datasets["healthy"]

    split = int(len(healthy) * 0.7)

    train = healthy.iloc[:split]
    healthy_test = healthy.iloc[split:]

    if len(train) < 20 or len(healthy_test) < 5:
        raise ValueError(
            "Insufficient healthy windows for "
            "training and held-out evaluation"
        )

    model = MultivariateAnomalyDetector(
        contamination=0.05,
        n_estimators=200,
        random_state=42
    )

    model.fit(train[FEATURE_COLUMNS].to_numpy())

    print("\nMULTIVARIATE ISOLATION FOREST")
    print("=" * 45)
    print("Training windows:", len(train))

    for stage, stage_df in [
        ("Held-out healthy", healthy_test),
        ("Fault", datasets["faulty"]),
        ("Recovery", datasets["recovery"])
    ]:
        if stage_df.empty:
            print(f"\n{stage}: no observations")
            continue

        predictions = model.predict(
            stage_df[FEATURE_COLUMNS].to_numpy()
        )

        flags = predictions["is_anomaly"]

        print(f"\n{stage}")
        print("Windows:", len(stage_df))
        print("Anomalies:", int(flags.sum()))
        print(f"Anomaly rate: {flags.mean():.2%}")


if __name__ == "__main__":
    main()
