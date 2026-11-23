
from pathlib import Path

import pandas as pd

from rootcause.detection.evaluate_detector import (
    load_dataset
)
from rootcause.detection.latency_detector import (
    LatencyAnomalyDetector
)
from rootcause.detection.isolation_forest_detector import (
    IsolationForestDetector
)


DATA_DIR = Path("data/processed")

SERVICE = "ad"
OPERATION = "oteldemo.AdService/GetAds"


def load_experiment():
    """
    Load the three experiment stages.
    """

    datasets = {}

    for stage in ["healthy", "faulty", "recovery"]:
        path = DATA_DIR / f"ad_{stage}.csv"

        datasets[stage] = load_dataset(
            path, SERVICE, OPERATION
        )

        if datasets[stage].empty:
            raise ValueError(
                f"No GetAds observations in {path}"
            )

    # Ensure stages don't reuse the same spans.
    stages = list(datasets.items())

    for i, (name_a, df_a) in enumerate(stages):
        ids_a = set(zip(
            df_a["trace_id"], df_a["span_id"]
        ))

        for name_b, df_b in stages[i + 1:]:
            ids_b = set(zip(
                df_b["trace_id"], df_b["span_id"]
            ))

            if ids_a & ids_b:
                raise ValueError(
                    f"{name_a} and {name_b} share spans"
                )

    # Ensure stages are chronologically ordered.
    for (_, first), (_, second) in zip(
        stages, stages[1:]
    ):
        if (
            first["started_at_utc"].max()
            >= second["started_at_utc"].min()
        ):
            raise ValueError(
                "Experiment stages overlap in time"
            )

    return datasets


def train_models(healthy):
    """
    Train both algorithms on identical healthy data.
    """

    split = int(len(healthy) * 0.7)

    train_df = healthy.iloc[:split]
    test_df = healthy.iloc[split:]

    if len(train_df) < 20 or len(test_df) < 5:
        raise ValueError(
            "Insufficient healthy observations"
        )

    durations = train_df[
        "duration_ms"
    ].to_numpy()

    zscore_model = LatencyAnomalyDetector(
        threshold=3.5,
        min_samples=20
    ).fit(durations)

    isolation_model = IsolationForestDetector(
        contamination=0.05,
        n_estimators=200,
        random_state=42
    ).fit(durations)

    print("\nTraining completed")
    print("Training samples:", len(train_df))
    print("Healthy test samples:", len(test_df))

    return zscore_model, isolation_model, test_df


def compare_predictions(
    zscore_model,
    isolation_model,
    df,
    stage
):
    """
    Evaluate both models on identical observations.
    """

    results = df.copy()

    zscore_predictions = [
        zscore_model.predict(value)
        for value in results["duration_ms"]
    ]

    isolation_predictions = [
        isolation_model.predict(value)
        for value in results["duration_ms"]
    ]

    results["zscore_score"] = [
        p["anomaly_score"]
        for p in zscore_predictions
    ]

    results["isolation_score"] = [
        p["anomaly_score"]
        for p in isolation_predictions
    ]

    results["zscore_anomaly"] = [
        p["is_anomaly"]
        for p in zscore_predictions
    ]

    results["isolation_anomaly"] = [
        p["is_anomaly"]
        for p in isolation_predictions
    ]

    results["disagreement"] = (
        results["zscore_anomaly"]
        != results["isolation_anomaly"]
    )

    results["stage"] = stage

    total = len(results)

    z_count = int(results["zscore_anomaly"].sum())
    i_count = int(results["isolation_anomaly"].sum())
    disagreements = int(results["disagreement"].sum())

    summary = {
        "stage": stage,
        "observations": total,
        "zscore_flags": z_count,
        "isolation_flags": i_count,
        "zscore_rate_pct": round(
            100 * z_count / total, 2
        ),
        "isolation_rate_pct": round(
            100 * i_count / total, 2
        ),
        "disagreements": disagreements
    }

    return results, summary


def main():
    datasets = load_experiment()

    healthy = datasets["healthy"]

    zscore_model, isolation_model, healthy_test = (
        train_models(healthy)
    )

    evaluation_sets = {
        "healthy": healthy_test,
        "faulty": datasets["faulty"],
        "recovery": datasets["recovery"]
    }

    all_results = []
    summaries = []

    for stage, df in evaluation_sets.items():

        results, summary = compare_predictions(
            zscore_model,
            isolation_model,
            df,
            stage
        )

        all_results.append(results)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    print("\nMODEL COMPARISON")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    detailed = pd.concat(
        all_results,
        ignore_index=True
    )

    output_path = (
        DATA_DIR / "model_comparison.csv"
    )

    detailed.to_csv(output_path, index=False)

    print("\nDetailed results saved:", output_path)

    disagreements = detailed[
        detailed["disagreement"]
    ]

    print("\nMODEL DISAGREEMENTS")
    print("=" * 70)

    if disagreements.empty:
        print("Both models made identical classifications")
    else:
        columns = [
            "stage",
            "duration_ms",
            "zscore_anomaly",
            "isolation_anomaly",
            "zscore_score",
            "isolation_score"
        ]

        print(
            disagreements[columns].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
