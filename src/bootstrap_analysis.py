"""Paired bootstrap comparison of MLP and Random Forest.

Run from the project root:
    python -m src.bootstrap_analysis

Uses saved predictions only. No training or data splitting.
"""

import numpy as np
import pandas as pd

from . import config as C


N_RESAMPLES = 10000
FRACTIONS = [0.10, 0.20]
IDS = ["RowNumber", "CustomerId"]


def load_predictions(model_name, labels):
    path = C.PRED_DIR / f"{model_name}_test_predictions.csv"
    data = pd.read_csv(path)

    required = IDS + ["y_true", "y_prob"]
    if not set(required).issubset(data.columns):
        raise ValueError(f"{model_name}: missing required columns.")

    data = data[required].sort_values(IDS).reset_index(drop=True)

    if data.isna().any().any() or data.duplicated(IDS).any():
        raise ValueError(f"{model_name}: missing values or duplicate IDs.")

    if not data[IDS + ["y_true"]].equals(labels):
        raise ValueError(f"{model_name}: customers or labels do not match.")

    probabilities = data["y_prob"].to_numpy(dtype=float)

    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(f"{model_name}: invalid probabilities.")

    return probabilities


def captured_churners(y, probabilities, row_ids, customer_ids, ks):
    # Highest probability first; resolve ties using identifiers.
    order = np.lexsort((customer_ids, row_ids, -probabilities))
    cumulative_churners = np.cumsum(y[order])
    return cumulative_churners[np.asarray(ks) - 1]


def main():
    labels = pd.read_csv(C.PRED_DIR / "test_labels.csv")
    labels = labels[IDS + ["y_true"]].sort_values(IDS)
    labels = labels.reset_index(drop=True)

    if labels.empty or labels.isna().any().any():
        raise ValueError("Test labels are empty or contain missing values.")

    if labels.duplicated(IDS).any():
        raise ValueError("Duplicate customer identifiers.")

    if set(labels["y_true"].unique()) != {0, 1}:
        raise ValueError("Test labels must contain both classes.")

    mlp_prob = load_predictions("mlp", labels)
    rf_prob = load_predictions("random_forest", labels)

    y = labels["y_true"].to_numpy(dtype=int)
    row_ids = labels["RowNumber"].to_numpy()
    customer_ids = labels["CustomerId"].to_numpy()

    n = len(y)
    total_churners = int(y.sum())
    ks = np.array([int(np.ceil(f * n)) for f in FRACTIONS])

    observed_mlp = captured_churners(
        y, mlp_prob, row_ids, customer_ids, ks
    )
    observed_rf = captured_churners(
        y, rf_prob, row_ids, customer_ids, ks
    )
    observed_difference = observed_mlp - observed_rf

    # Dimensions: resample, contact capacity, metric.
    # Metric 0 = Precision@k; metric 1 = Recall@k.
    differences = np.full((N_RESAMPLES, len(ks), 2), np.nan)
    rng = np.random.default_rng(C.SEED)

    print(f"Starting {N_RESAMPLES:,} paired bootstrap resamples...")

    for b in range(N_RESAMPLES):
        # Sample customers with replacement.
        # Both models use exactly the same sampled customers.
        indices = rng.integers(0, n, size=n)
        sampled_y = y[indices]

        mlp_count = captured_churners(
            sampled_y,
            mlp_prob[indices],
            row_ids[indices],
            customer_ids[indices],
            ks,
        )
        rf_count = captured_churners(
            sampled_y,
            rf_prob[indices],
            row_ids[indices],
            customer_ids[indices],
            ks,
        )

        delta = mlp_count - rf_count
        differences[b, :, 0] = delta / ks

        sampled_churners = int(sampled_y.sum())
        if sampled_churners > 0:
            differences[b, :, 1] = delta / sampled_churners

        if (b + 1) % 2000 == 0:
            print(f"Completed {b + 1:,} resamples.")

    rows = []

    for j, (fraction, k) in enumerate(zip(FRACTIONS, ks)):
        for metric_index, metric in enumerate(
            ["precision_at_k", "recall_at_k"]
        ):
            values = differences[:, j, metric_index]
            valid = values[np.isfinite(values)]
            lower, upper = np.quantile(valid, [0.025, 0.975])

            denominator = k if metric_index == 0 else total_churners

            rows.append({
                "comparison": "MLP minus Random Forest",
                "contact_fraction": fraction,
                "k": int(k),
                "metric": metric,
                "observed_difference": observed_difference[j] / denominator,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "includes_zero": bool(lower <= 0 <= upper),
                "valid_resamples": len(valid),
                "seed": C.SEED,
                "method": "paired percentile bootstrap",
            })

    results = pd.DataFrame(rows)
    C.RES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = C.RES_DIR / "bootstrap_comparison.csv"
    results.to_csv(output_path, index=False, float_format="%.8f")

    print("\nDifferences are MLP minus Random Forest.")
    print("Positive values favor MLP; negative values favor Random Forest.")
    print(results.round(4).to_string(index=False))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()