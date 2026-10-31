"""Evaluate all three models, including top-10% and top-20% selection.

Run from the project folder:
    python -m src.evaluate
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from . import config as C


MODELS = ["logistic_regression", "random_forest", "mlp"]
CONTACT_FRACTIONS = [0.10, 0.20]
ID_COLUMNS = ["RowNumber", "CustomerId"]


def main():
    # The baseline script saves these shared test labels.
    labels = pd.read_csv(C.PRED_DIR / "test_labels.csv")
    required_labels = ID_COLUMNS + ["y_true"]

    if not set(required_labels).issubset(labels.columns):
        raise ValueError("test_labels.csv is missing required columns.")

    labels = labels[required_labels].copy()

    if labels.empty or labels.isna().any().any():
        raise ValueError("Test labels are empty or contain missing values.")

    if labels.duplicated(ID_COLUMNS).any():
        raise ValueError("Duplicate customer identifiers in test labels.")

    if set(labels["y_true"].unique()) != {0, 1}:
        raise ValueError("Test labels must contain both classes: 0 and 1.")

    overall_rows = []
    topk_rows = []

    for model_name in MODELS:
        path = C.PRED_DIR / f"{model_name}_test_predictions.csv"
        predictions = pd.read_csv(path)
        required = ID_COLUMNS + ["y_true", "y_pred", "y_prob"]

        if not set(required).issubset(predictions.columns):
            raise ValueError(f"{model_name}: missing required columns.")

        predictions = predictions[required].copy()

        if predictions.isna().any().any():
            raise ValueError(f"{model_name}: missing prediction values.")

        # Match by identifiers instead of assuming identical row order.
        data = labels.merge(
            predictions,
            on=ID_COLUMNS,
            how="outer",
            suffixes=("_reference", ""),
            indicator=True,
            validate="one_to_one",
        )

        if not data["_merge"].eq("both").all():
            raise ValueError(f"{model_name}: test customers do not match.")

        if not data["y_true"].eq(data["y_true_reference"]).all():
            raise ValueError(f"{model_name}: test labels do not match.")

        y_true = data["y_true"].to_numpy(dtype=int)
        probabilities = pd.to_numeric(
            data["y_prob"], errors="raise"
        ).to_numpy(dtype=float)

        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError(f"{model_name}: invalid probabilities.")

        y_pred = data["y_pred"].to_numpy()

        if not np.isin(y_pred, [0, 1]).all():
            raise ValueError(f"{model_name}: predictions must be 0 or 1.")

        # Use the supplied predictions made with threshold 0.5.
        # Probabilities in the hand-off files are rounded to 8 decimals.
        overall_rows.append({
            "model": model_name,
            "threshold": 0.5,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, probabilities),
        })

        data["y_prob"] = probabilities

        # Break probability ties by customer identifiers, never by labels.
        ranked = data.sort_values(
            ["y_prob", "RowNumber", "CustomerId"],
            ascending=[False, True, True],
        )

        total_churners = int(y_true.sum())
        n_test = len(data)

        for fraction in CONTACT_FRACTIONS:
            # Round up if the selected percentage is not a whole number.
            k = int(np.ceil(fraction * n_test))
            selected = ranked.head(k)
            captured = int(selected["y_true"].sum())

            topk_rows.append({
                "model": model_name,
                "contact_fraction": fraction,
                "n_test": n_test,
                "k": k,
                "actual_selected_fraction": k / n_test,
                "captured_churners": captured,
                "total_churners": total_churners,
                "precision_at_k": captured / k,
                "recall_at_k": captured / total_churners,
            })

    overall = pd.DataFrame(overall_rows)
    topk = pd.DataFrame(topk_rows)

    C.RES_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(
        C.RES_DIR / "overall_comparison.csv",
        index=False,
        float_format="%.8f",
    )
    topk.to_csv(
        C.RES_DIR / "topk_comparison.csv",
        index=False,
        float_format="%.8f",
    )

    print("\nOverall performance (threshold = 0.5):")
    print(overall.round(4).to_string(index=False))

    print("\nTop-10% and top-20% performance:")
    print(topk.round(4).to_string(index=False))

    print("\nSaved comparison tables in the results folder.")


if __name__ == "__main__":
    main()
