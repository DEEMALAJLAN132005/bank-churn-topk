"""Step 3: train and evaluate the MLP.

Run:  python -m src.mlp      (after python -m src.preprocess)

Protocol
  * Fit on TRAIN only. Two architectures are compared on VALIDATION ROC-AUC.
  * Headline predictions use threshold 0.5. A validation-tuned (max-F1)
    threshold is reported as a supplementary result in results/mlp_metrics.json;
    it never uses test data.
  * Test predictions are written in the row order of data/processed/split_test.csv.
"""
import json
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from . import config as C
from .baselines import best_f1_threshold, metrics


def load_split(name):
    """Load the preprocessed features and labels for one data split."""
    data = np.load(C.PROC_DIR / f"{name}.npz")
    return data["X"], data["y"]


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    # Compare two candidate architectures using validation ROC-AUC.
    settings = [
        {"hidden_layer_sizes": (32, 16), "alpha": 0.001},
        {"hidden_layer_sizes": (64, 32), "alpha": 0.001},
    ]

    best_model = None
    best_setting = None
    best_auc = -1

    for setting in settings:
        model = MLPClassifier(
            **setting,
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=C.SEED,
        )
        fit_started = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - fit_started

        val_prob = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)
        print(f"{setting}: validation ROC-AUC = {val_auc:.4f}, "
              f"fit time = {fit_seconds:.3f} s ({model.n_iter_} epochs)")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model
            best_setting = {k: list(v) if isinstance(v, tuple) else v
                            for k, v in setting.items()}

    # Generate churn probabilities and class predictions for the test split.
    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    # Use the shared test split file to preserve the original customer order.
    test_ids = pd.read_csv(C.PROC_DIR / "split_test.csv")
    output = test_ids[["RowNumber", "CustomerId"]].copy()
    output["y_true"] = y_test
    output["y_pred"] = test_pred
    output["y_prob"] = test_prob

    # Save predictions for evaluation and comparison with the baseline models.
    C.PRED_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        C.PRED_DIR / "mlp_test_predictions.csv",
        index=False,
        float_format="%.8f",
    )

    # Supplementary: decision threshold tuned for F1 on VALIDATION only
    # (same rule as baselines.py), then applied once to the test split.
    val_prob = best_model.predict_proba(X_val)[:, 1]
    thr_val = best_f1_threshold(y_val, val_prob)
    report = {
        "selected_params": best_setting,
        "val": metrics(y_val, val_prob),
        "test": metrics(y_test, test_prob),
        "test_valtuned_threshold": metrics(y_test, test_prob, thr_val),
    }
    (C.RES_DIR / "mlp_metrics.json").write_text(json.dumps(report, indent=2))
    tuned = report["test_valtuned_threshold"]
    print(f"Validation-tuned threshold = {thr_val:.4f}: test precision "
          f"{tuned['precision']:.4f}, recall {tuned['recall']:.4f}, "
          f"F1 {tuned['f1']:.4f}")

    print(f"Best validation ROC-AUC: {best_auc:.4f}")
    print("Saved predictions to results/predictions/mlp_test_predictions.csv")


if __name__ == "__main__":
    main()
