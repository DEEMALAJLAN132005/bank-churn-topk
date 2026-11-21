"""Supplementary check: remove NumOfProducts from the inputs and refit.

Run:  python -m src.ablation_numofproducts   (after python -m src.baselines
      and python -m src.mlp)

Zhang et al. removed NumOfProducts to avoid label leakage in their multi-task
setting. This script measures how much that input change alone moves test
ROC-AUC for our Random Forest and MLP. Hyper-parameters are the ones already
selected on validation (read from the metrics files); nothing is re-tuned and
the test split is used only to report the result.
"""
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from . import config as C
from .baselines import load


def main():
    X_train, y_train = load("train")
    X_test, y_test = load("test")
    names = (C.PROC_DIR / "feature_names.txt").read_text().split()
    drop = names.index("NumOfProducts")
    keep = [i for i in range(len(names)) if i != drop]

    rf_params = json.loads((C.RES_DIR / "baseline_metrics.json").read_text())[
        "random_forest"]["selected_params"]
    mlp_params = json.loads((C.RES_DIR / "mlp_metrics.json").read_text())[
        "selected_params"]

    models = {
        "random_forest": RandomForestClassifier(
            **rf_params, n_jobs=-1, random_state=C.SEED),
        "mlp": MLPClassifier(
            hidden_layer_sizes=tuple(mlp_params["hidden_layer_sizes"]),
            alpha=mlp_params["alpha"], activation="relu", solver="adam",
            learning_rate_init=0.001, max_iter=500, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20, random_state=C.SEED),
    }
    rows = []
    for name, model in models.items():
        full = model.fit(X_train, y_train)
        auc_full = roc_auc_score(y_test, full.predict_proba(X_test)[:, 1])
        reduced = model.fit(X_train[:, keep], y_train)
        auc_reduced = roc_auc_score(
            y_test, reduced.predict_proba(X_test[:, keep])[:, 1])
        rows.append(dict(model=name, auc_all_inputs=auc_full,
                         auc_without_numofproducts=auc_reduced,
                         drop=auc_full - auc_reduced))
    out = pd.DataFrame(rows)
    out.to_csv(C.RES_DIR / "ablation_numofproducts.csv", index=False,
               float_format="%.4f")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
