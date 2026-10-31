"""Step 2: train & evaluate Logistic Regression and Random Forest baselines.

Run:  python -m src.baselines      (after python -m src.preprocess)

Protocol
  * Fit on TRAIN only. Hyper-parameters chosen on VALIDATION ROC-AUC only.
  * TEST is used once, after selection, to produce the reported numbers.
  * Class decision threshold = 0.5 (headline). A validation-tuned (max-F1)
    threshold is reported as a supplementary row; it never uses test data.
  * Hand-off files for Member 3 are written in the row order of
    data/processed/split_test.csv.
"""
import itertools
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

from . import config as C


def load(name):
    d = np.load(C.PROC_DIR / f"{name}.npz")
    return d["X"], d["y"]


def metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return dict(threshold=round(float(thr), 4),
                accuracy=accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred),
                f1=f1_score(y, pred),
                roc_auc=roc_auc_score(y, prob),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def best_f1_threshold(y, prob):
    p, r, t = precision_recall_curve(y, prob)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[np.argmax(f1)])


MODELS = {
    "logistic_regression": dict(
        make=lambda C_: LogisticRegression(C=C_, max_iter=2000, solver="lbfgs",
                                           random_state=C.SEED),
        grid=[{"C_": c} for c in (0.01, 0.1, 1, 10, 100)]),
    "random_forest": dict(
        make=lambda n_estimators, max_depth, min_samples_leaf: RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=C.SEED),
        grid=[dict(n_estimators=300, max_depth=d, min_samples_leaf=l)
              for d, l in itertools.product((None, 8, 12), (1, 5, 10))]),
}


def main():
    C.PRED_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr = load("train"); Xva, yva = load("val"); Xte, yte = load("test")
    test_ids = pd.read_csv(C.PROC_DIR / "split_test.csv")[["RowNumber", "CustomerId"]]
    assert len(test_ids) == len(yte)

    report, search_log = {}, []
    for name, spec in MODELS.items():
        best = None
        for params in spec["grid"]:
            m = spec["make"](**params).fit(Xtr, ytr)
            auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
            search_log.append(dict(model=name, params=str(params), val_roc_auc=auc))
            if best is None or auc > best[0]:
                best = (auc, params, m)
        auc, params, model = best
        pva = model.predict_proba(Xva)[:, 1]
        pte = model.predict_proba(Xte)[:, 1]
        thr_val = best_f1_threshold(yva, pva)

        report[name] = dict(
            selected_params={k: (None if v is None else v) for k, v in params.items()},
            val=metrics(yva, pva),
            test=metrics(yte, pte),
            test_valtuned_threshold=metrics(yte, pte, thr_val))

        # hand-off for Member 3 (same order as split_test.csv)
        out = test_ids.copy()
        out["y_true"] = yte
        out["y_pred"] = (pte >= 0.5).astype(int)
        out["y_prob"] = pte
        out.to_csv(C.PRED_DIR / f"{name}_test_predictions.csv", index=False,
                   float_format="%.8f")
        joblib.dump(model, C.RES_DIR / f"{name}.joblib")

    labels = test_ids.copy(); labels["y_true"] = yte
    labels.to_csv(C.PRED_DIR / "test_labels.csv", index=False)

    (C.RES_DIR / "baseline_metrics.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(search_log).to_csv(C.RES_DIR / "baseline_validation_search.csv", index=False)

    rows = []
    for name, r in report.items():
        for split, key in (("val", "val"), ("test", "test"),
                           ("test (val-tuned thr)", "test_valtuned_threshold")):
            rows.append(dict(model=name, split=split, **{k: r[key][k] for k in
                        ("threshold", "accuracy", "precision", "recall", "f1", "roc_auc")}))
    tbl = pd.DataFrame(rows)
    tbl.to_csv(C.RES_DIR / "baseline_metrics.csv", index=False, float_format="%.4f")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for n, r in report.items():
        print(n, "selected:", r["selected_params"])


if __name__ == "__main__":
    main()
