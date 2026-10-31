"""Sanity tests. Run:  python -m tests.test_pipeline   (or: pytest -q)"""
import numpy as np
import pandas as pd

from src import config as C


def _splits():
    return {n: pd.read_csv(C.PROC_DIR / f"split_{n}.csv") for n in ("train", "val", "test")}


def test_sizes_and_stratification():
    s = _splits()
    assert [len(s[n]) for n in ("train", "val", "test")] == [7000, 1500, 1500]
    for n in s:
        assert abs(s[n][C.TARGET].mean() - 0.2037) < 0.002, n


def test_no_overlap_and_full_coverage():
    s = _splits()
    ids = [set(s[n].RowNumber) for n in ("train", "val", "test")]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert len(ids[0] | ids[1] | ids[2]) == 10_000


def test_scaler_fitted_on_train_only():
    tr = np.load(C.PROC_DIR / "train.npz")["X"]
    va = np.load(C.PROC_DIR / "val.npz")["X"]
    assert np.allclose(tr[:, :6].mean(0), 0, atol=1e-8)       # exactly 0 on train
    assert np.allclose(tr[:, :6].std(0), 1, atol=1e-8)
    assert not np.allclose(va[:, :6].mean(0), 0, atol=1e-8)   # not 0 on val => no leakage


def test_prediction_files_aligned_with_test_split():
    test = pd.read_csv(C.PROC_DIR / "split_test.csv")
    labels = pd.read_csv(C.PRED_DIR / "test_labels.csv")
    assert (labels.RowNumber.values == test.RowNumber.values).all()
    assert (labels.y_true.values == test[C.TARGET].values).all()
    for m in ("logistic_regression", "random_forest"):
        p = pd.read_csv(C.PRED_DIR / f"{m}_test_predictions.csv")
        assert (p.RowNumber.values == test.RowNumber.values).all()
        assert (p.CustomerId.values == test.CustomerId.values).all()
        assert (p.y_true.values == test[C.TARGET].values).all()
        assert p.y_prob.between(0, 1).all()
        assert ((p.y_prob >= 0.5).astype(int) == p.y_pred).all()


if __name__ == "__main__":
    for f in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        f(); print("PASS", f.__name__)
