"""Step 1: load raw data, create the shared stratified 70/15/15 split,
fit preprocessing on the TRAINING set only, and save processed arrays.

Run:  python -m src.preprocess
Outputs (data/processed/):
  split_{train,val,test}.csv   original rows (incl. RowNumber, CustomerId) in the
                               exact order used by every model
  {train,val,test}.npz         X (float64), y (int64)  -- same row order as above
  feature_names.txt            column names of X
  preprocessor.joblib          the fitted ColumnTransformer
"""
import hashlib
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config as C

GEOS = ["France", "Germany", "Spain"]


# SHA-256 of Churn_Modelling.csv with carriage returns removed (line-ending independent)
EXPECTED_SHA256 = "ee2a5c9d8daf3aed0d91c4d883a84ebb77d271447bebc0f0c80aafb0f288ceab"


def load_raw() -> pd.DataFrame:
    digest = hashlib.sha256(C.RAW_CSV.read_bytes().replace(b"\r", b"")).hexdigest()
    if digest != EXPECTED_SHA256:
        warnings.warn("Raw CSV differs from the file used for the reported results "
                      f"(sha256 {digest[:12]}...). Numbers may not match the paper.")
    df = pd.read_csv(C.RAW_CSV)
    assert df.shape == (10_000, 14), df.shape
    assert df.isna().sum().sum() == 0 and not df.duplicated().any()
    return df


def make_split(df: pd.DataFrame):
    """Stratified 70/15/15. Two-step: 70 / 30, then the 30 is halved.
    Each part is sorted by RowNumber so the order is deterministic and auditable."""
    train, rest = train_test_split(
        df, test_size=C.VAL_FRAC + C.TEST_FRAC,
        stratify=df[C.TARGET], random_state=C.SEED)
    val, test = train_test_split(
        rest, test_size=C.TEST_FRAC / (C.VAL_FRAC + C.TEST_FRAC),
        stratify=rest[C.TARGET], random_state=C.SEED)
    return [p.sort_values("RowNumber").reset_index(drop=True) for p in (train, val, test)]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), C.NUMERIC_SCALED),
        ("bin", "passthrough", C.BINARY_PASSTHROUGH),
        ("gender", OrdinalEncoder(categories=[["Female", "Male"]]), C.BINARY_ENCODED),
        ("geo", OneHotEncoder(categories=[GEOS], sparse_output=False), C.ONEHOT),
    ])


def feature_names():
    return (list(C.NUMERIC_SCALED) + list(C.BINARY_PASSTHROUGH)
            + ["Gender_Male"] + [f"Geography_{g}" for g in GEOS])


def main():
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    train, val, test = make_split(df)

    drop = [C.TARGET] + C.ID_COLS
    pre = build_preprocessor().fit(train.drop(columns=drop))      # fitted on TRAIN only
    for name, part in zip(["train", "val", "test"], [train, val, test]):
        X = pre.transform(part.drop(columns=drop)).astype("float64")
        y = part[C.TARGET].to_numpy("int64")
        np.savez(C.PROC_DIR / f"{name}.npz", X=X, y=y)
        part.to_csv(C.PROC_DIR / f"split_{name}.csv", index=False)
        print(f"{name:5s}: n={len(part):5d}  churners={y.sum():4d} ({y.mean():.4f})  X={X.shape}")

    (C.PROC_DIR / "feature_names.txt").write_text("\n".join(feature_names()) + "\n")
    joblib.dump(pre, C.PROC_DIR / "preprocessor.joblib")


if __name__ == "__main__":
    main()
