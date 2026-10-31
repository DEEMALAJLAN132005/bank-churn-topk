"""Shared configuration. Every member imports from here so the split,
seed and paths are identical across models."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "Churn_Modelling.csv"
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = ROOT / "results" / "predictions"
RES_DIR = ROOT / "results"

SEED = 42                      # the ONLY seed used anywhere in the project
TEST_FRAC = 0.15
VAL_FRAC = 0.15                # 70 / 15 / 15

TARGET = "Exited"
ID_COLS = ["RowNumber", "CustomerId", "Surname"]   # dropped, never used as features
NUMERIC_SCALED = ["CreditScore", "Age", "Tenure", "Balance",
                  "NumOfProducts", "EstimatedSalary"]
BINARY_PASSTHROUGH = ["HasCrCard", "IsActiveMember"]
BINARY_ENCODED = ["Gender"]            # Female=0, Male=1
ONEHOT = ["Geography"]                 # France / Germany / Spain
