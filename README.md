# Bank customer churn under limited contact capacity

CAI 421 Checkpoint 2 project. We compare Logistic Regression (LR), Random Forest (RF), and a multilayer perceptron (MLP) on a shared stratified bank churn split. We evaluate classification and the top 10% and 20% of customers ranked by predicted churn probability.

## Requirements and reproduction

Python 3.12 and CPU were used for the supplied run. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash reproduce.sh
```

On Windows without Bash, run the modules individually:

```bash
python -m src.preprocess
python -m src.baselines
python -m src.mlp
python -m src.evaluate
python -m tests.test_pipeline
```

All randomness uses seed 42 from `src/config.py`. The dataset is [Churn for Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers). Its raw CSV belongs at `data/raw/Churn_Modelling.csv`.

## Shared split and models

The 10,000 records have 14 original columns and 2,037 churners (`Exited=1`). `RowNumber`, `CustomerId`, and `Surname` do not enter any model. The stratified train/validation/test split contains 7,000/1,500/1,500 customers. All three models use this same split.

LR and RF serve as baselines. The MLP compares hidden layers (32,16) and (64,32) using validation ROC-AUC and selects (32,16). Classification results use a 0.5 threshold.

## Test results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8127 | 0.6818 | 0.1475 | 0.2426 | 0.7956 |
| Random Forest | 0.8673 | 0.8397 | 0.4295 | 0.5683 | 0.8684 |
| MLP | 0.8647 | 0.7550 | 0.4951 | 0.5980 | 0.8648 |

The project also evaluates Precision@k and Recall@k for the top 10% and 20% of customers. See `results/topk_comparison.csv` after all project files have been added.

## Project files

- `src/`: preprocessing, baseline models, MLP, and evaluation code.
- `data/raw/`: original dataset.
- `data/processed/`: shared data splits and processed arrays.
- `results/`: metrics, predictions, and saved models.
- `paper/`: Checkpoint 2 manuscript and architecture diagram.
- `requirements.txt` and `reproduce.sh`: dependencies and reproduction commands.
