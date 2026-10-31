# Bank Customer Churn Prediction

CAI 421 Checkpoint 2 project comparing Logistic Regression, Random Forest, and a Multilayer Perceptron (MLP) for bank customer churn prediction.

The models use the same stratified train, validation, and test split. Performance is evaluated using accuracy, precision, recall, F1-score, ROC-AUC, Precision@k, and Recall@k for the top 10% and 20% of customers ranked by predicted churn probability.

## Dataset

The project uses the [Churn for Bank Customers dataset from Kaggle](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers), which is listed under the CC0 Public Domain license.

The dataset contains:

- 10,000 customer records
- 14 original columns
- 2,037 customers with `Exited = 1`
- 7,963 customers with `Exited = 0`

The prediction target is `Exited`, where:

- `1` means the customer left the bank.
- `0` means the customer remained with the bank.

`RowNumber`, `CustomerId`, and `Surname` are excluded from the model inputs because they are identifiers rather than meaningful predictors.

## Data Split

The customers are divided using a stratified split:

- Training: 7,000 customers (70%)
- Validation: 1,500 customers (15%)
- Testing: 1,500 customers (15%)

Stratification maintains a similar churn proportion in all three sets. Random seed `42` is used to reproduce the same split and model results.

All models use the same customers and row order for a fair comparison.

## Preprocessing

Preprocessing is fitted using the training data only.

The pipeline:

- Standardizes `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, and `EstimatedSalary`.
- Retains `HasCrCard` and `IsActiveMember` as binary features.
- Encodes `Gender`.
- One-hot encodes `Geography`.
- Produces 12 model input features.

The fitted preprocessing object and transformed data files are stored in `data/processed/`.

## Models

### Logistic Regression

Logistic Regression is used as an interpretable linear baseline. The regularization parameter `C` is selected using validation ROC-AUC from:

```text
0.01, 0.1, 1, 10, 100
```

The selected value is `C = 0.01`.

### Random Forest

Random Forest is used as a nonlinear tree-based baseline. The validation search compares:

- `max_depth`: None, 8, and 12
- `min_samples_leaf`: 1, 5, and 10
- `n_estimators`: 300

The selected settings are:

```text
max_depth = 8
min_samples_leaf = 1
n_estimators = 300
```

### Multilayer Perceptron

The MLP compares two candidate hidden-layer architectures using validation ROC-AUC:

```text
(32, 16)
(64, 32)
```

The selected architecture is `(32, 16)`.

The MLP uses:

- ReLU activation
- Adam optimizer
- Learning rate: 0.001
- L2 regularization (`alpha`): 0.001
- Maximum iterations: 500
- Early stopping
- Random seed: 42

## Test Results

All primary classification results use a decision threshold of `0.5`.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8127 | 0.6818 | 0.1475 | 0.2426 | 0.7956 |
| Random Forest | 0.8673 | 0.8397 | 0.4295 | 0.5683 | 0.8684 |
| MLP | 0.8647 | 0.7550 | 0.4951 | 0.5980 | 0.8648 |

Random Forest achieves the highest accuracy, precision, and ROC-AUC. The MLP achieves the highest recall and F1-score.

The project also evaluates each model under limited contact capacity using Precision@k and Recall@k for the top 10% and 20% of customers ranked by predicted churn probability.

## Repository Structure

```text
bank-churn-topk/
├── data/
│   ├── raw/
│   │   ├── Churn_Modelling.csv
│   │   └── README.md
│   └── processed/
│       ├── feature_names.txt
│       ├── preprocessor.joblib
│       ├── split_train.csv
│       ├── split_val.csv
│       ├── split_test.csv
│       ├── train.npz
│       ├── val.npz
│       └── test.npz
├── docs/
│   └── dataset_section.md
├── paper/
│   ├── Checkpoint2_Bank_Churn.pdf
│   ├── architecture_diagram.pdf
│   └── README.md
├── results/
│   ├── predictions/
│   │   ├── logistic_regression_test_predictions.csv
│   │   ├── random_forest_test_predictions.csv
│   │   ├── mlp_test_predictions.csv
│   │   └── test_labels.csv
│   ├── baseline_metrics.csv
│   ├── baseline_metrics.json
│   ├── baseline_validation_search.csv
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── overall_comparison.csv
│   └── topk_comparison.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── baselines.py
│   ├── mlp.py
│   └── evaluate.py
├── tests/
│   └── test_pipeline.py
├── .gitignore
├── README.md
├── reproduce.sh
└── requirements.txt
```

## Requirements

The supplied results were produced using Python 3.12 on CPU.

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Reproducing the Project

From the repository root, run:

```bash
bash reproduce.sh
```

The script:

1. Preprocesses the dataset and creates the shared split.
2. Trains Logistic Regression and Random Forest.
3. Trains the MLP.
4. Evaluates all three models.
5. Runs pipeline integrity checks.

On Windows without Bash, run:

```bash
python -m src.preprocess
python -m src.baselines
python -m src.mlp
python -m src.evaluate
python -m tests.test_pipeline
```

## Outputs

- `data/processed/` contains the fixed data splits, transformed arrays, feature names, and fitted preprocessor.
- `results/predictions/` contains aligned test labels, class predictions, and churn probabilities.
- `results/overall_comparison.csv` contains the standard test metrics.
- `results/topk_comparison.csv` contains Precision@k and Recall@k results.
- `paper/Checkpoint2_Bank_Churn.pdf` contains the Checkpoint 2 paper.
- `paper/architecture_diagram.pdf` contains the final architecture and evaluation pipeline diagram.
