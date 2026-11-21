# Bank Customer Churn Prediction

This project (CAI 421) compares Logistic Regression, Random Forest, and a multilayer perceptron (MLP) on the same stratified 70/15/15 split (seed 42). The pipeline reports classification and ranking metrics at contact capacities of 10% and 20% of the held-out test set.

## Setup

Use Python 3.12 on CPU. From the repository root (the directory containing `requirements.txt`), create a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, create the environment and install packages in Command Prompt:

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The input CSV from the Kaggle Churn for Bank Customers dataset is included at `data/raw/Churn_Modelling.csv`. See the [dataset notes](data/raw/README.md) for the source and license.

## Reproduce the results

With the environment activated, run:

```bash
bash reproduce.sh
```

On Windows without Bash, run these commands from the same directory:

```bat
.venv\Scripts\python.exe -m src.preprocess
.venv\Scripts\python.exe -m src.baselines
.venv\Scripts\python.exe -m src.mlp
.venv\Scripts\python.exe -m src.evaluate
.venv\Scripts\python.exe -m src.ablation_numofproducts
.venv\Scripts\python.exe -m tests.test_pipeline
.venv\Scripts\python.exe -m src.bootstrap_analysis
.venv\Scripts\python.exe -m src.plot_results
```

The preprocessing transform is fitted on the 7,000 training customers and yields 12 features. The external validation set (1,500 customers) selects configurations by ROC-AUC. The MLP compares hidden widths (32, 16) and (64, 32), using Adam, ReLU hidden activations, a learning rate of 0.001, L2 alpha of 0.001, and a maximum of 500 epochs. Scikit-learn's automatic batch size gives 200 samples. Early stopping monitors accuracy on an internal 10% holdout from the training split with tolerance 0.0001 and `n_iter_no_change=20`. The MLP script prints wall-clock fit time and epochs for each candidate.

## Outputs

| Location | Contents |
| --- | --- |
| `data/processed/` | Split CSVs, transformed arrays, feature names, and the fitted preprocessing object |
| `results/predictions/` | Test labels and aligned class predictions and churn probabilities |
| `results/baseline_*` and `results/*.joblib` | Baseline metrics, validation search, and fitted baselines |
| `results/mlp_metrics.json` | MLP metrics at threshold 0.5 and at the validation-tuned threshold (supplementary), with the selected architecture |
| `results/ablation_numofproducts.csv` | Test ROC-AUC of Random Forest and MLP with and without the NumOfProducts input |
| `results/overall_comparison.csv` | Accuracy, precision, recall, F1, and ROC-AUC on the 1,500 test customers |
| `results/topk_comparison.csv` | Precision@k and Recall@k for k = 150 and 300 |
| `results/bootstrap_comparison.csv` | Approximate 95% percentile intervals for the MLP minus Random Forest differences across 10,000 paired resamples (seed 42) |
| `results/figures/topk_comparison.png` and `results/figures/topk_comparison.svg` | Precision@k and Recall@k comparison figure |

After a full run, `python -m src.bootstrap_analysis` reruns the paired analysis using the saved test predictions. `python -m src.plot_results` recreates the figure from the saved top-k table.

Source modules are in `src/`, integrity checks are in `tests/`, dataset notes are in `docs/`, and the paper materials are in `paper/`.

## Paper

- [Bank Customer Churn Prediction (PDF)](paper/Bank_Churn_Paper_GitHub.pdf)
- [Experimental pipeline and MLP architecture (PDF)](paper/architecture_diagram.pdf)

## Verification run

The full Bash workflow passed on an Apple M4 CPU with 24 GB RAM, Python 3.12.14, and scikit-learn 1.8.0. MLP fit times were 0.623 s for (32, 16) and 0.409 s for (64, 32), measured around each `fit` call. These timings describe the verification run. A second full run on Python 3.13.x with scikit-learn 1.9.1 reproduced the reported test metrics, top-k counts, and bootstrap intervals; all four bootstrap intervals include zero.
