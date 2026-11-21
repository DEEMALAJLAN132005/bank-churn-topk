#!/usr/bin/env bash
# Reproduce the shared split, all models, evaluation, figures, and integrity checks.
set -euo pipefail
cd "$(dirname "$0")"
python -m src.preprocess
python -m src.baselines
python -m src.mlp
python -m src.evaluate
python -m src.ablation_numofproducts
python -m tests.test_pipeline
python -m src.bootstrap_analysis
python -m src.plot_results
