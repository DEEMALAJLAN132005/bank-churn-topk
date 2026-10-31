#!/usr/bin/env bash
# Reproduce the shared split, all models, evaluation, and integrity checks.
set -euo pipefail
cd "$(dirname "$0")"
python -m src.preprocess
python -m src.baselines
python -m src.mlp
python -m src.evaluate
python -m tests.test_pipeline
