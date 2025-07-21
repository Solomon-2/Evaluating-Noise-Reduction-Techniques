#!/bin/bash
# Quick evaluation script for command line usage

echo "=== Quick Noise Reduction Evaluation ==="
echo "Running unified evaluation on all methods..."

cd "$(dirname "$0")"

python src/evaluate_all_methods.py \
    --ground_csv tests/raw/output.csv \
    --denoised_dir tests/denoised \
    --output_csv results/quick_evaluation.csv

echo ""
echo "Results saved to: results/quick_evaluation.csv"
echo "Done!"
