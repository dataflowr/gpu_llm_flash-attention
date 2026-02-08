#!/usr/bin/env bash
set -uo pipefail

# Save GPU type to file
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null > gpu_type.txt || echo 'No NVIDIA GPU detected' > gpu_type.txt
echo "GPU type saved to gpu_type.txt"

# Install package in editable mode
pip install -e . -q

echo ""
echo "========================================"
echo "Running tests"
echo "========================================"
pytest -v ./tests --junitxml=test_results.xml || true
echo "Done running tests"

echo ""
echo "========================================"
echo "Running benchmarks"
echo "========================================"

# Track benchmark results
benchmark_failures=()

# Benchmark 1: Softmax Matmul
echo ""
echo "----------------------------------------"
echo "Benchmark: Softmax Matmul"
echo "----------------------------------------"
if python benchmarking/bench_softmax_matmul.py; then
    echo "✓ Softmax Matmul benchmark completed successfully"
else
    echo "✗ Softmax Matmul benchmark FAILED"
    benchmark_failures+=("bench_softmax_matmul.py")
fi

# Benchmark 2: Attention
echo ""
echo "----------------------------------------"
echo "Benchmark: Attention"
echo "----------------------------------------"
if python benchmarking/bench_attention.py; then
    echo "✓ Attention benchmark completed successfully"
else
    echo "✗ Attention benchmark FAILED"
    benchmark_failures+=("bench_attention.py")
fi

# Summary
echo ""
echo "========================================"
echo "Summary"
echo "========================================"
if [ ${#benchmark_failures[@]} -eq 0 ]; then
    echo "All benchmarks completed successfully!"
else
    echo "The following benchmarks failed:"
    for failed in "${benchmark_failures[@]}"; do
        echo "  - $failed"
    done
    echo ""
    echo "Please check the errors above and fix your implementation."
fi

echo ""
echo "Done!"