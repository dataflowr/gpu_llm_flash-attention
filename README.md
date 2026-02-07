# Flash-Attention in Triton

Learn how to implement [FlashAttention-2](https://arxiv.org/abs/2307.08691) from scratch using [Triton](https://triton-lang.org/main/index.html), a Python-based language for writing GPU kernels.

## Table of Contents

- [Course Material](#course-material)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Homework](#homework)
- [Running Tests](#running-tests)
- [Benchmarking](#benchmarking)
- [Submission](#submission)

## Course Material

The course notebook covers:
- The FlashAttention algorithm and its memory-efficient approach
- Online softmax computation
- Implementing attention kernels in Triton

📓 **Notebook:** [FlashAttention_empty.ipynb](https://github.com/dataflowr/gpu_llm_flash-attention/blob/main/FlashAttention_empty.ipynb)

### Running the Notebook

You need access to a GPU. Choose one of these options:

| Platform | Link |
|----------|------|
| SSP Cloud (recommended) | [Launch on Datalab](https://datalab.sspcloud.fr/launcher/ide/jupyter-pytorch-gpu?autoLaunch=true&name=flash-attention&init.personalInit=%C2%ABhttps://raw.githubusercontent.com/dataflowr/gpu_llm_flash-attention/refs/heads/main/utils/open-notebook.sh%C2%BB) |
| Google Colab | [Open in Colab](https://colab.research.google.com/github/dataflowr/gpu_llm_flash-attention/blob/main/FlashAttention_empty.ipynb) |

> **Note:** SSP Cloud requires account creation on [datalab.sspcloud.fr](https://datalab.sspcloud.fr/)

## Getting Started

### Requirements

- Python >= 3.8
- CUDA-capable GPU
- PyTorch, Triton, NumPy, Pandas, Matplotlib, Einops

### Installation

```bash
pip install -e .
```


## Project Structure

```
├── flash_attention/       # Flash Attention implementations (TODO)
├── online_softmax/        # Online softmax algorithm
├── softmax_matmul/        # Softmax-matmul kernel (TODO)
├── benchmarking/          # Performance benchmarks (TODO)
├── tests/                 # Test suite
└── FlashAttention_empty.ipynb  # Course notebook
```

## Homework

After completing the course, implement the full Flash-Attention algorithm:

1. **Softmax-Matmul** — Verify your Triton implementation and benchmark it
2. **Flash-Attention in PyTorch** — Implement forward and backward passes
3. **Flash-Attention in Triton** — Port to Triton, test and benchmark

> ⚠️ **GPU Compatibility:** Triton is optimized for Hopper architecture (H100). There are known issues with Turing GPUs (T4). As a result, it might be difficult to have Triton code running properly on Turing GPUs and if possible, you should use a H100 for your Triton implementation of Flash-Attention.


## Running Tests

Run the full test suite:

```bash
pytest -v ./tests
```

Run specific test modules:

```bash
pytest tests/test_online_softmax.py -v   # Online softmax
pytest tests/test_softmax_matmul.py -v   # Softmax-matmul
pytest tests/test_flash_attention.py -v  # Flash Attention
pytest tests/test_attention.py -v        # Standard attention
```

## Benchmarking

Compare performance of different implementations:

```bash
python -m benchmarking.bench_online_softmax
python -m benchmarking.bench_softmax_matmul
python -m benchmarking.bench_attention
```

## Submission

Run the automated test and submission script:

```bash
bash test_and_submit.sh
```

This script will:
1. Detect and save your GPU type to `gpu_type.txt`
2. Install the package in editable mode
3. Run all tests and generate `test_results.xml`
4. Package everything into `llm2026-flashattention-submission.zip`

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.