# Flash-Attention in Triton

Learn how to implement [FlashAttention-2](https://arxiv.org/abs/2307.08691) from scratch using [Triton](https://triton-lang.org/main/index.html), a Python-based language for writing GPU kernels.

## Table of Contents

- [Course Material](#course-material)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Homework](#homework)
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
- PyTorch, Triton, NumPy, Pandas, Matplotlib, Einops, Jaxtyping

### Installation

```bash
pip install -e .
```

The `-e` flag installs the package in *editable* (development) mode. Instead of copying files, pip creates a link to your source code. This means any changes you make to the code take effect immediately without reinstalling.

To verify your installation, run:

```bash
pytest tests/test_online_softmax.py -v
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

1. [**Softmax-Matmul**](homework/01_softmax_matmul.md) — Verify your Triton implementation and benchmark it
2. [**Flash-Attention in PyTorch**](homework/02_flash_attention_pytorch.md) — Implement forward and backward passes
3. [**Flash-Attention in Triton**](homework/03_flash_attention_triton.md) — Port to Triton, test and benchmark

📄 **Complete instructions:** [homework_all.pdf](homework/homework_all.pdf)

> ⚠️ **GPU Compatibility:** Triton is optimized for Hopper architecture (H100). There are known issues with Turing GPUs (T4). As a result, it might be difficult to have Triton code running properly on Turing GPUs and if possible, you should use a H100 for your Triton implementation of Flash-Attention.

## Submission

Once you have completed the homework, run the submission script to execute all tests and benchmarks:

```bash
./test_and_submit.sh
```

The script will report which tests pass and which benchmarks complete successfully. If a benchmark fails, you will see an error message indicating which one needs attention.

## Leaderboard (Optional)

Want to go further? Try to improve the performance of your implementation using any optimization tricks you can think of.

**Rules:**
- You cannot change the function's input/output signature
- You must use Triton (no CUDA)
- The implementation must be your own work (no pre-existing implementations)

**Benchmarking:** Measure your performance on an H100 GPU using `benchmarking/submit_leaderboard.py`.

**Optimization ideas:**
- Tune tile sizes for your kernel (use Triton's autotune feature)
- Experiment with additional Triton configuration parameters
- Implement the backward pass directly in Triton instead of relying on `torch.compile`
- Use two separate passes for the backward computation: one for dQ and another for dK/dV, avoiding atomics or inter-block synchronization
- Exit program instances early during causal masking by skipping tiles that are entirely zeroed out
- Separate non-masked tiles from diagonal tiles: compute the former without index comparisons, and the latter with a single comparison