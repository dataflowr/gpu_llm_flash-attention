# Part 1: Fused Softmax-Matmul in Triton [8 points]

In this first part, you will implement a **fused softmax-matmul** kernel in Triton. This operation computes `softmax(X) @ V` without materializing the full softmax output matrix, which is a key building block for Flash Attention.

## Objectives

- Extend the online softmax algorithm to include matrix multiplication
- Implement a memory-efficient fused kernel in Triton
- Benchmark your implementation against the naive PyTorch version

## Background

### From Online Softmax to Fused Softmax-Matmul

During the course, we implemented the **online softmax** algorithm in Triton. The key insight was that we can compute softmax row by row using running statistics:

```
For each block j:
    m_j = max(m_{j-1}, max(X_j))           # running max
    l_j = l_{j-1} * exp(m_{j-1} - m_j) + sum(exp(X_j - m_j))   # running sum
```

Now we extend this to **simultaneously** compute the matrix multiplication with V:

```
output = softmax(X) @ V
```

where:
- `X` has shape `(batch, d1, d2)` — the attention scores
- `V` has shape `(batch, d2, d3)` — the values
- `output` has shape `(batch, d1, d3)`

### The Fused Algorithm

The key observation is that we can accumulate the output incrementally as we process blocks of X and V:

```
For each block j:
    # Update running max and sum (same as online softmax)
    m_j = max(m_{j-1}, max(X_j))
    exp_X_j = exp(X_j - m_j)
    l_j = l_{j-1} * exp(m_{j-1} - m_j) + sum(exp_X_j)

    # Update output accumulator with rescaling
    scale = (l_{j-1} / l_j) * exp(m_{j-1} - m_j)
    normalized_j = exp_X_j / l_j
    O_j = O_{j-1} * scale + normalized_j @ V_j
```

This avoids storing the full `(d1, d2)` softmax matrix in memory.

## Your Task

### File Structure

The implementation of the **online softmax** (seen during the course) is given in the following files:

```
online_softmax/
└── online_softmax.py
```
with the associated tests provided in:

```
tests
└── test_online_softmax.py
```

You can run the tests for the **online softmax** with:
```bash
pytest tests/test_online_softmax.py -v
```

To complete this part 1, you will need to complete the following files:

```
softmax_matmul/
└── softmax_matmul.py

benchmarking/
└── bench_softmax_matmul.py
```
The file `sofmmax_matmul/softmax_matmul.py` will contain your Fused Softmax-Matmul Implementation (Part 1.A) and the file  `benchmarking/bench_softmax_matmul.py` will contain the code for benchmarking your implementation (Part 1.B).

### Part 1.A: Fused Softmax-Matmul Implementation [4 points]

The `softmax_matmul.py` already contains:

#### 1. Reference Implementation

```python
def softmax_mult(x, V, dim=-1):
    """
    Reference implementation using PyTorch.

    Args:
        x: Input tensor of shape (batch, d1, d2)
        V: Value tensor of shape (batch, d2, d3)
        dim: Dimension for softmax (default: -1)

    Returns:
        Output tensor of shape (batch, d1, d3)
    """
    return F.softmax(x, dim=dim) @ V
```

Your `softmax_matmul.py` must implement:

#### 2. Triton Kernel

```python
@triton.jit
def fused_softmax_kernel(
    x_ptr,
    V_ptr,
    output_ptr,
    # ... strides and dimensions
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    BLOCK_1: tl.constexpr,
    BLOCK_2: tl.constexpr,
):
    """
    Fused softmax-matmul kernel.

    Computes softmax(X) @ V for a block of rows.
    """
    # Your code here
```

#### 3. Wrapper Function

```python
def fused_softmax(x, V, BLOCK_1=16, BLOCK_2=16):
    """
    Compute fused softmax(x) @ V using Triton.

    Args:
        x: Input tensor of shape (batch, d1, d2)
        V: Value tensor of shape (batch, d2, d3)
        BLOCK_1: Block size for d1 dimension
        BLOCK_2: Block size for d2 dimension

    Returns:
        Output tensor of shape (batch, d1, d3)
    """
    # Your code here
```

## Implementation Hints

If you already implemented this during the practicals, you can just copy and paste your code from the Jupyter Notebook to the file.

### Starting Point

Use the `online_softmax` implementation from the [FlashAttention_empty.ipynb ](https://github.com/dataflowr/gpu_llm_flash-attention/blob/main/FlashAttention_empty.ipynb) notebook as a template. The main differences are:

1. **Additional input**: You now have a second tensor `V`
2. **Output shape**: Output is `(batch, d1, d3)` instead of `(batch, d1, d2)`
3. **Accumulator**: You need to maintain an output accumulator of shape `(BLOCK_1, d3)`

### Block Pointer Setup

You'll need three block pointers:
- `x_block`: iterates over blocks of X in the d2 dimension
- `V_block`: iterates over corresponding blocks of V in the d2 dimension
- `output_block`: stores the final result (no iteration needed)

### Numerical Stability

Use `float32` accumulators for:
- `m_prev`: running max
- `l_prev`: running sum
- `out_prev`: output accumulator

### GPU Compatibility Note

There's a known bug with Triton on Turing GPUs (T4, RTX 8000) that requires explicit casting for dot products. If you're on Turing:

```python
# Cast to float16 for dot product
... tl.dot(a.to(tl.float16), b.to(tl.float16)).to(tl.float32)
```

On Hopper (H100), you can skip the casting.

## Testing Your Implementation

Run the tests to verify correctness:

```bash
pytest tests/test_softmax_matmul.py -v
```

The tests check:
- Correctness against the reference implementation
- Output shape
- Numerical stability with large values
- Various tensor dimensions

---

### Part B: Benchmarking [4 points]

Once your implementation passes the tests, write a benchmark to compare performance.

#### File to Create

Create `benchmarking/bench_softmax_matmul.py` that compares your Triton implementation `fused_softmax` against the PyTorch version `softmax_mult`.

#### Benchmark Requirements

Your benchmark should:

1. **Compare both implementations**: `softmax_mult` (PyTorch) vs `fused_softmax` (Triton)

2. **Vary the sequence length** (d2): Test with increasing values, `[64, 128, 256, 512, 1024, 2048, 4096, 8192]`

3. **Test different block sizes**: Try `B=BLOCK_1=BLOCK_2` values `[16, 32, 64]`

4. **Measure execution time**: Use CUDA events for accurate GPU timing:
   ```python
   start = torch.cuda.Event(enable_timing=True)
   end = torch.cuda.Event(enable_timing=True)

   torch.cuda.synchronize()
   start.record()
   # ... run your function ...
   end.record()
   torch.cuda.synchronize()

   elapsed_ms = start.elapsed_time(end)
   ```

5. **Include warmup iterations**: Run a few iterations before timing to allow for JIT compilation

6. **Report results**: Print a summary table and save your results to CSV in `outputs/softmax_matmul_benchmark.csv`

The format of your CSV should have the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `batch_size` | int | Batch size used (16) |
| `d1` | int | First dimension (2048) |
| `d2` | int | Sequence length being varied |
| `d3` | int | Third dimension (512) |
| `triton` | bool | `True` for Triton implementation, `False` for PyTorch |
| `BLOCK` | Int64 (nullable) | Block size used (`16`, `32`, `64`), or `<NA>` for PyTorch |
| `forward_ms_mean` | float (nullable) | Mean forward pass time in milliseconds |
| `forward_ms_std` | float (nullable) | Standard deviation of forward pass time |
| `forward_peak_MiB` | float (nullable) | Peak GPU memory usage in MiB |

**Note:** Nullable columns will contain `None`/`<NA>` values for configurations that failed due to OOM or incompatible block sizes.

#### Required Parameters

```python
device = "cuda"
dtype = torch.float32
batch_size = 16
nb_warmup = 10
nb_passes = 100
d1 = 2048       
d2 = [64, 128, 256, 512, 1024, 2048, 4096, 8192]  
d3 = 512  
B = [16, 32, 64]
```

#### Running Your Benchmark

```bash
python -m benchmarking.bench_softmax_matmul
```

#### Expected Results

Your fused kernel should:
- Be **faster** than the naive implementation for large d2
- Use **less memory** (no intermediate softmax matrix stored)

#### Handling Errors

Some configurations will fail due to resource limits or incompatible parameters. Your benchmark should:

1. **Run all configurations** — Don't stop at the first error
2. **Catch and log errors** — Record which configs failed and why
3. **Clean up GPU memory** — Call `torch.cuda.empty_cache()` after OOM errors

Here's a pattern for robust benchmarking:

```python
from triton.compiler.errors import CompileTimeAssertionFailure
from triton.runtime.errors import OutOfResources

results = []

for d2_val in d2_values:
    for BLOCK in block_sizes:
        try:
            # Run benchmark for this configuration
            result = run_benchmark_config(d2_val, BLOCK, ...)
            results.append(result)

        except (RuntimeError, OutOfResources) as e:
            err_str = str(e).lower()
            if "out of memory" in err_str:
                print(f"OOM for d2={d2_val}, BLOCK={BLOCK}")
                torch.cuda.empty_cache()  # Free memory before next config
                # Record as failed result with None values
                results.append({
                    "d2": d2_val,
                    "BLOCK": BLOCK,
                    "forward_ms": None,  # Mark as failed
                })
            else:
                raise  # Re-raise unexpected errors

        except CompileTimeAssertionFailure:
            # Block size incompatible with dimensions
            print(f"Skipping: BLOCK={BLOCK} incompatible with d2={d2_val}")
            results.append({
                "d2": d2_val,
                "BLOCK": BLOCK,
                "forward_ms": None,
            })
```

**Common errors you might encounter:**

| Error | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: CUDA out of memory` | Tensor too large for GPU | Catch and continue, log as failed |
| `OutOfResources` | Triton kernel needs too many registers | Catch and continue |
| `CompileTimeAssertionFailure` | `d1 % BLOCK_1 != 0` or `d2 % BLOCK_2 != 0` | Skip this config |

**Tip:** Use `pandas.DataFrame` to collect results and display them nicely:

```python
import pandas as pd

df = pd.DataFrame(results)
df["BLOCK"] = df["BLOCK"].astype("Int64")  # Nullable int for None values
print(df.to_string())
df.to_csv("outputs/softmax_matmul_benchmark.csv", index=False)
```

---

