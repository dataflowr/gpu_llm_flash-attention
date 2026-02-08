# Part 3: Flash Attention in Triton [8 points (+ 4 points)]

In this final part, you will port your Flash Attention implementation to **Triton**. You have already implemented and understood the algorithm in Part 2 using PyTorch. Now you will write a GPU-optimized Triton kernel for the forward pass, leveraging the same online softmax algorithm but with explicit control over memory access patterns and parallelization.

## Objectives

- Port the Flash Attention forward pass to a Triton kernel
- Use block pointers and tiled computation for efficient GPU execution
- Integrate your Triton kernel with PyTorch's autograd system
- Benchmark your implementation against PyTorch's optimized `scaled_dot_product_attention`

## Prerequisites

This part assumes you have completed **Part 2: Flash Attention in PyTorch**. You should be familiar with:
- The Flash Attention forward algorithm (online softmax over tiles)
- The backward pass formulas ($dQ$, $dK$, $dV$)
- Saving the log-sum-exp $L$ for the backward pass
- Causal masking implementation

Refer to Part 2 for the algorithm details if needed.

---

## Your Task

### File Structure

You will complete the implementation in:

```
flash_attention/
└── flash_attention.py

benchmarking/
└── bench_attention.py
```

Your `flash_attention.py` should contain:
- `FlashAttentionPytorch`: Your implementation from Part 2
- `FlashAttentionTriton`: The new Triton-based implementation (this part)

### Part 3.A: Triton Forward Kernel [6 points]

#### 1. Triton Kernel

Implement the forward kernel that parallelizes over query blocks and batches:

```python
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    Flash Attention forward kernel using online softmax algorithm.

    Each program instance processes one query block for one batch element.
    """
    # Your implementation here
```

The kernel should:
- Use `tl.program_id(0)` for query block index and `tl.program_id(1)` for batch index
- Create block pointers for Q, K, V, O, and L
- Implement the same online softmax loop as in Part 2, but using Triton primitives
- Store both output `O` and log-sum-exp `L`

#### 2. Autograd Function

Wrap your Triton kernel in an autograd function:

```python
class FlashAttentionTriton(torch.autograd.Function):
    """Flash Attention using Triton kernel for forward pass."""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Forward pass using Triton kernel.

        Args:
            Q: Query tensor of shape (batch, seq_q, d)
            K: Key tensor of shape (batch, seq_k, d)
            V: Value tensor of shape (batch, seq_k, d)
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor of shape (batch, seq_q, d)
        """
        # Allocate output tensors O and L
        # Choose block sizes (e.g., BLOCK_Q = BLOCK_K = 64)
        # Configure grid: (num_query_blocks, batch_size)
        # Launch flash_fwd_kernel
        # Save tensors for backward
        # Return O

    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass — reuse your PyTorch implementation from Part 2.
        """
        Q, K, V, L, O = ctx.saved_tensors
        dQ, dK, dV, _ = attention_backward_impl(
            Q, K, V, L, O, dO, ctx.sqrt_d, ctx.is_causal
        )
        return dQ, dK, dV, None
```

**Note:** For the backward pass, you can directly reuse the `attention_backward_impl` function from Part 2. The forward pass produces the same outputs (O and L), so the backward pass is identical.

---

## Implementation Hints

### From PyTorch to Triton

Your Part 2 forward pass has this structure:
```python
for i in range(num_query_blocks):      # Outer loop over query blocks
    for j in range(num_key_blocks):    # Inner loop over key/value blocks
        # Online softmax update
```

In Triton, the **outer loop becomes parallelization**:
- Each program instance handles one query block (and one batch element)
- Only the inner loop over key/value blocks remains as an explicit loop

### Block Pointer Setup

Set up block pointers after computing batch and query block offsets:

```python
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    is_causal: tl.constexpr,
):
    """Flash Attention forward kernel using online softmax algorithm."""
    # Get program IDs
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)

    # Create block pointers for this batch and query tile
    Q_ptr = Q_ptr + pid_b * stride_qb
    # Create block pointer for queries (fixed position)
    Q_block = tl.make_block_ptr(
        Q_ptr,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D),
        order=(1, 0),
    )

# ...
```
where `scale` is $\frac{1}{\sqrt{d}}
$ 

### Causal Masking in Triton

You can use [`tl.arange`](https://triton-lang.org/main/python-api/generated/triton.language.arange.html) and [`tl.where`](https://triton-lang.org/main/python-api/generated/triton.language.where.html).


### Casting

Triton requires matching dtypes for matrix multiplications. You need to use `float32` for accumulators for numerical stability, you need to cast tensors appropriately:

1. **Before `tl.dot`**: Cast $P_j$ (the attention weights) to match $V_j$'s dtype:
   ```python
   O_i = O_i * alpha[:, None] + tl.dot(P.to(V_j.dtype), V_j)
   ```

2. **Before storing output**: Cast the final output back to the input dtype:
   ```python
   O_i = O_i.to(Q_i.dtype)
   tl.store(O_block, O_i, ...)
   ```

You can access dtypes using:
- `tensor.dtype` for Triton tensors
- `block_ptr.type.element_ty` for block pointers

### Grid Configuration

For grid configuration, use the same strategy as in Part 1: each Triton program instance will load only elements from a single batch index, and only write to a single query tile of $Q$, $O$ and $L$.

### GPU Compatibility Note

Triton works best on Ampere (A100) and Hopper (H100) GPUs. On older Turing GPUs (T4, RTX 8000), some kernels may fail to compile or produce incorrect results. If your code works on H100 but fails on T4, this is expected behavior — focus on newer GPU architectures for this assignment. 

---

## Testing Your Implementation

Run the tests to verify correctness:

```bash
# Test correctness (forward and backward)
pytest tests/test_flash_attention.py -v

# Test memory efficiency
pytest tests/test_flash_memory.py -v
```

The tests check:
- **Correctness**: Output matches reference implementation (forward and backward)
- **Causal masking**: Proper application of the causal mask
- **Output shape**: Correct tensor dimensions
- **Memory efficiency**: No quadratic-size tensors saved for backward pass
- **Saved tensors**: Correct tensors (Q, K, V, O, L) saved for backward

---

### Part 3.B: Benchmarking [2 points]

Once your implementation passes the tests, benchmark it against PyTorch's optimized attention.

#### File to Complete

Complete `benchmarking/bench_attention.py` to compare your Triton implementation against PyTorch's `scaled_dot_product_attention`.

#### Benchmark Requirements

Your benchmark should:

1. **Compare both implementations**:
   - `pytorch_sdpa`: `torch.nn.functional.scaled_dot_product_attention` (compiled)
   - `flash_triton`: Your `FlashAttentionTriton` implementation (compiled)

2. **Vary the context length**: Test with increasing sequence lengths:
   `[256, 1024, 4096, 8192, 16384]`

3. **Measure performance**:
   - Forward pass execution time
   - Backward pass execution time
   - Peak GPU memory usage
   - Saved activations memory

4. **Use CUDA events for accurate timing**
    and  **Include warmup iterations**

6. **Report results**: Print a summary table and save to CSV

#### Required Parameters

```python
device = "cuda"
dtype = torch.float32
batch_size = 8
nb_warmup = 10
nb_forward_passes = 100
nb_backward_passes = 100

d_models = [64]
context_lengths = [256, 1024, 4096, 8192, 16384]
```

#### CSV Output Format

Save results to `outputs/csv/attention_benchmark.csv` with columns:

| Column | Type | Description |
|--------|------|-------------|
| `implementation` | str | Implementation name (`pytorch_sdpa` or `flash_triton`) |
| `d_model` | int | Model dimension |
| `seq_len` | int | Sequence length |
| `forward_ms` | float | Forward pass time in milliseconds |
| `forward_peak_MiB` | float | Peak GPU memory during forward pass |
| `backward_ms` | float (nullable) | Backward pass time in milliseconds |
| `backward_peak_MiB` | float (nullable) | Peak GPU memory during backward pass |
| `saved_activations_MiB` | float | Memory for saved activations |
| `status` | str | `ok`, `OOM`, or `OOM(backward)` |
| `gpu` | str | GPU name |

#### Running Your Benchmark

```bash
# Run all implementations
python -m benchmarking.bench_attention

# Run only Triton implementation
python -m benchmarking.bench_attention --impl flash_triton

# Run only PyTorch SDPA
python -m benchmarking.bench_attention --impl pytorch_sdpa
```

#### Expected Results

Your Flash Attention implementation should:
- Match or approach PyTorch SDPA performance (which uses FlashAttention internally on supported hardware)
- Use **linear memory** for the forward pass (no N×N attention matrix)
- Scale to longer sequences than naive attention

---

## Bonus: Triton Backward Kernel [+4 points]

For bonus points, implement the backward pass in Triton as well. This requires writing a kernel that:

1. Recomputes attention scores tile-by-tile (like the forward pass)
2. Computes gradients $dQ$, $dK$, $dV$ using the formulas from Part 2
3. Uses atomic operations or careful accumulation for $dK$ and $dV$ (since multiple query blocks contribute to the same key/value gradients)


---
