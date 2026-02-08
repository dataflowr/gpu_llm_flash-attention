# Part 2: Flash Attention in PyTorch [8 points]

In this part, you will implement **Flash Attention** in pure PyTorch as a custom `torch.autograd.Function` with explicit forward and backward passes. This prepares you for the Triton implementation by understanding the algorithm deeply.

## Objectives

- Implement the Flash Attention forward pass using the tiling algorithm
- Implement the backward pass with gradient computation for Q, K, and V
- Wrap your implementation as a reusable PyTorch module
- Understand memory-efficient attention without relying on Triton

---

## Background

### Standard Attention vs Flash Attention

**Standard Attention** computes:

$$
X = \frac{QK^\top}{\sqrt d}, \qquad
P = \mathrm{softmax}(X), \qquad
O = PV
$$

This requires materializing the full $(N \times N)$ attention matrix, which has $O(N^2)$ memory complexity.

**Flash Attention** computes the same result without materializing the full score matrix by processing attention in **tiles (blocks)** and computing softmax **online** using a running max and running normalizer.

Memory becomes $O(N)$ instead of $O(N^2)$.

---


### The Algorithm: Almost-Flash-Attention

Here, we recall the algorithm presented in [FlashAttention_empty.ipynb](https://github.com/dataflowr/gpu_llm_flash-attention/blob/main/FlashAttention_empty.ipynb) computing `softmax(x) @ V` in a fused manner.

**Input:** $x_1, \ldots, x_{d_2} \in \mathbb{R}^{d_1}$ and $v \in \mathbb{R}^{d_2 \times d_3}$

**Output:** $o_1, \ldots, o_{d_2} \in \mathbb{R}^{d_3}$

**One pass:** 

$$
\begin{align}
m_0 &= -\infty \\
\ell'_0 &= 0 \\
o'_0 &= 0\\
\text{for } & i = 1, \ldots, d_2\\
m_i &\leftarrow \max(m_{i-1}, x_i) \\
\ell'_i &\leftarrow \ell'_{i-1}e^{m_{i-1}-m_i} + e^{x_i - m_i}\\
o'_i &\leftarrow e^{m_{i-1}-m_i}\frac{\ell'_{i-1}}{\ell'_i} \odot o'_{i-1} + \frac{e^{x_i - m_i}}{\ell'_i} \otimes v[i,:]
\end{align}
$$



---

## Flash Attention

#### Setup and notation

We use:

* $Q \in \mathbb{R}^{d_1 \times d}$
* $K \in \mathbb{R}^{d_2 \times d}$
* $V \in \mathbb{R}^{d_2 \times d_3}$

Scores:

$$
X = \frac{QK^\top}{\sqrt d} \quad\in\mathbb{R}^{d_1\times d_2}
$$

Probabilities (row-wise softmax):

$$
P = \mathrm{softmax}(X)
$$

Output:

$$
O = PV \quad\in\mathbb{R}^{d_1\times d_3}
$$

In practice, we have $d_1=d_2=d_3$, but we find it easier to present the algorithm where each dimension is identified by its size.

---

### Forward Pass — FlashAttention



We want to apply the above algorithm with $x = \frac{QK^T}{\sqrt{d}}$.
We use the following notations, for $j\in [d_1]$ and $i \in [d_2]$,
$Q^{(j)} = Q[j,:]\in \mathbb{R}^{d}$, $K_{i} = K[i,:], V_{i} = V[i,:] \in \mathbb{R}^d$ and $x_{ij} = x^{(j)}_i = \frac{Q^{(j)}(K_{i})^T}{\sqrt{d}}$ and $L \in \mathbb{R}^{d_1}$, with $L^{(j)} = \sum_{i\leq d_2}e^{x^{(j)}_i - \max_{i\leq d_2} x^{(j)}_i}$.


$$
\begin{align}
\text{for } j = 1, &\ldots, d_1 \text{ (outer loop)}\\
m_0 &= -\infty \\
\ell^{(j)}_0 &= 0 \\
o^{(j)}_0 &= 0\\
\text{for } & i = 1, \ldots, d_2 \text{ (inner loop)}\\
&x^{(j)}_i = \frac{Q^{(j)}(K_{i})^T}{\sqrt{d}}\\
&m_i \leftarrow \max(m_{i-1}, x_i^{(j)}) \\
&\ell^{(j)}_i \leftarrow \ell^{(j)}_{i-1}e^{m_{i-1}-m_i} + e^{x^{(j)}_i - m_i}\\
& o^{(j)}_i \leftarrow e^{m_{i-1}-m_i}\frac{\ell^{(j)}_{i-1}}{\ell^{(j)}_i} \odot o^{(j)}_{i-1} + \frac{e^{x_i - m_i}}{\ell^{(j)}_i} \otimes V_{i}
\end{align}
$$

Here, we have $x^{(j)}_i, m_i, \ell^{(j)}_i \in \mathbb{R}^{d_1}$ and $o^{(j)}_i\in \mathbb{R}^{d_3}$. Moreover, we have
$$
\begin{align}
o^{(j)}_{d_2} = O[j,:] \text{ and, } \ell^{(j)}_{d_2} = \sum_{i\leq d_2} e^{x^{(j)}_i - \max_{i\leq d_2} x^{(j)}_i} = L^{(j)}
\end{align}
$$

Hence, this algorithm computes $O$ and $L$.

In your implementation, you will work with tiles corresponding to blocks in the dimensions of $j$ and $i$ (for simplicity take the same block dimension in each direction).

---


### Backward Pass Derivation for Flash Attention



#### Goal

Given upstream gradient

$$
dO = \frac{\partial \mathcal L}{\partial O},
$$

compute $dQ$, $dK$, $dV$.

---

#### Computational graph

We differentiate **backwards** through:

$$
Q,K \rightarrow X \rightarrow P \rightarrow O
$$

So we apply the chain rule in three steps:

1. $O = PV$
2. $P = \mathrm{softmax}(X)$
3. $X = \frac{QK^\top}{\sqrt d}$

---

#### Step A — Gradient through $O = PV$

**Forward definition:**

$$
O_{i\ell} = \sum_{j=1}^{d_2} P_{ij} V_{j\ell}
$$

**A.1 — gradient w.r.t. $V$:**

$$
\frac{\partial O_{i\ell}}{\partial V_{j\ell}} = P_{ij}
$$

Thus:

$$
dV_{j\ell} = \sum_{i=1}^{d_1} P_{ij} \, dO_{i\ell}
$$

Matrix form:

$$
\boxed{dV = P^\top dO}
$$

**A.2 — gradient w.r.t. $P$:**

$$
\frac{\partial O_{i\ell}}{\partial P_{ij}} = V_{j\ell}
$$

Thus:

$$
dP_{ij} = \sum_{\ell=1}^{d_3} dO_{i\ell} \, V_{j\ell}
$$

Matrix form:

$$
\boxed{dP = dO \, V^\top}
$$


---

#### Step B — Gradient through softmax

Softmax is applied **row-wise**.

For each row $i$:

$$
P_{ij} = \frac{e^{X_{ij}}}{\sum_{k} e^{X_{ik}}}
$$

Let $Z_i = \sum_k e^{X_{ik}}$.

**B.1 — softmax derivative:**

Two cases:

If $j=k$:

$$
\frac{\partial P_{ij}}{\partial X_{ij}} = P_{ij}(1-P_{ij})
$$

If $j \neq k$:

$$
\frac{\partial P_{ij}}{\partial X_{ik}} = -P_{ij}P_{ik}
$$

**B.2 — compact Jacobian form:**

$$
\frac{\partial P_{ij}}{\partial X_{ik}} = P_{ij}(\delta_{jk}-P_{ik})
$$

**B.3 — apply chain rule:**

$$
dX_{ik} = \sum_j dP_{ij} \frac{\partial P_{ij}}{\partial X_{ik}}
$$

Substitute:

$$
dX_{ik} = \sum_j dP_{ij} P_{ij}(\delta_{jk}-P_{ik})
$$

Split:

$$
dX_{ik} = P_{ik} \, dP_{ik} - P_{ik} \sum_j P_{ij} \, dP_{ij}
$$

Define row scalar:

$$
D_i = \sum_j P_{ij} \, dP_{ij}
$$

So:

$$
\boxed{dX_{ik} = P_{ik}\big(dP_{ik}-D_i\big)}
$$

**Matrix form:**

$$
\boxed{dX = P \odot (dP - D)}
$$

where $D \in \mathbb{R}^{d_1\times 1}$ is broadcast across columns.

---

#### Useful simplification (used in FlashAttention)

We can avoid explicitly computing $P \odot dP$ by deriving an equivalent expression for $D$.

Recall that:
- $D_i = \sum_j P_{ij} \, dP_{ij}$ (from the softmax backward)
- $dP = dO \, V^\top$, so $dP_{ij} = \sum_\ell dO_{i\ell} \, V_{j\ell}$
- $O = PV$, so $O_{i\ell} = \sum_j P_{ij} \, V_{j\ell}$

Substituting the expression for $dP_{ij}$ into $D_i$:

$$
D_i = \sum_j P_{ij} \, dP_{ij} = \sum_j P_{ij} \sum_\ell dO_{i\ell} \, V_{j\ell}
$$

Rearranging the sums:

$$
D_i = \sum_\ell dO_{i\ell} \sum_j P_{ij} \, V_{j\ell}
$$

But $\sum_j P_{ij} \, V_{j\ell} = O_{i\ell}$ (by definition of $O$), so:

$$
\boxed{D_i = \sum_{\ell} O_{i\ell} \, dO_{i\ell}}
$$

In matrix form:

$$
D = \mathrm{rowsum}(O \odot dO)
$$

**Why this matters for FlashAttention:** This identity allows us to compute $D$ directly from the output $O$ and its gradient $dO$, without needing to explicitly compute $P \odot dP$ first.

Note that we still need $P$ in the backward pass (for $dV = P^\top dO$ and $dX = P \odot (dP - D)$). The key insight of FlashAttention is that instead of **storing** the full $O(N^2)$ matrix $P$ from the forward pass, we **recompute** it on-the-fly during the backward pass using the saved log-sum-exp values $L$:

$$
P = e^{S - L}, \quad \text{where } S = \frac{QK^\top}{\sqrt{d}}
$$

This recomputation is done block by block, so we never materialize the full $P$ matrix in memory.

---

#### Step C — Gradient through $X = \frac{QK^\top}{\sqrt d}$

**Forward definition:**

$$
X_{ij} = \frac{1}{\sqrt d} \sum_{r=1}^d Q_{ir} K_{jr}
$$

**C.1 — gradient w.r.t. $Q$:**

$$
\frac{\partial X_{ij}}{\partial Q_{ir}} = \frac{K_{jr}}{\sqrt d}
$$

Thus:

$$
dQ_{ir} = \frac{1}{\sqrt d} \sum_j dX_{ij} K_{jr}
$$

Matrix form:

$$
\boxed{dQ = \frac{dX \, K}{\sqrt d}}
$$

**C.2 — gradient w.r.t. $K$:**

$$
\frac{\partial X_{ij}}{\partial K_{jr}} = \frac{Q_{ir}}{\sqrt d}
$$

Thus:

$$
dK_{jr} = \frac{1}{\sqrt d} \sum_i dX_{ij} Q_{ir}
$$

Matrix form:

$$
\boxed{dK = \frac{dX^\top Q}{\sqrt d}}
$$

---

#### Final backward algorithm 

Given $dO$:

**1. Matmul backward:**

$$
dV = P^\top dO
$$

$$
dP = dO \, V^\top
$$

**2. Softmax backward:**

$$
D = \mathrm{rowsum}(O \odot dO)
$$

$$
dX = P \odot (dP - D)
$$


**3. Score backward:**

$$
dQ = \frac{dX \, K}{\sqrt d}
$$

$$
dK = \frac{dX^\top Q}{\sqrt d}
$$


---

## Your Task

### File Structure

Create the following file:

```
flash_attention/
    flash_attention.py
```

### Implementation Requirements

Your `flash_attention/flash_attention.py` must implement:

#### 1. Custom Autograd Function (Forward Pass)

```python
class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Flash Attention forward pass using tiled online softmax.

        Args:
            ctx: Context object for saving tensors for backward
            Q: Query tensor of shape (batch, seq_len, head_dim)
            K: Key tensor of shape (batch, seq_len, head_dim)
            V: Value tensor of shape (batch, seq_len, head_dim)
            is_causal: Whether to apply causal masking (default: False)

        Returns:
            O: Output tensor of shape (batch, seq_len, head_dim)
        """
        # Your code here

        # Use tiled computation with online softmax:
        # - Outer loop over query blocks
        # - Inner loop over key/value blocks
        # - Maintain running (m, l, O) accumulators
        # - Handle causal masking when is_causal=True

        # IMPORTANT: Save tensors needed for backward
        # ctx.save_for_backward(Q, K, V, L, O)
        # ctx.is_causal = is_causal
        # ctx.sqrt_d = sqrt_d

    @staticmethod
    def backward(ctx, dO):
        """
        Attention backward pass.

        Args:
            ctx: Context object with saved tensors
            dO: Gradient of loss w.r.t. output

        Returns:
            dQ: Gradient w.r.t. Q
            dK: Gradient w.r.t. K
            dV: Gradient w.r.t. V
            None: No gradient for is_causal
        """
        Q, K, V, L, O = ctx.saved_tensors
        dQ, dK, dV, _ = attention_backward_impl(
            Q, K, V, L, O, dO, ctx.sqrt_d, ctx.is_causal
        )
        return dQ, dK, dV, None
```

#### 2. Backward Pass Implementation

```python
def attention_backward_impl(Q, K, V, L, O, dO, sqrt_d, is_causal):
    """
    Backward pass implementation for Flash Attention.

    Uses standard attention gradient formulas with recomputation of P
    from saved log-sum-exp values L.

    Args:
        Q: Query tensor of shape (batch, seq_q, d)
        K: Key tensor of shape (batch, seq_k, d)
        V: Value tensor of shape (batch, seq_k, d)
        L: Log-sum-exp values from forward pass, shape (batch, seq_q)
        O: Output from forward pass, shape (batch, seq_q, d)
        dO: Gradient of loss w.r.t. output, shape (batch, seq_q, d)
        sqrt_d: Square root of head dimension (for scaling)
        is_causal: Whether to apply causal masking

    Returns:
        dQ: Gradient w.r.t. Q
        dK: Gradient w.r.t. K
        dV: Gradient w.r.t. V
        None: Placeholder for compatibility
    """
    # Your code here
    #
    # Steps:
    # 1. Compute D = rowsum(O ⊙ dO)
    # 2. Recompute S = Q @ K^T / sqrt(d)
    # 3. Apply causal mask if is_causal
    # 4. Recompute P = exp(S - L)
    # 5. Compute dV = P^T @ dO
    # 6. Compute dP = dO @ V^T
    # 7. Compute dS = P ⊙ (dP - D)
    # 8. Compute dQ = dS @ K / sqrt(d)
    # 9. Compute dK = dS^T @ Q / sqrt(d)
```

#### 3. Causal Masking

Your implementation must support **causal attention** when `is_causal=True`. In causal attention, position $i$ can only attend to positions $j \leq i$.

**In the forward pass:**
- Skip key/value blocks that are entirely beyond the current query positions (early exit optimization)
- For blocks that partially overlap, apply a causal mask:
  ```python
  # For query positions [offset_i : end_i] and key positions [offset_j : end_j]
  mask = torch.arange(offset_i, end_i, device=device).unsqueeze(-1) >= \
         torch.arange(offset_j, end_j, device=device).unsqueeze(0)
  S = torch.where(mask, S, torch.tensor(float("-inf"), device=device, dtype=dtype))
  ```

**In the backward pass:**
- Apply the same causal mask when recomputing attention scores:
  ```python
  if is_causal:
      causal_mask = torch.triu(torch.ones(seq_q, seq_k, device=Q.device, dtype=torch.bool), diagonal=1)
      S = S.masked_fill(causal_mask, float("-inf"))
  ```

---

## Implementation Hints

### Forward Pass Tips

1. **Loop structure**: Use nested loops over query blocks (outer) and key/value blocks (inner). Take blocks of sizes $16 x 16$

2. **Handling non-divisible sequences**: If `seq_len % block_size != 0`, handle the last partial block:
   ```python
   for i in range(0, seq_len, block_size):
       block_end = min(i + block_size, seq_len)
       Q_block = Q[:, :, i:block_end, :]
       # ...
   ```

3. **Scaling factor**: Don't forget to divide by `sqrt(d)`:
   ```python
   sqrt_d = math.sqrt(head_dim)
   ```

4. **Numerical stability**: Always subtract the max before taking exp.

5. **Save for backward**: Store `L` (log-sum-exp) per row, and store `O` at the end:
   ```python
   ctx.save_for_backward(Q, K, V, L, O)
   ```

### Backward Pass Tips

1. **Recompute, don't store**: recompute the attention scores $X$ (without tiling, i.e. not like in the forward pass).

2. **D computation**: This term accounts for the normalization in softmax:
   ```python
   D = (dO * O).sum(dim=-1, keepdim=True)
   ```


---

## Testing Your Implementation

Run the tests to verify correctness:

```bash
pytest tests/test_flash_attention_pytorch.py -v
```

The tests check:
- Forward pass correctness against standard attention
- Backward pass correctness using `torch.autograd.gradcheck`
- Output shapes
- Numerical stability
- Various sequence lengths and head dimensions

### Gradient Checking

You can manually verify gradients with:

```python
from torch.autograd import gradcheck

Q = torch.randn(1, 32, 16, dtype=torch.float64, requires_grad=True)
K = torch.randn(1, 32, 16, dtype=torch.float64, requires_grad=True)
V = torch.randn(1, 32, 16, dtype=torch.float64, requires_grad=True)

assert gradcheck(
    lambda q, k, v: FlashAttentionFunction.apply(q, k, v),
    (Q, K, V),
    eps=1e-6,
    atol=1e-4,
    rtol=1e-3
)
```


**Note:** Use `float64` for gradient checking to get accurate numerical gradients.

---

## Expected Behavior

Your implementation should:

1. **Match standard attention output** within floating-point tolerance (rtol=1e-3 for float32)

2. **Pass gradient checks** for all inputs Q, K, V

3. **Handle various shapes**:
   - Different batch sizes
   - Different numbers of heads
   - Sequence lengths that are/aren't divisible by block_size
   - Different head dimensions

4. **Be memory efficient**: For large sequences, peak memory should be $O(N \cdot \text{block\_size})$ rather than $O(N^2)$


---
