import torch
import math
import triton
import triton.language as tl
import random


@triton.jit
def online_softmax_kernel(
    x_ptr,
    softmax_ptr,
    stride_xbatch,
    stride_xrow,
    stride_xcol,
    stride_sbatch,
    stride_srow,
    stride_scol,
    d1: tl.constexpr,
    d2: tl.constexpr,
    BLOCK_1: tl.constexpr,
    BLOCK_2: tl.constexpr,
):

    tl.static_assert(d2 % BLOCK_2 == 0, "d2 must be divisible by BLOCK_2")
    tl.static_assert(d1 % BLOCK_1 == 0, "d1 must be divisible by BLOCK_1")

    # Each program handles one block of rows (BLOCK_1 rows)
    pid_batch = tl.program_id(0)
    pid_row = tl.program_id(1)

    x_block = tl.make_block_ptr(
        x_ptr + pid_batch * stride_xbatch,
        shape=(d1, d2),
        strides=(stride_xrow, stride_xcol),
        offsets=(pid_row * BLOCK_1, 0),
        block_shape=(BLOCK_1, BLOCK_2),
        order=(1, 0),
    )

    # Number of blocks in the column dimension
    Num_blocks = tl.cdiv(d2, BLOCK_2)

    # Initialize m_prev and l_prev for this block of rows
    m_prev = tl.full((BLOCK_1,), float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros((BLOCK_1,), dtype=tl.float32)

    # First pass: compute global max and sum
    for _ in range(Num_blocks):
        x = tl.load(x_block, boundary_check=(0, 1), padding_option="zero")

        # Compute block max
        block_max = tl.max(x, axis=1)
        m_curr = tl.maximum(m_prev, block_max)

        # Update running sum with rescaling
        exp_x_block = tl.exp(x - m_curr[:, None])
        l_prev = l_prev * tl.exp(m_prev - m_curr) + tl.sum(exp_x_block, axis=1)
        m_prev = m_curr

        x_block = x_block.advance((0, BLOCK_2))

    # Second pass: compute and store softmax output
    softmax_block = tl.make_block_ptr(
        softmax_ptr + pid_batch * stride_sbatch,
        shape=(d1, d2),
        strides=(stride_srow, stride_scol),
        offsets=(pid_row * BLOCK_1, 0),
        block_shape=(BLOCK_1, BLOCK_2),
        order=(1, 0),
    )
    x_block = tl.make_block_ptr(
        x_ptr + pid_batch * stride_xbatch,
        shape=(d1, d2),
        strides=(stride_xrow, stride_xcol),
        offsets=(pid_row * BLOCK_1, 0),
        block_shape=(BLOCK_1, BLOCK_2),
        order=(1, 0),
    )
    # tl.device_print("m_prev:", m_prev)
    # tl.device_print("l_prev:", l_prev)

    for _ in range(Num_blocks):
        x = tl.load(x_block, boundary_check=(0, 1), padding_option="zero")
        # Compute softmax for this block
        tl.store(
            softmax_block,
            tl.exp(x - m_prev[:, None]) / l_prev[:, None],
            boundary_check=(0, 1),
        )

        x_block = x_block.advance((0, BLOCK_2))
        softmax_block = softmax_block.advance((0, BLOCK_2))


def online_softmax(x, BLOCK_1=16, BLOCK_2=16):
    """
    Compute softmax using Triton kernel with online algorithm.

    Args:
        x: Input tensor of shape (batch_size, d1, d2)
        BLOCK_1: Block size for dimension d1 (rows)
        BLOCK_2: Block size for dimension d2 (columns, softmax dimension)
    """
    batch_size, d1, d2 = x.shape
    softmax_output = torch.empty_like(x)

    # Calculate grid dimensions
    grid = (batch_size, triton.cdiv(d1, BLOCK_1))

    # Launch kernel
    online_softmax_kernel[grid](
        x,
        softmax_output,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        softmax_output.stride(0),
        softmax_output.stride(1),
        softmax_output.stride(2),
        d1,
        d2,
        BLOCK_1=BLOCK_1,
        BLOCK_2=BLOCK_2,
    )

    return softmax_output


def softmax_backward(grad_output, output):
    sum_grad_output = torch.sum(grad_output * output, dim=-1, keepdim=True)
    grad_input = output * (grad_output - sum_grad_output)
    return grad_input


class OnlineSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, BLOCK_1=16, BLOCK_2=16):
        y = online_softmax(x, BLOCK_1=BLOCK_1, BLOCK_2=BLOCK_2)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        return softmax_backward(grad_output, y), None, None
