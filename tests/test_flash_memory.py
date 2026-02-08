"""
Flash Attention memory efficiency tests.

These tests verify that the FlashAttention implementation:
1. Does NOT save the full N×N attention matrix (O(N²) memory)
2. Saves the expected tensors for backward pass (Q, K, V, O, L)
3. Forward pass memory scales linearly with sequence length

These tests help detect if students wrap F.scaled_dot_product_attention instead
of implementing the true Flash Attention algorithm, since SDPA doesn't expose
the same autograd structure (no saved_tensors with specific shapes).
"""

import pytest
import torch

from adapters import get_flashattention_autograd_function_pytorch

# Get the .apply callable from the autograd class
FlashAttentionPytorch = get_flashattention_autograd_function_pytorch()


def _get_flash_attention_apply():
    """Get the .apply method for the Flash Attention autograd function."""
    return FlashAttentionPytorch.apply


def test_flash_attention_no_quadratic_saved_tensors():
    """Verify implementation doesn't save the full N×N attention matrix."""
    impl = _get_flash_attention_apply()

    batch_size = 2
    seq_len = 512
    d_model = 64

    q = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    o = impl(q, k, v, False)

    # Check saved tensors - none should have shape (batch, seq, seq)
    # which would indicate materializing the attention matrix
    for tensor in o.grad_fn.saved_tensors:
        assert tensor.shape != (batch_size, seq_len, seq_len), (
            f"Found tensor with shape {tensor.shape} in saved tensors. "
            "Flash Attention should NOT save the full attention matrix (N×N). "
            "This suggests you're using standard attention, not the tiled/online algorithm."
        )

    # Check no saved tensor is larger than O(N*d)
    max_allowed_elements = batch_size * seq_len * d_model * 2  # generous bound
    for tensor in o.grad_fn.saved_tensors:
        assert tensor.numel() <= max_allowed_elements, (
            f"Saved tensor has {tensor.numel()} elements, exceeding O(N*d) bound. "
            "Flash Attention should only save Q, K, V, O, and L (log-sum-exp)."
        )


def test_flash_attention_expected_saved_tensors():
    """Verify the implementation saves the expected tensors for backward pass."""
    impl = _get_flash_attention_apply()

    batch_size = 2
    seq_len = 256
    d_model = 64

    q = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    o = impl(q, k, v, False)

    saved_shapes = [t.shape for t in o.grad_fn.saved_tensors]

    # Flash Attention should save: Q, K, V, O (all batch x seq x d) and L (batch x seq)
    qkvo_shape = (batch_size, seq_len, d_model)
    l_shape = (batch_size, seq_len)

    qkvo_count = sum(1 for s in saved_shapes if s == qkvo_shape)
    l_count = sum(1 for s in saved_shapes if s == l_shape)

    assert qkvo_count >= 4, (
        f"Expected at least 4 tensors of shape {qkvo_shape} (Q, K, V, O), "
        f"but found {qkvo_count}. Saved shapes: {saved_shapes}"
    )

    assert l_count >= 1, (
        f"Expected at least 1 tensor of shape {l_shape} (log-sum-exp L), "
        f"but found {l_count}. Saved shapes: {saved_shapes}"
    )


def test_flash_attention_forward_memory(device):
    """Verify Flash Attention forward pass doesn't materialize the full attention matrix.

    Standard attention for seq_len=4096 with batch=8 would need:
    8 * 4096 * 4096 * 4 bytes ≈ 512 MB just for the attention matrix.
    Flash Attention forward should use much less memory.

    Note: The backward pass may still use quadratic memory if not optimized with Triton.
    This test only checks the forward pass.
    """
    impl = _get_flash_attention_apply()

    batch_size = 8
    seq_len = 4096
    d_model = 64

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    # Only test forward pass - backward may use quadratic memory
    with torch.no_grad():
        o = impl(q, k, v, False)

    peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6

    # Input tensors: 3 * 8 * 4096 * 64 * 4 bytes ≈ 24 MB
    # Output + L: 8 * 4096 * 64 * 4 + 8 * 4096 * 4 ≈ 8.5 MB
    # Standard attention matrix would be: 8 * 4096 * 4096 * 4 ≈ 512 MB
    # Flash Attention forward should stay well below that

    input_memory_mb = 3 * batch_size * seq_len * d_model * 4 / 1e6
    output_memory_mb = batch_size * seq_len * (d_model + 1) * 4 / 1e6  # O and L
    quadratic_memory_mb = batch_size * seq_len * seq_len * 4 / 1e6

    # Allow for inputs + outputs + some overhead, but not the full attention matrix
    max_expected = input_memory_mb + output_memory_mb + quadratic_memory_mb * 0.3

    assert peak_memory_mb < max_expected, (
        f"Forward pass peak memory {peak_memory_mb:.1f}MB is too high. "
        f"Standard attention would need ~{quadratic_memory_mb:.1f}MB for attention matrix alone. "
        f"Expected less than {max_expected:.1f}MB. "
        "True Flash Attention forward should not materialize the full N×N matrix."
    )
