"""
Flash Attention GPU correctness tests.

Tests both PyTorch and Triton implementations on GPU with various configurations
(batch sizes, sequence lengths, causal/non-causal). Compares against a reference
standard attention implementation.

For CPU tests of PyTorch implementation, see test_attention.py.
For memory efficiency tests, see test_flash_memory.py.
"""
import pytest
import torch
import math

from adapters import (
    get_flashattention_autograd_function_pytorch,
    get_flashattention_autograd_function_triton,
)
from conftest import skip_on_triton_error


ATOL = 1e-2
RTOL = 1e-2

# Get implementation classes from adapters
FlashAttentionPytorch = get_flashattention_autograd_function_pytorch()
FlashAttentionTriton = get_flashattention_autograd_function_triton()


def run_flash_attention_triton(Q, K, V, is_causal):
    """Wrapper that catches Triton compilation errors on unsupported hardware."""
    return skip_on_triton_error(FlashAttentionTriton.apply, Q, K, V, is_causal)


def run_flash_attention_pytorch(Q, K, V, is_causal):
    """Run PyTorch implementation."""
    return FlashAttentionPytorch.apply(Q, K, V, is_causal)


# Parametrize implementations for testing
IMPLEMENTATIONS = [
    pytest.param(run_flash_attention_triton, id="triton"),
    pytest.param(run_flash_attention_pytorch, id="pytorch"),
]


def reference_attention(Q, K, V, is_causal=False):
    """Standard attention: softmax(Q @ K^T / sqrt(d)) @ V"""
    d = Q.shape[-1]
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d)

    if is_causal:
        seq_q, seq_k = Q.shape[1], K.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_q, seq_k, device=Q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return torch.bmm(attn, V)


class TestFlashAttentionForward:
    """Test forward pass of Flash Attention."""

    @pytest.mark.parametrize("impl", IMPLEMENTATIONS)
    @pytest.mark.parametrize(
        "batch,seq,d",
        [
            (1, 128, 128),  # Small
            (2, 256, 128),  # Medium
            (4, 512, 128),  # Larger
        ],
    )
    def test_forward_correctness(self, device, impl, batch, seq, d):
        """Test that Flash Attention matches reference implementation."""
        Q = torch.randn(batch, seq, d, device=device, dtype=torch.float32)
        K = torch.randn(batch, seq, d, device=device, dtype=torch.float32)
        V = torch.randn(batch, seq, d, device=device, dtype=torch.float32)

        y_ref = reference_attention(Q, K, V, is_causal=False)
        y_flash = impl(Q, K, V, False)

        torch.testing.assert_close(y_flash, y_ref, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("impl", IMPLEMENTATIONS)
    @pytest.mark.parametrize(
        "batch,seq,d",
        [
            (1, 128, 128),
            (2, 256, 128),
        ],
    )
    def test_causal_attention(self, device, impl, batch, seq, d):
        """Test causal masking in Flash Attention."""
        Q = torch.randn(batch, seq, d, device=device, dtype=torch.float32)
        K = torch.randn(batch, seq, d, device=device, dtype=torch.float32)
        V = torch.randn(batch, seq, d, device=device, dtype=torch.float32)

        y_ref = reference_attention(Q, K, V, is_causal=True)
        y_flash = impl(Q, K, V, True)

        torch.testing.assert_close(y_flash, y_ref, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("impl", IMPLEMENTATIONS)
    def test_output_shape(self, device, impl):
        """Test that output has correct shape."""
        batch, seq, d = 2, 256, 128
        Q = torch.randn(batch, seq, d, device=device)
        K = torch.randn(batch, seq, d, device=device)
        V = torch.randn(batch, seq, d, device=device)

        O = impl(Q, K, V, False)

        assert O.shape == Q.shape, f"Expected shape {Q.shape}, got {O.shape}"


class TestFlashAttentionBackward:
    """Test backward pass of Flash Attention."""

    @pytest.mark.parametrize("impl", IMPLEMENTATIONS)
    @pytest.mark.parametrize(
        "batch,seq,d",
        [
            (2, 128, 128),
            (2, 256, 128),
        ],
    )
    def test_backward_correctness(self, device, impl, batch, seq, d):
        """Test that backward pass produces correct gradients."""
        Q_ref = torch.randn(batch, seq, d, device=device, requires_grad=True)
        K_ref = torch.randn(batch, seq, d, device=device, requires_grad=True)
        V_ref = torch.randn(batch, seq, d, device=device, requires_grad=True)

        Q_flash = Q_ref.detach().clone().requires_grad_(True)
        K_flash = K_ref.detach().clone().requires_grad_(True)
        V_flash = V_ref.detach().clone().requires_grad_(True)

        # Forward
        y_ref = reference_attention(Q_ref, K_ref, V_ref, is_causal=False)
        y_flash = impl(Q_flash, K_flash, V_flash, False)

        # Backward
        grad_output = torch.randn_like(y_ref)
        y_ref.backward(grad_output)
        y_flash.backward(grad_output)

        torch.testing.assert_close(Q_flash.grad, Q_ref.grad, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(K_flash.grad, K_ref.grad, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(V_flash.grad, V_ref.grad, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("impl", IMPLEMENTATIONS)
    def test_backward_causal(self, device, impl):
        """Test backward pass with causal masking."""
        batch, seq, d = 2, 128, 128

        Q_ref = torch.randn(batch, seq, d, device=device, requires_grad=True)
        K_ref = torch.randn(batch, seq, d, device=device, requires_grad=True)
        V_ref = torch.randn(batch, seq, d, device=device, requires_grad=True)

        Q_flash = Q_ref.detach().clone().requires_grad_(True)
        K_flash = K_ref.detach().clone().requires_grad_(True)
        V_flash = V_ref.detach().clone().requires_grad_(True)

        # Forward with causal
        y_ref = reference_attention(Q_ref, K_ref, V_ref, is_causal=True)
        y_flash = impl(Q_flash, K_flash, V_flash, True)

        # Backward
        grad_output = torch.randn_like(y_ref)
        y_ref.backward(grad_output)
        y_flash.backward(grad_output)

        torch.testing.assert_close(Q_flash.grad, Q_ref.grad, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(K_flash.grad, K_ref.grad, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(V_flash.grad, V_ref.grad, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
