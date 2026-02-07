"""
Online softmax Triton kernel tests.

Tests the Triton implementation of online softmax algorithm against PyTorch reference.
Verifies forward pass correctness, numerical stability, and backward pass gradients.
"""
import pytest
import torch

from adapters import get_online_softmax_function, get_online_softmax_autograd_class

# Get implementations from adapters
online_softmax = get_online_softmax_function()
OnlineSoftmax = get_online_softmax_autograd_class()


ATOL = 1e-5
RTOL = 1e-5


class TestOnlineSoftmaxForward:
    """Test forward pass of Triton online softmax."""

    @pytest.mark.parametrize("shape,block_size", [
        ((1, 16, 32), 8),    # Small tensor
        ((2, 32, 64), 16),   # Medium tensor
        ((4, 64, 128), 32),  # Larger tensor
    ])
    def test_forward_correctness(self, device, shape, block_size):
        """Test that Triton softmax matches PyTorch softmax."""
        x = torch.randn(*shape, device=device, dtype=torch.float32)

        y_ref = torch.softmax(x, dim=-1)
        y_triton = online_softmax(x, BLOCK_1=block_size, BLOCK_2=block_size)

        torch.testing.assert_close(y_triton, y_ref, atol=ATOL, rtol=RTOL)

    def test_output_properties(self, device):
        """Test that softmax output sums to 1 and is non-negative."""
        x = torch.randn(4, 32, 64, device=device)
        y = online_softmax(x, BLOCK_1=16, BLOCK_2=16)

        # Check sums to 1
        sums = y.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

        # Check non-negative
        assert (y >= 0).all(), "Softmax output should be non-negative"

    def test_numerical_stability(self, device):
        """Test numerical stability with extreme input values."""
        # Large values
        x_large = torch.randn(2, 32, 64, device=device) * 100
        y_large_ref = torch.softmax(x_large, dim=-1)
        y_large_triton = online_softmax(x_large, BLOCK_1=16, BLOCK_2=16)
        torch.testing.assert_close(y_large_triton, y_large_ref, atol=1e-4, rtol=1e-4)

        # Small values
        x_small = torch.randn(2, 32, 64, device=device) * 0.001
        y_small_ref = torch.softmax(x_small, dim=-1)
        y_small_triton = online_softmax(x_small, BLOCK_1=16, BLOCK_2=16)
        torch.testing.assert_close(y_small_triton, y_small_ref, atol=ATOL, rtol=RTOL)


class TestOnlineSoftmaxBackward:
    """Test backward pass of Triton online softmax."""

    @pytest.mark.parametrize("shape,block_size", [
        ((2, 32, 64), 16),
        ((4, 64, 128), 32),
    ])
    def test_backward_correctness(self, device, shape, block_size):
        """Test that backward pass produces correct gradients."""
        x_ref = torch.randn(*shape, device=device, requires_grad=True)
        x_triton = x_ref.detach().clone().requires_grad_(True)

        # Forward pass
        y_ref = torch.softmax(x_ref, dim=-1)
        y_triton = OnlineSoftmax.apply(x_triton, block_size, block_size)

        # Backward pass
        grad_output = torch.randn_like(y_ref)
        y_ref.backward(grad_output)
        y_triton.backward(grad_output)

        torch.testing.assert_close(x_triton.grad, x_ref.grad, atol=1e-4, rtol=1e-4)
        assert x_triton.grad.shape == x_triton.shape, "Gradient shape should match input shape"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
