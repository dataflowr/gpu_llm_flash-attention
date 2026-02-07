"""
Fused softmax-matmul Triton kernel tests.

Tests the fused softmax(x) @ V operation implemented in Triton.
This is a key building block for Flash Attention - fusing softmax with the
value projection avoids materializing the full attention matrix.
"""
import pytest
import torch

from adapters import get_fused_softmax_function, get_softmax_matmul_reference

# Get implementations from adapters
fused_softmax = get_fused_softmax_function()
softmax_mult = get_softmax_matmul_reference()


ATOL = 1e-3
RTOL = 1e-3


class TestFusedSoftmaxForward:
    """Test fused softmax matmul produces correct results."""

    @pytest.mark.parametrize(
        "shape_x,shape_v,block_size",
        [
            ((1, 16, 32), (1, 32, 16), 16),  # Small tensors
            ((2, 32, 64), (2, 64, 32), 16),  # Medium tensors
            ((4, 64, 128), (4, 128, 64), 32),  # Larger tensors
        ],
    )
    def test_correctness(self, device, shape_x, shape_v, block_size):
        """Test that fused softmax matmul matches reference implementation."""
        x = torch.randn(*shape_x, device=device, dtype=torch.float32)
        V = torch.randn(*shape_v, device=device, dtype=torch.float32)

        y_ref = softmax_mult(x, V)
        y_fused = fused_softmax(x, V, BLOCK_1=block_size, BLOCK_2=block_size)

        torch.testing.assert_close(y_fused, y_ref, atol=ATOL, rtol=RTOL)

    def test_output_shape(self, device):
        """Test that output has correct shape (batch, d1, d3)."""
        batch, d1, d2, d3 = 2, 64, 128, 32
        x = torch.randn(batch, d1, d2, device=device)
        V = torch.randn(batch, d2, d3, device=device)

        y = fused_softmax(x, V, BLOCK_1=16, BLOCK_2=16)

        assert y.shape == (
            batch,
            d1,
            d3,
        ), f"Expected shape {(batch, d1, d3)}, got {y.shape}"

    def test_numerical_stability(self, device):
        """Test numerical stability with extreme input values."""
        x_large = torch.randn(2, 32, 64, device=device) * 100
        V = torch.randn(2, 64, 32, device=device)

        y_ref = softmax_mult(x_large, V)
        y_fused = fused_softmax(x_large, V, BLOCK_1=16, BLOCK_2=16)

        torch.testing.assert_close(y_fused, y_ref, atol=ATOL, rtol=RTOL)

    def test_different_d3_sizes(self, device):
        """Test with various output dimension sizes."""
        batch, d1, d2 = 2, 32, 64
        for d3 in [16, 32, 64, 128]:
            x = torch.randn(batch, d1, d2, device=device)
            V = torch.randn(batch, d2, d3, device=device)

            y_ref = softmax_mult(x, V)
            y_fused = fused_softmax(x, V, BLOCK_1=16, BLOCK_2=16)

            torch.testing.assert_close(y_fused, y_ref, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
