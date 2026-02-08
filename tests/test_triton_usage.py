"""
Tests to verify that implementations actually use Triton kernels.
"""

import inspect
import pytest

from adapters import (
    get_flashattention_autograd_function_pytorch,
    get_flashattention_autograd_function_triton,
)

# Get implementation classes from adapters
FlashAttentionPytorch = get_flashattention_autograd_function_pytorch()
FlashAttentionTriton = get_flashattention_autograd_function_triton()


def test_triton_class_uses_triton_kernels():
    """Verify that the Triton implementation actually uses Triton, not just PyTorch."""
    # Get the module where FlashAttentionTriton is defined
    module = inspect.getmodule(FlashAttentionTriton)
    module_source = inspect.getsource(module)

    # Check for Triton kernel indicators in the module
    triton_indicators = [
        "@triton.jit",
        "triton.cdiv",
        "tl.program_id",
    ]

    found_indicators = [ind for ind in triton_indicators if ind in module_source]
    assert len(found_indicators) >= 2, (
        f"FlashAttentionTriton module does not appear to use Triton kernels. "
        f"Expected to find Triton indicators like @triton.jit, triton.cdiv, tl.program_id. "
        f"Found: {found_indicators}"
    )


def test_triton_forward_calls_kernel():
    """Verify that the Triton forward pass calls a Triton kernel."""
    # Get source of the forward method
    forward_source = inspect.getsource(FlashAttentionTriton.forward)

    # The forward method should contain a kernel launch (grid pattern)
    kernel_launch_indicators = [
        "[grid]",  # kernel[grid](...) launch syntax
        "_kernel[",  # alternative naming
        "fwd_kernel[",
    ]

    has_kernel_launch = any(ind in forward_source for ind in kernel_launch_indicators)
    assert has_kernel_launch, (
        "FlashAttentionTriton.forward does not appear to launch a Triton kernel. "
        "Expected to find kernel launch syntax like 'kernel[grid](...)'"
    )


def test_triton_class_different_from_pytorch():
    """Verify that FlashAttentionTriton is not just a copy of FlashAttentionPytorch."""
    triton_forward = inspect.getsource(FlashAttentionTriton.forward)
    pytorch_forward = inspect.getsource(FlashAttentionPytorch.forward)

    # The implementations should be substantially different
    # (Triton version should be shorter due to kernel call vs explicit loops)
    assert (
        triton_forward != pytorch_forward
    ), "FlashAttentionTriton.forward is identical to FlashAttentionPytorch.forward"

    # Additional check: Triton version should contain kernel launch, PyTorch should not
    assert (
        "[grid]" in triton_forward or "_kernel[" in triton_forward
    ), "FlashAttentionTriton.forward should contain a Triton kernel launch"
    assert (
        "[grid]" not in pytorch_forward
    ), "FlashAttentionPytorch.forward should not contain a Triton kernel launch"
