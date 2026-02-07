"""
Test adapters for your implementations.

You should modify this file if your function/class names differ from the defaults.
All test files import from this module, so changes here affect all tests.
"""

from __future__ import annotations

from typing import Callable, Type


# =============================================================================
# Online Softmax Adapters
# =============================================================================


def get_online_softmax_function() -> Callable:
    """
    Returns the online softmax function implemented in Triton.

    Returns:
        A callable that computes softmax using the online algorithm.
    """
    from online_softmax.online_softmax import online_softmax

    return online_softmax


def get_online_softmax_autograd_class() -> Type:
    """
    Returns the autograd class for online softmax.

    Returns:
        A torch.autograd.Function subclass for online softmax.
    """
    from online_softmax.online_softmax import OnlineSoftmax

    return OnlineSoftmax


# =============================================================================
# Fused Softmax-Matmul Adapters
# =============================================================================


def get_fused_softmax_function() -> Callable:
    """
    Returns the fused softmax-matmul function implemented in Triton.
    Computes: softmax(x) @ V without materializing the full softmax output.

    Returns:
        A callable that computes fused softmax-matmul.
    """
    from softmax_matmul.softmax_matmul import fused_softmax

    return fused_softmax


def get_softmax_matmul_reference() -> Callable:
    """
    Returns the reference implementation of softmax-matmul.
    Computes: softmax(x) @ V using standard PyTorch operations.

    Returns:
        A callable that computes the reference softmax-matmul.
    """
    from softmax_matmul.softmax_matmul import softmax_mult

    return softmax_mult


# =============================================================================
# Flash Attention Adapters
# =============================================================================


def get_flashattention_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    from flash_attention.flash_attention import FlashAttentionPytorch

    return FlashAttentionPytorch


def get_flashattention_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.

    Returns:
        A class object (not an instance of the class)
    """
    from flash_attention.flash_attention import FlashAttentionTriton

    return FlashAttentionTriton
