import torch
import triton
import triton.language as tl
import torch.nn.functional as F


def softmax_mult(x, V, dim=-1):
    return F.softmax(x, dim=dim) @ V
