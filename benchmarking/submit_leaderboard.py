import os
import torch
import triton
import torch.nn.functional as F

from flash_attention.flash_attention import FlashAttentionTriton


def get_gpu_name():
    """Get GPU name if available."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "No GPU"


def benchmark_flash_forward_backward(compile=True):
    """Benchmark Flash Attention forward + backward pass."""
    n_heads = 16
    d_head = 64
    sequence_length = 8192

    q = torch.randn(
        n_heads,
        sequence_length,
        d_head,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn(
        n_heads,
        sequence_length,
        d_head,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    v = torch.randn(
        n_heads,
        sequence_length,
        d_head,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    # flash_apply = lambda q, k, v, is_causal: F.#scaled_dot_product_attention(
    #    q, k, v, is_causal=is_causal
    # )
    flash_apply = FlashAttentionTriton.apply
    flash_apply = torch.compile(flash_apply, fullgraph=False)

    def flash_forward_backward():
        o = flash_apply(q, k, v, True)
        loss = o.sum()
        loss.backward()
        # Clear gradients to avoid accumulation
        q.grad = k.grad = v.grad = None

    # triton.testing.do_bench returns time in ms
    time_ms = triton.testing.do_bench(flash_forward_backward, rep=10000, warmup=1000)
    return time_ms


if __name__ == "__main__":
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}")
    print(f"{'='*60}")

    # Benchmark with compile
    print("Benchmarking Flash Attention")
    time_compiled = benchmark_flash_forward_backward(compile=False)
    print(f"  Forward + Backward: {time_compiled:.3f} ms")

    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(script_dir, "..", "outputs", "leaderboard_result.txt")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        f.write(f"GPU: {gpu_name}\n")
        f.write(f"Forward + Backward: {time_compiled:.3f} ms\n")
    print(f"\nResults saved to {out_file}")
