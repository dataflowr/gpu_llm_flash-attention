import pytest
import torch
import warnings


def pytest_configure(config):
    """Initialize CUDA and cuBLAS before any tests run."""
    if torch.cuda.is_available():
        # Force CUDA context and cuBLAS initialization
        _ = torch.empty(1, device="cuda") @ torch.empty(1, 1, device="cuda")
        torch.cuda.synchronize()

    # Filter the cuBLAS warning
    warnings.filterwarnings(
        "ignore",
        message="Attempting to run cuBLAS, but there was no current CUDA context!",
        category=UserWarning,
    )


@pytest.fixture(scope="module")
def device():
    """Shared CUDA device fixture for all tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda"


def skip_on_triton_error(func, *args, **kwargs):
    """Wrapper that skips tests on Triton compilation errors (T4/Turing GPUs)."""
    try:
        return func(*args, **kwargs)
    except IndexError as e:
        if "map::at" in str(e):
            pytest.skip("Triton compilation not supported on this GPU architecture (T4/Turing)")
        raise
