"""Shared utilities used across all notebooks."""

import torch


def get_device() -> str:
    """Return best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_scaler(device: str):
    """Return GradScaler for CUDA only (MPS and CPU use None)."""
    if device == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


def autocast_ctx(device: str):
    """Return autocast context for CUDA only."""
    if device == "cuda":
        return torch.autocast("cuda")
    return torch.autocast("cpu", enabled=False)
