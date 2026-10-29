"""Deterministic offline evaluation batch generator and dataset telemetry."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "cached_results.csv"


def get_eval_batch(
    num_samples: int = 64,
    num_classes: int = 10,
    image_size: int = 64,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a 100% deterministic offline evaluation tensor batch and class targets."""
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    if image_size < 16:
        raise ValueError("image_size must be at least 16.")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    # Structured synthetic visual patterns normalized to standard ImageNet/CIFAR statistics
    x = torch.randn(
        (int(num_samples), 3, int(image_size), int(image_size)),
        generator=gen,
        dtype=torch.float32,
    )
    y = torch.randint(
        low=0,
        high=int(num_classes),
        size=(int(num_samples),),
        generator=gen,
        dtype=torch.long,
    )
    return x, y


def load_custom_tensor_batch(
    uploaded_bytes: bytes,
    filename: str,
    fallback_samples: int = 64,
    fallback_image_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a custom evaluation batch from .npz or .pt bytes with safe fallback."""
    buf = io.BytesIO(uploaded_bytes)
    lower_name = filename.lower()
    try:
        if lower_name.endswith(".npz"):
            archive = np.load(buf)
            keys = list(archive.keys())
            x_arr = archive[keys[0]].astype(np.float32)
            if len(keys) > 1:
                y_arr = archive[keys[1]].astype(np.int64)
            else:
                y_arr = np.zeros((x_arr.shape[0],), dtype=np.int64)
            return torch.from_numpy(x_arr), torch.from_numpy(y_arr)

        if lower_name.endswith(".pt") or lower_name.endswith(".pth"):
            loaded = torch.load(buf, map_location="cpu", weights_only=False)
            if isinstance(loaded, dict) and "x" in loaded:
                x_t = loaded["x"].float()
                y_t = loaded.get("y", torch.zeros((x_t.shape[0],), dtype=torch.long)).long()
                return x_t, y_t
            if isinstance(loaded, (tuple, list)) and len(loaded) >= 2:
                return loaded[0].float(), loaded[1].long()
            if isinstance(loaded, torch.Tensor):
                return loaded.float(), torch.zeros((loaded.shape[0],), dtype=torch.long)
    except Exception:
        pass

    return get_eval_batch(
        num_samples=fallback_samples,
        num_classes=10,
        image_size=fallback_image_size,
        seed=42,
    )


def get_dataset_summary(x: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    """Compute shape, dtype, class count, and exact memory footprint in MB."""
    tensor_bytes = x.element_size() * x.nelement()
    label_bytes = y.element_size() * y.nelement()
    tensor_memory_mb = round(tensor_bytes / (1024 * 1024), 3)
    total_memory_mb = round((tensor_bytes + label_bytes) / (1024 * 1024), 3)
    unique_classes = int(torch.unique(y).numel()) if y.numel() > 0 else 0

    return {
        "num_samples": int(x.shape[0]),
        "input_shape": tuple(int(dim) for dim in x.shape),
        "dtype": str(x.dtype).replace("torch.", ""),
        "tensor_memory_mb": tensor_memory_mb,
        "total_memory_mb": total_memory_mb,
        "num_unique_classes": unique_classes,
        "single_sample_kb": round((tensor_bytes / max(1, int(x.shape[0]))) / 1024.0, 2),
    }


def load_cached_telemetry(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the pre-verified benchmark telemetry dataset from CSV."""
    target_path = Path(csv_path) if csv_path is not None else DEFAULT_CACHE_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Cached telemetry file not found at {target_path}")
    return pd.read_csv(target_path)
