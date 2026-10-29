"""Teacher (ResNet-50 / MobileNetV3) and Student (ResNet-18) model registry."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as tv_models


MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "resnet50": {
        "display_name": "ResNet-50 (FP32 Baseline - 25.6M Params, 98.0 MB)",
        "domain": "Computer Vision (Image Classification)",
        "student_pair": "ResNet-18 Distilled Student (11.7M Params, 44.0 MB)",
        "student_arch": "resnet18",
        "reference_size_mb": 98.0,
        "student_size_mb": 44.0,
    },
    "mobilenet_v3_large": {
        "display_name": "MobileNetV3-Large (FP32 Baseline - 5.4M Params, 21.1 MB)",
        "domain": "Computer Vision (Image Classification)",
        "student_pair": "MobileNetV3-Small Distilled Student (2.5M Params, 9.8 MB)",
        "student_arch": "mobilenet_v3_small",
        "reference_size_mb": 21.1,
        "student_size_mb": 9.8,
    },
    "bert_base_uncased": {
        "display_name": "BERT-Base-Uncased (FP32 Baseline - 109.5M Params, 418.0 MB)",
        "domain": "Natural Language Processing (Sequence Classification)",
        "student_pair": "DistilBERT-Base Student (66.4M Params, 255.0 MB)",
        "student_arch": "distilbert_base",
        "reference_size_mb": 98.0,
        "student_size_mb": 44.0,
    },
    "roberta_base": {
        "display_name": "RoBERTa-Base (FP32 Baseline - 124.6M Params, 475.0 MB)",
        "domain": "Natural Language Processing (Sequence Classification)",
        "student_pair": "DistilRoBERTa-Base Student (82.1M Params, 313.0 MB)",
        "student_arch": "distilroberta_base",
        "reference_size_mb": 98.0,
        "student_size_mb": 44.0,
    },
}


def load_teacher_model(
    arch: str = "resnet50",
    num_classes: int = 10,
    seed: int = 42,
) -> nn.Module:
    """Deterministically instantiate the FP32 teacher model in eval mode."""
    torch.manual_seed(int(seed))
    clean_arch = (arch or "resnet50").strip().lower()

    if clean_arch in ("mobilenet_v3_large", "mobilenet"):
        model = tv_models.mobilenet_v3_large(weights=None, num_classes=int(num_classes))
    else:
        # Default to ResNet-50 teacher architecture
        model = tv_models.resnet50(weights=None, num_classes=int(num_classes))

    model.eval()
    return model


def load_student_model(
    num_classes: int = 10,
    seed: int = 42,
    arch: str = "resnet18",
) -> nn.Module:
    """Deterministically instantiate the compact Knowledge Distillation student model."""
    torch.manual_seed(int(seed))
    clean_arch = (arch or "resnet18").strip().lower()

    if clean_arch in ("mobilenet_v3_small",):
        model = tv_models.mobilenet_v3_small(weights=None, num_classes=int(num_classes))
    else:
        model = tv_models.resnet18(weights=None, num_classes=int(num_classes))

    model.eval()
    return model


def count_parameters(model: nn.Module) -> dict[str, Any]:
    """Count total, trainable, and non-zero parameters along with overall weight sparsity."""
    total_params = 0
    trainable_params = 0
    nonzero_params = 0

    with torch.no_grad():
        for param in model.parameters():
            numel = int(param.numel())
            total_params += numel
            if param.requires_grad:
                trainable_params += numel
            nonzero_params += int(torch.count_nonzero(param).item())

    zero_params = max(0, total_params - nonzero_params)
    sparsity_pct = round((zero_params / total_params) * 100.0, 2) if total_params > 0 else 0.0

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "nonzero_params": nonzero_params,
        "zero_params": zero_params,
        "sparsity_pct": sparsity_pct,
    }


def get_model_size_mb(model_or_path: nn.Module | str | Path) -> float:
    """Compute exact serialized size in MB for either a file path or a PyTorch nn.Module."""
    if isinstance(model_or_path, (str, Path)):
        path_obj = Path(model_or_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file does not exist: {path_obj}")
        size_bytes = path_obj.stat().st_size
        return round(size_bytes / (1024 * 1024), 2)

    if isinstance(model_or_path, nn.Module):
        buffer = io.BytesIO()
        torch.save(model_or_path.state_dict(), buffer)
        size_bytes = buffer.tell()
        return round(size_bytes / (1024 * 1024), 2)

    raise TypeError("Expected a torch.nn.Module or a file path (str / Path).")
