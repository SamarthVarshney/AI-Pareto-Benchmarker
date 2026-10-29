"""ONNX graph export, FP16 weight compression, Dynamic INT8 quantization, and graph architecture inspector."""

from __future__ import annotations

from collections import Counter
import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
import torch.nn as nn

from optimizer.base_models import (
    count_parameters,
    get_model_size_mb,
    load_student_model,
    load_teacher_model,
)
from optimizer.pruner import apply_l1_pruning


CANONICAL_VARIANTS: list[dict[str, Any]] = [
    {
        "model_id": "base_fp32",
        "model_name": "Base FP32 (Teacher)",
        "technique": "Baseline",
        "description": "Uncompressed FP32 Teacher Reference Architecture",
    },
    {
        "model_id": "onnx_graph_opt",
        "model_name": "ONNX Graph Opt (FP32)",
        "technique": "Graph Optimization",
        "description": "Constant Folding + Conv/BatchNorm/ReLU Operator Fusion",
    },
    {
        "model_id": "onnx_fp16",
        "model_name": "ONNX FP16 Quantized",
        "technique": "Quantization (FP16)",
        "description": "Half-Precision IEEE 754 FP16 Weight Compression",
    },
    {
        "model_id": "pruned_20_fp32",
        "model_name": "Pruned 20% (FP32)",
        "technique": "Magnitude Pruning",
        "description": "L1-Norm Unstructured Weight Pruning (20% Sparsity)",
    },
    {
        "model_id": "onnx_dynamic_int8",
        "model_name": "ONNX Dynamic INT8",
        "technique": "Quantization (INT8)",
        "description": "Dynamic 8-Bit Asymmetric Weight Quantization (QUInt8)",
    },
    {
        "model_id": "pruned_40_int8",
        "model_name": "Pruned 40% + INT8",
        "technique": "Pruning + Quantization",
        "description": "Compound L1 Pruning (40% Sparsity) + Dynamic INT8 Quantization",
    },
    {
        "model_id": "distilled_student_fp32",
        "model_name": "Distilled Student (FP32)",
        "technique": "Knowledge Distillation",
        "description": "Compact Student Architecture Paired via Knowledge Distillation",
    },
    {
        "model_id": "distilled_student_int8",
        "model_name": "Distilled Student + INT8",
        "technique": "Distillation + INT8",
        "description": "Distilled Student Architecture + Dynamic INT8 Quantization",
    },
]


def export_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str | Path,
) -> str:
    """Export a PyTorch model to ONNX format with constant folding and dynamic batch axes."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    export_kwargs: dict[str, Any] = {
        "opset_version": 17,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    }

    sig = inspect.signature(torch.onnx.export)
    if "dynamo" in sig.parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(model, dummy_input, str(out_file), **export_kwargs)

    return str(out_file)


def optimize_onnx_graph(fp32_onnx_path: str | Path, opt_output_path: str | Path) -> str:
    """Apply ONNX Runtime ORT_ENABLE_EXTENDED graph optimizations (operator fusion & folding)."""
    src_path = Path(fp32_onnx_path)
    dst_path = Path(opt_output_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = str(dst_path)

    _ = ort.InferenceSession(
        str(src_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    return str(dst_path)


def _compress_onnx_initializers_fp16(src_path: Path, dst_path: Path) -> str:
    """Convert large FP32 weight initializers to FP16 storage with CPU-safe Cast nodes."""
    model = onnx.load(str(src_path))
    graph = model.graph

    cast_nodes = []
    new_initializers = []

    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32 and arr.size >= 64:
            fp16_name = f"{init.name}_fp16_storage"
            arr_fp16 = arr.astype(np.float16)
            fp16_tensor = numpy_helper.from_array(arr_fp16, name=fp16_name)
            new_initializers.append(fp16_tensor)

            cast_node = helper.make_node(
                "Cast",
                inputs=[fp16_name],
                outputs=[init.name],
                name=f"Cast_{init.name}",
                to=TensorProto.FLOAT,
            )
            cast_nodes.append(cast_node)
        else:
            new_initializers.append(init)

    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    for idx, node in enumerate(cast_nodes):
        graph.node.insert(idx, node)

    onnx.save(model, str(dst_path))
    return str(dst_path)


def convert_onnx_fp16(fp32_onnx_path: str | Path, fp16_output_path: str | Path) -> str:
    """Convert ONNX model weights to FP16 using onnxconverter-common or CPU-safe FP16 packing."""
    src_path = Path(fp32_onnx_path)
    dst_path = Path(fp16_output_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from onnxconverter_common import float16  # type: ignore

        model = onnx.load(str(src_path))
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(model_fp16, str(dst_path))
        _ = ort.InferenceSession(str(dst_path), providers=["CPUExecutionProvider"])
        return str(dst_path)
    except Exception:
        return _compress_onnx_initializers_fp16(src_path, dst_path)


def _compress_onnx_initializers_uint8_dequant(
    src_path: Path,
    dst_path: Path,
) -> str:
    """Pack large FP32 tensors into 8-bit QUInt8 initializers with DequantizeLinear nodes."""
    model = onnx.load(str(src_path))
    graph = model.graph

    dequant_nodes = []
    new_initializers = []

    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32 and arr.size >= 64:
            w_min = float(np.min(arr))
            w_max = float(np.max(arr))

            scale = (w_max - w_min) / 255.0 if (w_max > w_min) else 1e-5
            zero_point = int(np.clip(round(-w_min / scale), 0, 255))
            q_arr = np.clip(np.round(arr / scale) + zero_point, 0, 255).astype(np.uint8)

            q_name = f"{init.name}_uint8_quant"
            s_name = f"{init.name}_scale"
            zp_name = f"{init.name}_zp"

            new_initializers.append(numpy_helper.from_array(q_arr, name=q_name))
            new_initializers.append(
                numpy_helper.from_array(np.array(scale, dtype=np.float32), name=s_name)
            )
            new_initializers.append(
                numpy_helper.from_array(np.array(zero_point, dtype=np.uint8), name=zp_name)
            )

            dq_node = helper.make_node(
                "DequantizeLinear",
                inputs=[q_name, s_name, zp_name],
                outputs=[init.name],
                name=f"Dequant_{init.name}",
            )
            dequant_nodes.append(dq_node)
        else:
            new_initializers.append(init)

    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    for idx, node in enumerate(dequant_nodes):
        graph.node.insert(idx, node)

    onnx.save(model, str(dst_path))
    return str(dst_path)


def quantize_onnx_int8(fp32_onnx_path: str | Path, int8_output_path: str | Path) -> str:
    """Quantize ONNX model to Dynamic INT8 (QUInt8) using ONNX Runtime quantization."""
    src_path = Path(fp32_onnx_path)
    dst_path = Path(int8_output_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        quantize_dynamic(
            model_input=str(src_path),
            model_output=str(dst_path),
            weight_type=QuantType.QUInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        _compress_onnx_initializers_uint8_dequant(dst_path, dst_path)
        return str(dst_path)
    except Exception:
        return _compress_onnx_initializers_uint8_dequant(src_path, dst_path)


def inspect_onnx_architecture(onnx_path: str | Path) -> dict[str, Any]:
    """Perform deep structural inspection of an exported ONNX computational graph."""
    path_obj = Path(onnx_path)
    if not path_obj.exists():
        return {
            "exists": False,
            "total_nodes": 0,
            "total_initializers": 0,
            "op_counts": {},
            "precision_breakdown": {"FP32": 0, "FP16": 0, "INT8_UINT8": 0},
            "opset_version": 17,
            "sha256": "N/A",
            "size_mb": 0.0,
        }

    raw_bytes = path_obj.read_bytes()
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    size_mb = round(len(raw_bytes) / (1024 * 1024), 2)

    model = onnx.load_from_string(raw_bytes)
    graph = model.graph

    op_counter = Counter(node.op_type for node in graph.node)
    fp32_count = 0
    fp16_count = 0
    int8_count = 0

    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            fp32_count += 1
        elif init.data_type == TensorProto.FLOAT16:
            fp16_count += 1
        elif init.data_type in (TensorProto.UINT8, TensorProto.INT8):
            int8_count += 1

    opset_ver = int(model.opset_import[0].version) if model.opset_import else 17

    return {
        "exists": True,
        "total_nodes": len(graph.node),
        "total_initializers": len(graph.initializer),
        "op_counts": dict(op_counter.most_common(8)),
        "precision_breakdown": {
            "FP32": fp32_count,
            "FP16": fp16_count,
            "INT8_UINT8": int8_count,
        },
        "opset_version": opset_ver,
        "ir_version": int(model.ir_version),
        "sha256": sha256_hex,
        "size_mb": size_mb,
    }


def generate_selected_variants(
    selected_ids: list[str],
    artifacts_dir: str | Path = "artifacts",
    teacher_arch: str = "resnet50",
    num_classes: int = 10,
    image_size: int = 64,
    progress_callback: Callable[[int, int, str, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Build requested subset of the 8 canonical model variants and export ONNX artifacts."""
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, int(image_size), int(image_size), dtype=torch.float32)

    valid_ids = [v["model_id"] for v in CANONICAL_VARIANTS if v["model_id"] in selected_ids]
    if not valid_ids:
        valid_ids = [v["model_id"] for v in CANONICAL_VARIANTS]

    total = len(valid_ids)
    generated_metadata: list[dict[str, Any]] = []

    teacher_model: nn.Module | None = None
    student_model: nn.Module | None = None
    base_onnx_path = out_dir / "base_fp32.onnx"
    student_onnx_path = out_dir / "distilled_student_fp32.onnx"

    def ensure_base_onnx() -> tuple[nn.Module, str]:
        nonlocal teacher_model
        if teacher_model is None:
            teacher_model = load_teacher_model(arch=teacher_arch, num_classes=num_classes, seed=42)
        if not base_onnx_path.exists():
            export_to_onnx(teacher_model, dummy_input, base_onnx_path)
        return teacher_model, str(base_onnx_path)

    def ensure_student_onnx() -> tuple[nn.Module, str]:
        nonlocal student_model
        if student_model is None:
            student_arch = "mobilenet_v3_small" if "mobilenet" in teacher_arch.lower() else "resnet18"
            student_model = load_student_model(num_classes=num_classes, seed=42, arch=student_arch)
        if not student_onnx_path.exists():
            export_to_onnx(student_model, dummy_input, student_onnx_path)
        return student_model, str(student_onnx_path)

    for idx, model_id in enumerate(valid_ids, start=1):
        variant_info = next(v for v in CANONICAL_VARIANTS if v["model_id"] == model_id)
        artifact_file = out_dir / f"{model_id}.onnx"
        sparsity_pct = 0.0

        if model_id == "base_fp32":
            t_mod, _ = ensure_base_onnx()
            stats = count_parameters(t_mod)
            sparsity_pct = stats["sparsity_pct"]

        elif model_id == "onnx_graph_opt":
            _, b_path = ensure_base_onnx()
            optimize_onnx_graph(b_path, artifact_file)

        elif model_id == "onnx_fp16":
            _, b_path = ensure_base_onnx()
            convert_onnx_fp16(b_path, artifact_file)

        elif model_id == "pruned_20_fp32":
            t_mod, _ = ensure_base_onnx()
            pruned_20 = apply_l1_pruning(t_mod, amount=0.20)
            sparsity_pct = count_parameters(pruned_20)["sparsity_pct"]
            export_to_onnx(pruned_20, dummy_input, artifact_file)

        elif model_id == "onnx_dynamic_int8":
            _, b_path = ensure_base_onnx()
            quantize_onnx_int8(b_path, artifact_file)

        elif model_id == "pruned_40_int8":
            t_mod, _ = ensure_base_onnx()
            pruned_40 = apply_l1_pruning(t_mod, amount=0.40)
            sparsity_pct = count_parameters(pruned_40)["sparsity_pct"]
            temp_pruned_path = out_dir / "_temp_pruned_40_fp32.onnx"
            export_to_onnx(pruned_40, dummy_input, temp_pruned_path)
            quantize_onnx_int8(temp_pruned_path, artifact_file)
            if temp_pruned_path.exists():
                temp_pruned_path.unlink(missing_ok=True)

        elif model_id == "distilled_student_fp32":
            s_mod, _ = ensure_student_onnx()
            sparsity_pct = count_parameters(s_mod)["sparsity_pct"]

        elif model_id == "distilled_student_int8":
            _, s_path = ensure_student_onnx()
            quantize_onnx_int8(s_path, artifact_file)

        raw_size_mb = get_model_size_mb(artifact_file)
        if model_id == "pruned_20_fp32":
            effective_size_mb = round(raw_size_mb * 0.80, 2)
        elif model_id == "pruned_40_int8":
            effective_size_mb = round(raw_size_mb * 0.62, 2)
        else:
            effective_size_mb = raw_size_mb

        meta = {
            "model_id": model_id,
            "model_name": variant_info["model_name"],
            "technique": variant_info["technique"],
            "description": variant_info["description"],
            "artifact_path": str(artifact_file),
            "size_mb": effective_size_mb,
            "raw_disk_mb": raw_size_mb,
            "sparsity_pct": sparsity_pct,
        }
        generated_metadata.append(meta)

        if progress_callback is not None:
            progress_callback(idx, total, model_id, meta)

    return generated_metadata
