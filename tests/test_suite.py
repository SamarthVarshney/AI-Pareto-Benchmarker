"""Automated pytest verification suite for AI Energy Pareto Benchmarker."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auth.session_manager import (
    authenticate_user,
    get_security_posture_report,
    initialize_judge_session,
    probe_hardware_environment,
    sign_deployment_manifest,
    validate_uploaded_artifact,
    verify_deployment_manifest,
    verify_session_token,
)
from data.loader import get_dataset_summary, get_eval_batch, load_cached_telemetry
from optimizer.base_models import (
    count_parameters,
    get_model_size_mb,
    load_student_model,
    load_teacher_model,
)
from optimizer.onnx_pipeline import export_to_onnx, inspect_onnx_architecture
from optimizer.pruner import apply_l1_pruning, compute_prunable_sparsity
from pareto.frontier import compute_pareto_frontier, compute_pareto_hypervolume
from pareto.recommender import (
    diagnose_infeasible_constraints,
    generate_deployment_bundle_templates,
    recommend_model,
    simulate_enterprise_impact,
)
from ui.charts import (
    build_3d_pareto_scatter_figure,
    build_normalized_tradeoff_bar_figure,
    build_pareto_frontier_2d_figure,
)


def test_auth_and_hardware_probe() -> None:
    """Verify hardware telemetry probe, engineer login, and judge quick-access bypass."""
    hw = probe_hardware_environment()
    assert isinstance(hw, dict)
    assert hw["logical_cores"] >= 1
    assert hw["ram_total_gb"] > 0.0
    assert hw["ram_available_gb"] > 0.0
    assert "CPUExecutionProvider" in hw["onnx_providers"]

    ok, session = authenticate_user("engineer@pareto.ai", "pareto2026")
    assert ok is True
    assert session["user_id"] == "engineer@pareto.ai"
    assert "login_timestamp" in session
    assert verify_session_token(session) is True

    bad_ok, _ = authenticate_user("", "12")
    assert bad_ok is False

    judge_session = initialize_judge_session(
        role="Hackathon Judge",
        preset_mode="Instant Pre-Profiled Telemetry",
    )
    assert judge_session["role"] == "Hackathon Judge"
    assert judge_session["preset_mode"] == "Instant Pre-Profiled Telemetry"
    assert verify_session_token(judge_session) is True


def test_deterministic_data_loader() -> None:
    """Verify get_eval_batch(seed=42) returns identical tensors and accurate RAM footprint."""
    x1, y1 = get_eval_batch(num_samples=32, num_classes=10, image_size=64, seed=42)
    x2, y2 = get_eval_batch(num_samples=32, num_classes=10, image_size=64, seed=42)

    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
    assert x1.shape == (32, 3, 64, 64)
    assert y1.shape == (32,)

    summary = get_dataset_summary(x1, y1)
    expected_mb = round((32 * 3 * 64 * 64 * 4) / (1024 * 1024), 3)
    assert summary["num_samples"] == 32
    assert summary["input_shape"] == (32, 3, 64, 64)
    assert summary["tensor_memory_mb"] == expected_mb


def test_teacher_and_student_registry() -> None:
    """Verify Teacher (ResNet-50) parameter count and size exceed Student (ResNet-18)."""
    teacher = load_teacher_model(arch="resnet50", num_classes=10, seed=42)
    student = load_student_model(num_classes=10, seed=42)

    t_stats = count_parameters(teacher)
    s_stats = count_parameters(student)

    assert t_stats["total_params"] > s_stats["total_params"]
    assert t_stats["total_params"] > 20_000_000
    assert s_stats["total_params"] > 10_000_000

    t_size = get_model_size_mb(teacher)
    s_size = get_model_size_mb(student)
    assert t_size > s_size
    assert t_size > 80.0
    assert s_size > 40.0


def test_l1_pruning_sparsity() -> None:
    """Verify apply_l1_pruning(model, amount=0.3) permanently zeroes ~30% of Conv2d/Linear weights."""
    student = load_student_model(num_classes=10, seed=42)
    pruned = apply_l1_pruning(student, amount=0.30)

    # Confirm pruning hooks were permanently removed
    for module in pruned.modules():
        assert not hasattr(module, "weight_mask")
        assert not hasattr(module, "weight_orig")

    prunable_sparsity = compute_prunable_sparsity(pruned)
    assert 29.5 <= prunable_sparsity <= 30.5

    overall_stats = count_parameters(pruned)
    assert 29.0 <= overall_stats["sparsity_pct"] <= 30.5


def test_pareto_frontier_mathematics() -> None:
    """Verify exact Pareto dominance classification on cached_results.csv."""
    df = load_cached_telemetry()
    pareto_df = compute_pareto_frontier(df).set_index("model_id")

    # Strictly dominated models
    assert bool(pareto_df.loc["pruned_20_fp32", "is_pareto"]) is False
    assert bool(pareto_df.loc["distilled_student_fp32", "is_pareto"]) is False
    assert bool(pareto_df.loc["pruned_40_int8", "is_pareto"]) is False
    assert bool(pareto_df.loc["base_fp32", "is_pareto"]) is False

    # Non-dominated Pareto frontier models
    assert bool(pareto_df.loc["onnx_graph_opt", "is_pareto"]) is True
    assert bool(pareto_df.loc["onnx_fp16", "is_pareto"]) is True
    assert bool(pareto_df.loc["onnx_dynamic_int8", "is_pareto"]) is True
    assert bool(pareto_df.loc["distilled_student_int8", "is_pareto"]) is True


def test_constraint_recommender_and_fallback() -> None:
    """Verify recommend_model selects onnx_dynamic_int8 and handles infeasible constraints."""
    df = load_cached_telemetry()

    annotated_df, winner = recommend_model(
        df=df,
        min_accuracy=91.0,
        max_energy_wh=2.0,
        max_latency_ms=20.0,
        efficiency_weight=0.65,
    )
    assert winner is not None
    assert winner["model_id"] == "onnx_dynamic_int8"
    assert winner["is_pareto"] is True
    assert winner["meets_constraints"] is True
    assert winner["energy_saved_pct"] == 76.0
    assert winner["latency_speedup_x"] == 3.75

    # Infeasible constraint check (A_min = 99.0%)
    infeasible_df, no_winner = recommend_model(
        df=df,
        min_accuracy=99.0,
        max_energy_wh=2.0,
        max_latency_ms=20.0,
        efficiency_weight=0.65,
    )
    assert no_winner is None
    assert int(infeasible_df["meets_constraints"].sum()) == 0

    diag = diagnose_infeasible_constraints(
        df=infeasible_df,
        min_accuracy=99.0,
        max_energy_wh=2.0,
        max_latency_ms=20.0,
    )
    assert diag["acc_pass_count"] == 0
    assert len(diag["reasons"]) >= 1
    assert diag["fallback_model_id"] in (
        "onnx_graph_opt",
        "onnx_fp16",
        "onnx_dynamic_int8",
        "distilled_student_int8",
    )

    # Verify enterprise simulator and Plotly chart builders
    sim = simulate_enterprise_impact(annotated_df, winner, daily_inferences=10_000_000, cost_per_kwh=0.12)
    assert sim["saved_annual_kwh"] > 0
    assert sim["saved_annual_cost_usd"] > 0
    assert sim["saved_annual_co2_kg"] > 0

    fig_2d = build_pareto_frontier_2d_figure(annotated_df, 91.0, 2.0, winner)
    fig_3d = build_3d_pareto_scatter_figure(annotated_df)
    fig_bar = build_normalized_tradeoff_bar_figure(annotated_df)
    assert len(fig_2d.data) >= 2
    assert len(fig_3d.data) >= 1
    assert len(fig_bar.data) == 3


def test_pareto_hypervolume_and_marginal_elasticity() -> None:
    """Verify 2D/3D Pareto Hypervolume indicator, Knee-Point detection, and marginal elasticity."""
    df = load_cached_telemetry()
    hv = compute_pareto_hypervolume(df)

    assert hv["pareto_count"] == 4
    assert hv["dominated_count"] == 4
    assert 50.0 <= hv["hypervolume_2d_pct"] <= 100.0
    assert 40.0 <= hv["hypervolume_3d_pct"] <= 100.0
    assert hv["knee_model_id"] in ("onnx_fp16", "onnx_dynamic_int8")
    assert len(hv["marginal_elasticity"]) == 4


def test_multi_region_carbon_and_cluster_qps_scalability() -> None:
    """Verify multi-region grid factors, fleet worker reduction, and deployment bundle generation."""
    df = load_cached_telemetry()
    annotated_df, winner = recommend_model(df, min_accuracy=90.5, max_energy_wh=2.5, max_latency_ms=25.0)
    assert winner is not None

    sim_ind = simulate_enterprise_impact(
        annotated_df, winner, daily_inferences=20_000_000, cost_per_kwh=0.12, grid_region="IND", horizon_years=3
    )
    sim_nordic = simulate_enterprise_impact(
        annotated_df, winner, daily_inferences=20_000_000, cost_per_kwh=0.12, grid_region="NORDIC", horizon_years=3
    )

    assert sim_ind["saved_annual_co2_kg"] > sim_nordic["saved_annual_co2_kg"]
    assert sim_ind["multi_year_saved_kwh"] == round(sim_ind["saved_annual_kwh"] * 3, 1)
    assert sim_ind["winner_qps_per_worker"] > sim_ind["baseline_qps_per_worker"]
    assert sim_ind["cluster_reduction_pct"] > 50.0
    assert sim_ind["equivalent_trees_planted"] > 0.0

    bundles = generate_deployment_bundle_templates(
        winner=winner,
        constraints={"min_accuracy": 90.5, "max_energy_wh": 2.5, "max_latency_ms": 25.0},
        sha256_hash="a" * 64,
    )
    assert "FROM python:3.12-slim" in bundles["dockerfile"]
    assert "FastAPI" in bundles["microservice_py"]
    assert "Pareto-SLA-Promotion-Gate" in bundles["ci_gate_yaml"]


def test_security_tokens_manifest_signing_and_upload_sanitizer() -> None:
    """Verify HMAC manifest attestation, tamper rejection, and malicious upload blocking."""
    sample_manifest = {
        "schema_version": "1.0.0",
        "recommended_winner": {"model_id": "onnx_dynamic_int8", "accuracy_pct": 91.7},
    }
    signed = sign_deployment_manifest(sample_manifest)
    assert verify_deployment_manifest(signed) is True

    # Tamper with accuracy inside signed manifest and verify rejection
    tampered = dict(signed)
    tampered["recommended_winner"] = {"model_id": "onnx_dynamic_int8", "accuracy_pct": 99.9}
    assert verify_deployment_manifest(tampered) is False

    # Safe ONNX binary payload check
    safe_scan = validate_uploaded_artifact("candidate.onnx", b"ONNX_SAFE_GRAPH_BYTES_123456")
    assert safe_scan["is_safe"] is True
    assert len(safe_scan["sha256"]) == 64

    # Unsafe extension check
    bad_ext = validate_uploaded_artifact("exploit.exe", b"MZ_EXECUTABLE")
    assert bad_ext["is_safe"] is False

    # Malicious pickle code execution payload check
    malicious_pickle = b"\x80\x04cos\nsystem\nq\x00X\x07\x00\x00\x00calc.exeq\x01\x85q\x02Rq\x03."
    blocked_scan = validate_uploaded_artifact("trojan_model.pt", malicious_pickle)
    assert blocked_scan["is_safe"] is False
    assert "opcode" in blocked_scan["reason"].lower()

    posture = get_security_posture_report()
    assert "PBKDF2" in posture["credential_hashing"]


def test_onnx_graph_inspector_on_artifact(tmp_path: Path) -> None:
    """Verify ONNX graph structural inspector accurately counts operators and initializers."""
    tiny_model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 10),
    ).eval()
    dummy_x = torch.randn(1, 3, 64, 64)
    onnx_file = tmp_path / "tiny_test.onnx"
    export_to_onnx(tiny_model, dummy_x, onnx_file)

    info = inspect_onnx_architecture(onnx_file)
    assert info["exists"] is True
    assert info["total_nodes"] >= 3
    assert info["total_initializers"] >= 2
    assert info["precision_breakdown"]["FP32"] >= 2
    assert len(info["sha256"]) == 64
