"""Multi-objective Pareto dominance solver, Hypervolume indicator, and marginal elasticity analyzer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("accuracy_pct", "energy_wh_per_1k", "latency_ms")


def compute_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Compute exact multi-objective Pareto frontier (Max Accuracy, Min Energy, Min Latency).

    Candidate `a` Pareto-dominates candidate `b` iff:
        (A_a >= A_b) and (E_a <= E_b) and (L_a <= L_b)
    and at least one inequality is strict:
        (A_a > A_b) or (E_a < E_b) or (L_a < L_b)
    """
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column for Pareto analysis: '{col}'")

    result_df = df.copy().reset_index(drop=True)
    n_models = len(result_df)

    if n_models == 0:
        result_df["is_pareto"] = pd.Series(dtype=bool)
        result_df["pareto_status"] = pd.Series(dtype=str)
        result_df["dominated_by"] = pd.Series(dtype=str)
        result_df["domination_count"] = pd.Series(dtype=int)
        return result_df

    acc = result_df["accuracy_pct"].to_numpy(dtype=np.float64)
    eng = result_df["energy_wh_per_1k"].to_numpy(dtype=np.float64)
    lat = result_df["latency_ms"].to_numpy(dtype=np.float64)
    model_ids = (
        result_df["model_id"].astype(str).tolist()
        if "model_id" in result_df.columns
        else [f"model_{i}" for i in range(n_models)]
    )

    is_pareto = np.ones(n_models, dtype=bool)
    dominated_by_list: list[str] = []
    domination_counts: list[int] = []

    for i in range(n_models):
        dominators: list[str] = []
        for j in range(n_models):
            if i == j:
                continue
            ge_all = (acc[j] >= acc[i]) and (eng[j] <= eng[i]) and (lat[j] <= lat[i])
            strict_any = (acc[j] > acc[i]) or (eng[j] < eng[i]) or (lat[j] < lat[i])
            if ge_all and strict_any:
                is_pareto[i] = False
                dominators.append(model_ids[j])
        dominated_by_list.append(", ".join(dominators) if dominators else "None")
        domination_counts.append(len(dominators))

    result_df["is_pareto"] = is_pareto
    result_df["pareto_status"] = np.where(is_pareto, "[OPTIMAL]", "[DOMINATED]")
    result_df["dominated_by"] = dominated_by_list
    result_df["domination_count"] = domination_counts
    return result_df


def compute_pareto_hypervolume(
    df: pd.DataFrame,
    ref_accuracy: float = 88.0,
    ref_energy_wh: float = 5.20,
    ref_latency_ms: float = 50.0,
) -> dict[str, Any]:
    """Compute 2D and 3D normalized Pareto Hypervolume (HV), Knee-Point, and marginal elasticity.

    Uses standard multi-objective optimization theory (Zitzler & Thiele):
    - Normalizes objectives to [0, 1] where 1.0 is the utopia ideal and 0.0 is the nadir reference.
    - Integrates the exact 2D Lebesque measure (area) dominated by the Pareto frontier in
      (Accuracy, Energy) space.
    - Identifies the mathematical Knee Point (minimum Euclidean distance to the Utopia Point [1, 1, 1]).
    - Computes step-by-step marginal energy cost (Delta Wh / Delta 1% Accuracy) along the frontier.
    """
    work_df = compute_pareto_frontier(df) if "is_pareto" not in df.columns else df.copy()
    pareto_df = work_df[work_df["is_pareto"] == True].sort_values(
        by="energy_wh_per_1k", ascending=True
    ).reset_index(drop=True)

    if pareto_df.empty:
        return {
            "hypervolume_2d_pct": 0.0,
            "hypervolume_3d_pct": 0.0,
            "pareto_count": 0,
            "dominated_count": len(work_df),
            "knee_model_id": "None",
            "knee_model_name": "None",
            "marginal_elasticity": [],
        }

    max_acc = max(float(work_df["accuracy_pct"].max()), ref_accuracy + 1.0)
    min_eng = min(float(work_df["energy_wh_per_1k"].min()), 0.40)
    min_lat = min(float(work_df["latency_ms"].min()), 4.0)

    # Normalize each Pareto point to [0, 1] where higher is better
    norm_a = np.clip(
        (pareto_df["accuracy_pct"].to_numpy(dtype=np.float64) - ref_accuracy)
        / max(1e-9, max_acc - ref_accuracy),
        0.0,
        1.0,
    )
    norm_e = np.clip(
        (ref_energy_wh - pareto_df["energy_wh_per_1k"].to_numpy(dtype=np.float64))
        / max(1e-9, ref_energy_wh - min_eng),
        0.0,
        1.0,
    )
    norm_l = np.clip(
        (ref_latency_ms - pareto_df["latency_ms"].to_numpy(dtype=np.float64))
        / max(1e-9, ref_latency_ms - min_lat),
        0.0,
        1.0,
    )

    # Exact 2D Hypervolume integration (sorted from highest norm_e to lowest norm_e)
    order = np.argsort(-norm_e)
    sorted_e = norm_e[order]
    sorted_a = norm_a[order]

    hv_2d = 0.0
    prev_a = 0.0
    for e_val, a_val in zip(sorted_e, sorted_a):
        if a_val > prev_a:
            hv_2d += float(e_val) * float(a_val - prev_a)
            prev_a = float(a_val)

    # 3D Hypervolume approximation combining 2D HV with latency domination depth
    mean_l_depth = float(np.mean(norm_l))
    hv_3d = hv_2d * (0.35 + 0.65 * mean_l_depth)

    # Utopia distance to locate the mathematical Knee Point
    utopia_dist = np.sqrt((1.0 - norm_a) ** 2 + (1.0 - norm_e) ** 2 + (1.0 - norm_l) ** 2)
    knee_idx = int(np.argmin(utopia_dist))
    knee_row = pareto_df.iloc[knee_idx]

    # Marginal Elasticity along the Pareto frontier (sorted by ascending Energy)
    elasticity_rows: list[dict[str, Any]] = []
    for idx in range(len(pareto_df)):
        cur = pareto_df.iloc[idx]
        if idx == 0:
            elasticity_rows.append(
                {
                    "from_model": "Minimum-Power Anchor",
                    "to_model": str(cur["model_id"]),
                    "delta_accuracy_pct": 0.0,
                    "delta_energy_wh": 0.0,
                    "wh_per_1pct_accuracy": 0.0,
                    "energy_multiplier": 1.0,
                }
            )
        else:
            prev = pareto_df.iloc[idx - 1]
            d_acc = float(cur["accuracy_pct"]) - float(prev["accuracy_pct"])
            d_eng = float(cur["energy_wh_per_1k"]) - float(prev["energy_wh_per_1k"])
            wh_per_pct = d_eng / max(0.01, d_acc) if d_acc > 0 else d_eng * 100.0
            mult = float(cur["energy_wh_per_1k"]) / max(1e-6, float(prev["energy_wh_per_1k"]))
            elasticity_rows.append(
                {
                    "from_model": str(prev["model_id"]),
                    "to_model": str(cur["model_id"]),
                    "delta_accuracy_pct": round(d_acc, 2),
                    "delta_energy_wh": round(d_eng, 2),
                    "wh_per_1pct_accuracy": round(wh_per_pct, 2),
                    "energy_multiplier": round(mult, 2),
                }
            )

    return {
        "hypervolume_2d_pct": round(hv_2d * 100.0, 2),
        "hypervolume_3d_pct": round(hv_3d * 100.0, 2),
        "pareto_count": int(len(pareto_df)),
        "dominated_count": int(len(work_df) - len(pareto_df)),
        "knee_model_id": str(knee_row.get("model_id", "")),
        "knee_model_name": str(knee_row.get("model_name", "")),
        "knee_utopia_distance": round(float(utopia_dist[knee_idx]), 4),
        "marginal_elasticity": elasticity_rows,
    }
