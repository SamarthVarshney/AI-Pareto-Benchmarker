"""Interactive Plotly visualizations for 2D Pareto Frontier, 3D Multi-Objective Space, and Bar Matrix."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _assign_smart_text_positions(
    sub_df: pd.DataFrame,
    x_min: float,
    x_max: float,
    winner_id: str | None = None,
) -> tuple[list[str], list[str]]:
    """Compute display labels and non-overlapping text positions for scatter markers."""
    labels: list[str] = []
    positions: list[str] = []
    x_span = max(x_max - x_min, 1e-6)

    for idx, row in sub_df.reset_index(drop=True).iterrows():
        mid = str(row.get("model_id", ""))
        x_val = float(row["energy_wh_per_1k"])
        rel_x = (x_val - x_min) / x_span

        # Omit duplicate raw text label on the winner marker since it gets a dedicated callout box
        if winner_id and mid == winner_id:
            labels.append("")
            positions.append("top center")
            continue

        labels.append(mid)
        if rel_x < 0.22:
            # Near left border: push label to the right so it never clips off the Y-axis
            positions.append("middle right" if idx % 2 == 0 else "bottom right")
        elif rel_x > 0.82:
            positions.append("bottom left")
        else:
            cycle = ["top center", "bottom center", "top right", "bottom right"]
            positions.append(cycle[idx % len(cycle)])

    return labels, positions


def build_pareto_frontier_2d_figure(
    df: pd.DataFrame,
    min_accuracy: float,
    max_energy_wh: float,
    winner: dict[str, Any] | None = None,
) -> go.Figure:
    """Create the 5-layer interactive 2D Pareto Frontier plot (Accuracy % vs Energy Wh/1k).

    Layers specified in Section 3 (View 3, Tab 1):
    - Layer 1 (Feasible Zone Shading): Shaded green rectangular region X <= E_max, Y >= A_min
    - Layer 2 (Threshold Lines): Dashed vertical line at E_max and dashed horizontal line at A_min
    - Layer 3 (Dominated Models): Muted zinc-grey circular markers (#52525B) + Blue Baseline (#3B82F6)
    - Layer 4 (Pareto Frontier Models): Emerald markers (#10B981) connected by frontier curve
    - Layer 5 (Recommended Winner): Highlighted outer diamond ring with non-overlapping callout annotation
    """
    fig = go.Figure()

    data_min_x = float(df["energy_wh_per_1k"].min())
    data_max_x = max(float(df["energy_wh_per_1k"].max()), float(max_energy_wh))
    x_span = max(data_max_x - data_min_x, 1.2)
    min_x = max(0.0, data_min_x - 0.18 * x_span)
    max_x = data_max_x + 0.16 * x_span

    data_min_y = min(float(df["accuracy_pct"].min()), float(min_accuracy))
    data_max_y = max(float(df["accuracy_pct"].max()), float(min_accuracy))
    y_span = max(data_max_y - data_min_y, 1.8)
    min_y = data_min_y - 0.24 * y_span
    max_y = data_max_y + 0.30 * y_span

    winner_id = str(winner["model_id"]) if winner is not None else None

    # Layer 1: Feasible Zone Shading (X <= E_max and Y >= A_min)
    fig.add_shape(
        type="rect",
        x0=min_x,
        x1=float(max_energy_wh),
        y0=float(min_accuracy),
        y1=max_y,
        fillcolor="rgba(16, 185, 129, 0.06)",
        line=dict(color="rgba(16, 185, 129, 0.25)", width=1),
        layer="below",
    )

    # Feasible zone badge tucked cleanly in the top-left corner of the feasible box
    fig.add_annotation(
        x=min_x + 0.03 * x_span,
        y=max_y - 0.05 * y_span,
        text="SLA FEASIBLE REGION [PASS]",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(9, 9, 11, 0.75)",
        bordercolor="rgba(16, 185, 129, 0.35)",
        borderwidth=1,
        borderpad=4,
        font=dict(family="JetBrains Mono, monospace", size=10, color="#10B981"),
    )

    # Layer 2: SLA Threshold Lines
    fig.add_vline(
        x=float(max_energy_wh),
        line_width=1.5,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text=f"  E_max = {max_energy_wh:.2f} Wh  ",
        annotation_position="top right",
        annotation_font=dict(family="JetBrains Mono, monospace", size=10, color="#F59E0B"),
    )
    fig.add_hline(
        y=float(min_accuracy),
        line_width=1.5,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text=f"  A_min = {min_accuracy:.1f}%  ",
        annotation_position="bottom right",
        annotation_font=dict(family="JetBrains Mono, monospace", size=10, color="#F59E0B"),
    )

    def _scale_marker_size(sizes: pd.Series) -> list[float]:
        return [float(np.clip(12.0 + (s / 98.0) * 16.0, 12.0, 28.0)) for s in sizes]

    dominated_df = df[df["is_pareto"] == False].sort_values(by="energy_wh_per_1k", ascending=True).copy()
    pareto_df = df[df["is_pareto"] == True].sort_values(by="energy_wh_per_1k", ascending=True).copy()

    # Layer 3: Dominated Models (#52525B, with base_fp32 highlighted in #3B82F6 Blue)
    if not dominated_df.empty:
        dom_labels, dom_positions = _assign_smart_text_positions(
            dominated_df, min_x, max_x, winner_id=winner_id
        )
        dom_colors = [
            "#3B82F6" if str(mid) == "base_fp32" else "#52525B"
            for mid in dominated_df["model_id"]
        ]
        dom_borders = [
            "#FAFAFA" if str(mid) == "base_fp32" else "#A1A1AA"
            for mid in dominated_df["model_id"]
        ]
        fig.add_trace(
            go.Scatter(
                x=dominated_df["energy_wh_per_1k"],
                y=dominated_df["accuracy_pct"],
                mode="markers+text",
                name="Dominated / Baseline [DOMINATED]",
                text=dom_labels,
                textposition=dom_positions,
                textfont=dict(family="JetBrains Mono, monospace", size=10, color="#A1A1AA"),
                cliponaxis=False,
                marker=dict(
                    size=_scale_marker_size(dominated_df["size_mb"]),
                    color=dom_colors,
                    symbol="circle",
                    line=dict(color=dom_borders, width=1.2),
                    opacity=0.85,
                ),
                customdata=np.stack(
                    [
                        dominated_df["model_name"],
                        dominated_df["technique"],
                        dominated_df["latency_ms"],
                        dominated_df["size_mb"],
                        dominated_df["utility_score"],
                        dominated_df["dominated_by"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> [DOMINATED]<br>"
                    "Technique: %{customdata[1]}<br>"
                    "Accuracy: <b>%{y:.2f}%</b><br>"
                    "Energy: <b>%{x:.2f} Wh / 1k</b><br>"
                    "Latency: %{customdata[2]:.1f} ms<br>"
                    "Size: %{customdata[3]:.1f} MB<br>"
                    "Utility Score: %{customdata[4]:.1f}<br>"
                    "Dominated By: %{customdata[5]}<extra></extra>"
                ),
            )
        )

    # Layer 4: Pareto Frontier Models (#10B981) connected by optimal trade-off boundary
    if not pareto_df.empty:
        par_labels, par_positions = _assign_smart_text_positions(
            pareto_df, min_x, max_x, winner_id=winner_id
        )
        # Subtle stepped trade-off envelope + direct frontier connectors
        fig.add_trace(
            go.Scatter(
                x=pareto_df["energy_wh_per_1k"],
                y=pareto_df["accuracy_pct"],
                mode="lines",
                name="Pareto Step Boundary",
                line=dict(color="rgba(16, 185, 129, 0.30)", width=1.5, dash="dot", shape="hv"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pareto_df["energy_wh_per_1k"],
                y=pareto_df["accuracy_pct"],
                mode="lines+markers+text",
                name="Pareto Frontier [OPTIMAL]",
                text=par_labels,
                textposition=par_positions,
                textfont=dict(family="JetBrains Mono, monospace", size=11, color="#FAFAFA"),
                cliponaxis=False,
                line=dict(color="#10B981", width=2.5, dash="dash"),
                marker=dict(
                    size=_scale_marker_size(pareto_df["size_mb"]),
                    color="#10B981",
                    symbol="circle",
                    line=dict(color="#FAFAFA", width=1.6),
                ),
                customdata=np.stack(
                    [
                        pareto_df["model_name"],
                        pareto_df["technique"],
                        pareto_df["latency_ms"],
                        pareto_df["size_mb"],
                        pareto_df["utility_score"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> [OPTIMAL]<br>"
                    "Technique: %{customdata[1]}<br>"
                    "Accuracy: <b>%{y:.2f}%</b><br>"
                    "Energy: <b>%{x:.2f} Wh / 1k</b><br>"
                    "Latency: %{customdata[2]:.1f} ms<br>"
                    "Size: %{customdata[3]:.1f} MB<br>"
                    "Utility Score: <b>%{customdata[4]:.1f}</b><extra></extra>"
                ),
            )
        )

    # Layer 5: Recommended Winner outer diamond ring + non-overlapping callout annotation
    if winner is not None:
        win_x = float(winner["energy_wh_per_1k"])
        win_y = float(winner["accuracy_pct"])
        win_name = str(winner["model_name"])
        win_score = float(winner["utility_score"])

        fig.add_trace(
            go.Scatter(
                x=[win_x],
                y=[win_y],
                mode="markers",
                name="Recommended Deployment Winner",
                marker=dict(
                    size=30,
                    color="rgba(16, 185, 129, 0.20)",
                    symbol="diamond-open",
                    line=dict(color="#10B981", width=3),
                ),
                hoverinfo="skip",
            )
        )

        # Place callout box in open space below-right or above-left to avoid overlapping neighbors
        rel_win_y = (win_y - min_y) / max(max_y - min_y, 1e-6)
        ay_offset = 46 if rel_win_y > 0.52 else -46
        ax_offset = 65 if (win_x - min_x) / x_span < 0.65 else -65

        fig.add_annotation(
            x=win_x,
            y=win_y,
            text=f"[RECOMMENDED] {win_name}  //  Score: {win_score:.1f}/100",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.6,
            arrowcolor="#10B981",
            ax=ax_offset,
            ay=ay_offset,
            bgcolor="#121215",
            bordercolor="#10B981",
            borderwidth=1.5,
            borderpad=6,
            font=dict(family="JetBrains Mono, monospace", size=11, color="#FAFAFA"),
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#121215",
        plot_bgcolor="#09090B",
        height=560,
        margin=dict(l=64, r=32, t=54, b=86),
        font=dict(family="Inter, sans-serif", color="#FAFAFA"),
        title=dict(
            text="2D PARETO FRONTIER // PREDICTIVE ACCURACY (%) VS. ACTIVE ENERGY DRAW (Wh / 1,000 INFERENCES)",
            font=dict(family="JetBrains Mono, monospace", size=12, color="#A1A1AA"),
            x=0.01,
            y=0.96,
        ),
        xaxis=dict(
            title=dict(
                text="Active Energy Consumption (Wh per 1,000 Inferences) [Lower is Better -> Left]",
                font=dict(family="JetBrains Mono, monospace", size=11, color="#A1A1AA"),
                standoff=14,
            ),
            range=[min_x, max_x],
            gridcolor="#18181B",
            zerolinecolor="#27272A",
            tickfont=dict(family="JetBrains Mono, monospace", size=11),
        ),
        yaxis=dict(
            title=dict(
                text="Top-1 Validation Accuracy (%) [Higher is Better -> Up]",
                font=dict(family="JetBrains Mono, monospace", size=11, color="#A1A1AA"),
                standoff=10,
            ),
            range=[min_y, max_y],
            gridcolor="#18181B",
            zerolinecolor="#27272A",
            tickfont=dict(family="JetBrains Mono, monospace", size=11),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
            bgcolor="#09090B",
            bordercolor="#27272A",
            borderwidth=1,
            font=dict(family="JetBrains Mono, monospace", size=10, color="#FAFAFA"),
        ),
    )
    return fig


def build_3d_pareto_scatter_figure(df: pd.DataFrame) -> go.Figure:
    """Create interactive 3D Multi-Objective Scatter (X=Energy Wh, Y=Latency ms, Z=Accuracy %)."""
    fig = go.Figure()

    pareto_df = df[df["is_pareto"] == True]
    dom_df = df[df["is_pareto"] == False]

    if not dom_df.empty:
        fig.add_trace(
            go.Scatter3d(
                x=dom_df["energy_wh_per_1k"],
                y=dom_df["latency_ms"],
                z=dom_df["accuracy_pct"],
                mode="markers+text",
                name="Dominated [DOMINATED]",
                text=dom_df["model_id"],
                textfont=dict(family="JetBrains Mono, monospace", size=9, color="#71717A"),
                marker=dict(
                    size=7,
                    color="#52525B",
                    opacity=0.8,
                    line=dict(color="#A1A1AA", width=1),
                ),
            )
        )

    if not pareto_df.empty:
        fig.add_trace(
            go.Scatter3d(
                x=pareto_df["energy_wh_per_1k"],
                y=pareto_df["latency_ms"],
                z=pareto_df["accuracy_pct"],
                mode="markers+text",
                name="Pareto Frontier [OPTIMAL]",
                text=pareto_df["model_id"],
                textfont=dict(family="JetBrains Mono, monospace", size=10, color="#10B981"),
                marker=dict(
                    size=9,
                    color="#10B981",
                    symbol="diamond",
                    line=dict(color="#FAFAFA", width=1.5),
                ),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#121215",
        plot_bgcolor="#09090B",
        height=500,
        margin=dict(l=10, r=10, t=48, b=64),
        title=dict(
            text="3D MULTI-OBJECTIVE MANIFOLD (ENERGY x LATENCY x ACCURACY)",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#A1A1AA"),
            x=0.02,
            y=0.95,
        ),
        scene=dict(
            bgcolor="#09090B",
            xaxis=dict(
                title="Energy (Wh/1k)",
                backgroundcolor="#09090B",
                gridcolor="#27272A",
            ),
            yaxis=dict(
                title="Latency (ms)",
                backgroundcolor="#09090B",
                gridcolor="#27272A",
            ),
            zaxis=dict(
                title="Accuracy (%)",
                backgroundcolor="#09090B",
                gridcolor="#27272A",
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(family="JetBrains Mono, monospace", size=10),
            bgcolor="#09090B",
            bordercolor="#27272A",
            borderwidth=1,
        ),
    )
    return fig


def build_normalized_tradeoff_bar_figure(df: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart comparing normalized Size, Latency, and Energy across all variants."""
    work_df = df.copy()
    max_size = max(float(work_df["size_mb"].max()), 1e-6)
    max_lat = max(float(work_df["latency_ms"].max()), 1e-6)
    max_eng = max(float(work_df["energy_wh_per_1k"].max()), 1e-6)

    size_pct = np.round((work_df["size_mb"] / max_size) * 100.0, 1)
    lat_pct = np.round((work_df["latency_ms"] / max_lat) * 100.0, 1)
    eng_pct = np.round((work_df["energy_wh_per_1k"] / max_eng) * 100.0, 1)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=work_df["model_id"],
            y=eng_pct,
            name="Relative Energy (%)",
            marker_color="#10B981",
        )
    )
    fig.add_trace(
        go.Bar(
            x=work_df["model_id"],
            y=lat_pct,
            name="Relative Latency (%)",
            marker_color="#3B82F6",
        )
    )
    fig.add_trace(
        go.Bar(
            x=work_df["model_id"],
            y=size_pct,
            name="Relative Disk Size (%)",
            marker_color="#A1A1AA",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        paper_bgcolor="#121215",
        plot_bgcolor="#09090B",
        height=500,
        margin=dict(l=48, r=24, t=48, b=86),
        title=dict(
            text="NORMALIZED RESOURCE FOOTPRINT (% OF BASELINE FP32 TEACHER)",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#A1A1AA"),
            x=0.02,
            y=0.95,
        ),
        xaxis=dict(
            title="Candidate Model Identifier",
            tickfont=dict(family="JetBrains Mono, monospace", size=10),
            gridcolor="#18181B",
        ),
        yaxis=dict(
            title="% of Baseline Maximum (Lower is Better)",
            tickfont=dict(family="JetBrains Mono, monospace", size=10),
            gridcolor="#18181B",
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(family="JetBrains Mono, monospace", size=10),
            bgcolor="#09090B",
            bordercolor="#27272A",
            borderwidth=1,
        ),
    )
    return fig
