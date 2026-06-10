"""
Linear-themed Plotly chart builders.

Every figure shares one registered template (``linear_dark``) so the dashboard
reads as a single dark, lavender-accented surface that matches
DESIGN-linear.app.md. Functions take plain data (DataFrames / arrays / the
training history dict) and return ``plotly.graph_objects.Figure`` objects; they
do not touch Streamlit, so they stay unit-testable.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from mcdm.scoring import MCDMResult
from utils import concepts as C
from utils.config import (
    CHART_SEQUENCE,
    COLORS,
    MASTERY_BANDS,
    MCDM_CRITERIA,
)

# ---------------------------------------------------------------------------
# Shared template.
# ---------------------------------------------------------------------------
_LINEAR_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=COLORS["canvas"],
        plot_bgcolor=COLORS["surface_1"],
        font=dict(color=COLORS["ink"], family="Inter, SF Pro Display, system-ui, sans-serif", size=13),
        title=dict(font=dict(color=COLORS["ink"], size=18)),
        colorway=CHART_SEQUENCE,
        xaxis=dict(gridcolor=COLORS["hairline"], zerolinecolor=COLORS["hairline"],
                   linecolor=COLORS["hairline"], tickfont=dict(color=COLORS["ink_subtle"])),
        yaxis=dict(gridcolor=COLORS["hairline"], zerolinecolor=COLORS["hairline"],
                   linecolor=COLORS["hairline"], tickfont=dict(color=COLORS["ink_subtle"])),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["ink_muted"])),
        margin=dict(l=60, r=30, t=50, b=50),
    )
)
pio.templates["linear_dark"] = _LINEAR_TEMPLATE

# Lavender -> light continuous scale for heatmaps (no second chromatic accent).
LAVENDER_SCALE = [
    [0.0, COLORS["canvas"]],
    [0.35, "#23263a"],
    [0.7, COLORS["primary"]],
    [1.0, COLORS["primary_hover"]],
]


def _base(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    fig.update_layout(template="linear_dark", height=height)
    if title:
        fig.update_layout(title=title)
    return fig


# ---------------------------------------------------------------------------
# 1. Concept dependency graph.
# ---------------------------------------------------------------------------
def concept_graph_figure(graph: nx.DiGraph, highlight: int | None = None,
                         path: list[int] | None = None) -> go.Figure:
    """Interactive concept DAG. Optionally highlight a node and/or a path."""
    # Layered layout: x = topological position, y spread within a layer.
    topo_pos = C.topological_position()
    layers: dict[int, list[int]] = {}
    for cid, pos in topo_pos.items():
        layers.setdefault(pos, []).append(cid)
    coords: dict[int, tuple[float, float]] = {}
    for pos, nodes in layers.items():
        for j, cid in enumerate(sorted(nodes)):
            coords[cid] = (pos, j - (len(nodes) - 1) / 2.0)

    path_edges = set()
    if path:
        path_edges = {(path[i], path[i + 1]) for i in range(len(path) - 1)}

    edge_x, edge_y = [], []
    hl_x, hl_y = [], []
    for u, v in graph.edges():
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        if (u, v) in path_edges:
            hl_x += [x0, x1, None]
            hl_y += [y0, y1, None]
        else:
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(color=COLORS["hairline_strong"], width=1),
                            hoverinfo="none", showlegend=False)
    traces = [edge_trace]
    if hl_x:
        traces.append(go.Scatter(x=hl_x, y=hl_y, mode="lines",
                                 line=dict(color=COLORS["primary"], width=3),
                                 hoverinfo="none", showlegend=False))

    node_x, node_y, text, color, size, customdata = [], [], [], [], [], []
    for cid in C.all_ids():
        x, y = coords[cid]
        node_x.append(x)
        node_y.append(y)
        data = graph.nodes[cid]
        text.append(data["name"])
        is_hl = (highlight is not None and cid == highlight) or (path and cid in path)
        color.append(COLORS["primary"] if is_hl else COLORS["surface_2"])
        size.append(30 if is_hl else 22)
        prereqs = ", ".join(C.concept_name(p) for p in data["prerequisites"]) or "none"
        customdata.append([data["name"], data["difficulty"], prereqs,
                           data["track"], data.get("avg_mastery", 0.0)])

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=text, textposition="top center",
        textfont=dict(color=COLORS["ink_muted"], size=10),
        marker=dict(size=size, color=color, line=dict(color=COLORS["primary_hover"], width=1.2)),
        customdata=customdata,
        hovertemplate=("<b>%{customdata[0]}</b><br>Difficulty: %{customdata[1]}/5<br>"
                       "Track: %{customdata[3]}<br>Prerequisites: %{customdata[2]}<br>"
                       "Avg mastery: %{customdata[4]:.2f}<extra></extra>"),
        showlegend=False,
    )
    traces.append(node_trace)

    fig = go.Figure(data=traces)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base(fig, height=560, title="Concept Dependency Graph")


# ---------------------------------------------------------------------------
# 2. Mastery progress.
# ---------------------------------------------------------------------------
def mastery_progress_figure(df: pd.DataFrame) -> go.Figure:
    """Distribution of students across mastery bands."""
    from rl.environment import discretize_mastery

    bands = df["Current_Mastery"].apply(discretize_mastery)
    counts = bands.value_counts().reindex(range(len(MASTERY_BANDS)), fill_value=0)
    fig = go.Figure(go.Bar(
        x=MASTERY_BANDS, y=counts.values,
        marker_color=CHART_SEQUENCE[:len(MASTERY_BANDS)],
        text=counts.values, textposition="outside",
    ))
    fig.update_layout(xaxis_title="Mastery band", yaxis_title="Students")
    return _base(fig, title="Mastery Band Distribution")


def mastery_growth_vs_study_figure(df: pd.DataFrame) -> go.Figure:
    """Scatter showing the consistency rule: more study hours -> more growth."""
    sample = df.sample(min(600, len(df)), random_state=42)
    fig = go.Figure(go.Scatter(
        x=sample["Study_Time_Hours"], y=sample["Mastery_Growth"],
        mode="markers",
        marker=dict(color=sample["Interest_Level"], colorscale=LAVENDER_SCALE,
                    showscale=True, colorbar=dict(title="Interest"), size=7, opacity=0.7),
        hovertemplate="Study: %{x:.1f}h<br>Growth: %{y:.2f}<extra></extra>",
    ))
    # Trend line.
    z = np.polyfit(df["Study_Time_Hours"], df["Mastery_Growth"], 1)
    xs = np.linspace(df["Study_Time_Hours"].min(), df["Study_Time_Hours"].max(), 50)
    fig.add_trace(go.Scatter(x=xs, y=np.polyval(z, xs), mode="lines",
                             line=dict(color=COLORS["primary_hover"], width=2, dash="dash"),
                             name="Trend"))
    fig.update_layout(xaxis_title="Weekly study hours", yaxis_title="Mastery growth",
                      showlegend=False)
    return _base(fig, title="Mastery Growth vs Study Time")


# ---------------------------------------------------------------------------
# 3. Reward / learning curves.
# ---------------------------------------------------------------------------
def _moving_average(values: list[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def reward_curve_figure(history: dict) -> go.Figure:
    """Per-episode reward (faint) with a moving-average overlay."""
    rewards = history["episode_rewards"]
    window = max(10, len(rewards) // 50)
    ma = _moving_average(rewards, window)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=rewards, mode="lines", name="Episode reward",
                             line=dict(color=COLORS["ink_tertiary"], width=1), opacity=0.5))
    fig.add_trace(go.Scatter(x=np.arange(window - 1, window - 1 + len(ma)), y=ma,
                             mode="lines", name=f"Moving avg ({window})",
                             line=dict(color=COLORS["primary"], width=2.5)))
    fig.update_layout(xaxis_title="Episode", yaxis_title="Total reward")
    return _base(fig, title="Reward Curve (training convergence)")


def learning_curve_figure(history: dict) -> go.Figure:
    """Cumulative mean reward — the classic convergence/learning curve."""
    rewards = np.asarray(history["episode_rewards"], dtype=float)
    cummean = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
    fig = go.Figure(go.Scatter(y=cummean, mode="lines",
                               line=dict(color=COLORS["primary"], width=2.5)))
    fig.update_layout(xaxis_title="Episode", yaxis_title="Cumulative mean reward")
    return _base(fig, title="Learning Curve")


# ---------------------------------------------------------------------------
# 4. Q-value heatmap.
# ---------------------------------------------------------------------------
def q_heatmap_figure(q_table: np.ndarray) -> go.Figure:
    """
    Mean Q-value per (current concept, action concept), averaged over the
    mastery and study-time bands. A bright lavender cell means "from this
    concept, moving to that concept is highly valued".
    """
    from utils.config import NUM_MASTERY_BANDS, NUM_STUDY_TIME_BANDS

    n = C.NUM_CONCEPTS
    block = NUM_MASTERY_BANDS * NUM_STUDY_TIME_BANDS
    agg = np.zeros((n, n))
    for cur in range(n):
        rows = q_table[cur * block:(cur + 1) * block]
        agg[cur] = rows.mean(axis=0)

    names = C.all_names()
    fig = go.Figure(go.Heatmap(
        z=agg, x=names, y=names, colorscale=LAVENDER_SCALE,
        colorbar=dict(title="Mean Q"),
        hovertemplate="From %{y}<br>To %{x}<br>Q=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Action (next concept)", yaxis_title="Current concept",
                      xaxis=dict(tickangle=-45))
    return _base(fig, height=620, title="Q-Value Heatmap (averaged over bands)")


# ---------------------------------------------------------------------------
# 5. Policy evolution / stability.
# ---------------------------------------------------------------------------
def policy_evolution_figure(history: dict) -> go.Figure:
    """% of states whose greedy action is unchanged between checkpoints."""
    ps = history.get("policy_stability") or []
    x = list(range(2, len(ps) + 2))  # checkpoint index (first comparison is #2)
    fig = go.Figure(go.Scatter(x=x, y=[p * 100 for p in ps], mode="lines+markers",
                               line=dict(color=COLORS["primary"], width=2.5),
                               marker=dict(color=COLORS["primary_hover"], size=7)))
    fig.update_layout(xaxis_title="Training checkpoint",
                      yaxis_title="States with unchanged greedy action (%)",
                      yaxis=dict(range=[0, 101]))
    return _base(fig, title="Policy Stability Across Checkpoints")


# ---------------------------------------------------------------------------
# 6. RL vs MCDM recommendation comparison.
# ---------------------------------------------------------------------------
def rl_vs_mcdm_figure(rl_ranking: list[tuple[int, float]],
                      mcdm_results: list[MCDMResult]) -> go.Figure:
    """
    Two side-by-side ranked lists: RL order (by Q) and MCDM order (by score),
    drawn as horizontal bars so re-ranking is visually obvious.
    """
    top_ids = [cid for cid, _ in rl_ranking[:len(mcdm_results)]]
    q_by_id = dict(rl_ranking)

    rl_names = [C.concept_name(cid) for cid in top_ids]
    rl_vals = [q_by_id[cid] for cid in top_ids]
    mcdm_names = [r.name for r in mcdm_results]
    mcdm_vals = [r.score for r in mcdm_results]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=rl_names[::-1], x=rl_vals[::-1], orientation="h",
                         name="RL (Q-value)", marker_color=COLORS["ink_subtle"],
                         xaxis="x1", yaxis="y1"))
    fig.add_trace(go.Bar(y=mcdm_names[::-1], x=mcdm_vals[::-1], orientation="h",
                         name="MCDM (score)", marker_color=COLORS["primary"],
                         xaxis="x2", yaxis="y2"))
    fig.update_layout(
        grid=dict(rows=1, columns=2, pattern="independent"),
        xaxis1=dict(title="Q-value", domain=[0, 0.45]),
        xaxis2=dict(title="MCDM score", domain=[0.55, 1.0]),
        yaxis1=dict(anchor="x1"), yaxis2=dict(anchor="x2"),
        showlegend=True,
    )
    return _base(fig, height=420, title="RL ranking vs MCDM re-ranking")


def mcdm_contributions_figure(mcdm_results: list[MCDMResult]) -> go.Figure:
    """Stacked per-criterion contribution to each candidate's MCDM score."""
    names = [r.name for r in mcdm_results]
    fig = go.Figure()
    for i, crit in enumerate(MCDM_CRITERIA):
        fig.add_trace(go.Bar(
            x=names, y=[r.contributions[crit] for r in mcdm_results],
            name=crit, marker_color=CHART_SEQUENCE[i % len(CHART_SEQUENCE)],
        ))
    fig.update_layout(barmode="stack", xaxis_title="Candidate concept",
                      yaxis_title="Weighted contribution")
    return _base(fig, title="MCDM Score Breakdown by Criterion")


# ---------------------------------------------------------------------------
# 7. Difficulty distribution.
# ---------------------------------------------------------------------------
def difficulty_distribution_figure(df: pd.DataFrame) -> go.Figure:
    """Students' current concept difficulty vs their difficulty preference."""
    diff = df["Concept_Current"].map(lambda n: C.difficulty_of(C.concept_index(n)))
    tmp = pd.DataFrame({"Difficulty": diff, "Preference": df["Difficulty_Preference"]})
    pivot = tmp.groupby(["Difficulty", "Preference"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for i, pref in enumerate(["Easy", "Medium", "Hard"]):
        if pref in pivot.columns:
            fig.add_trace(go.Bar(x=pivot.index, y=pivot[pref], name=pref,
                                 marker_color=CHART_SEQUENCE[i]))
    fig.update_layout(barmode="group", xaxis_title="Concept difficulty (1-5)",
                      yaxis_title="Students")
    return _base(fig, title="Difficulty Distribution by Preference")


# ---------------------------------------------------------------------------
# 8. Satisfaction analysis.
# ---------------------------------------------------------------------------
def satisfaction_figure(df: pd.DataFrame) -> go.Figure:
    """Mean mastery growth by interest level (a satisfaction proxy from data)."""
    grouped = df.groupby("Interest_Level")["Mastery_Growth"].mean()
    fig = go.Figure(go.Bar(x=[f"Interest {i}" for i in grouped.index], y=grouped.values,
                           marker_color=COLORS["primary"],
                           text=[f"{v:.2f}" for v in grouped.values], textposition="outside"))
    fig.update_layout(xaxis_title="Interest level", yaxis_title="Mean mastery growth")
    return _base(fig, title="Satisfaction Proxy: Growth by Interest")


# ---------------------------------------------------------------------------
# 9. Exploration vs exploitation.
# ---------------------------------------------------------------------------
def explore_exploit_figure(history: dict) -> go.Figure:
    """Donut of explore vs exploit action counts over the whole training run."""
    explore = history["explore_count"]
    exploit = history["exploit_count"]
    fig = go.Figure(go.Pie(
        labels=["Exploration", "Exploitation"], values=[explore, exploit], hole=0.6,
        marker=dict(colors=[COLORS["primary_hover"], COLORS["ink_subtle"]]),
        textinfo="label+percent",
    ))
    return _base(fig, height=380, title="Exploration vs Exploitation")


def epsilon_decay_figure(history: dict) -> go.Figure:
    """The epsilon schedule actually used during training."""
    eps = history["epsilon_curve"]
    fig = go.Figure(go.Scatter(y=eps, mode="lines",
                               line=dict(color=COLORS["primary"], width=2.5)))
    fig.update_layout(xaxis_title="Episode", yaxis_title="Epsilon (exploration rate)")
    return _base(fig, title="Epsilon Decay Schedule")
