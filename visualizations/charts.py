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
        # Transparent so the dark canvas / card surface shows through (design spec).
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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


# ===========================================================================
# Additions for the design-spec page layouts.
# ===========================================================================
_TRACKS = ["Foundations", "Programming", "CS Theory", "Data/ML"]


def mastery_histogram_figure(df: pd.DataFrame) -> go.Figure:
    """P1: continuous histogram of current mastery across the cohort."""
    fig = go.Figure(go.Histogram(
        x=df["Current_Mastery"], nbinsx=24,
        marker=dict(color=COLORS["primary"], line=dict(color=COLORS["canvas"], width=1)),
    ))
    fig.update_layout(xaxis_title="Current mastery (0–1)", yaxis_title="Students",
                      bargap=0.04)
    return _base(fig, title="Mastery Distribution")


def learning_overview_figure(df: pd.DataFrame) -> go.Figure:
    """P1: mean mastery as the cohort advances through the curriculum."""
    pos = C.topological_position()
    tmp = df.copy()
    tmp["_pos"] = tmp["Concept_Current"].map(lambda n: pos[C.concept_index(n)])
    grouped = tmp.groupby("_pos").agg(
        mastery=("Current_Mastery", "mean"),
        growth=("Mastery_Growth", "mean"),
    ).reset_index().sort_values("_pos")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped["_pos"], y=grouped["mastery"], mode="lines+markers",
                             name="Avg mastery", line=dict(color=COLORS["primary"], width=2.5),
                             marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=grouped["_pos"], y=grouped["growth"], mode="lines",
                             name="Avg growth", line=dict(color=COLORS["primary_hover"],
                                                          width=2, dash="dash")))
    fig.update_layout(xaxis_title="Curriculum position (topological)",
                      yaxis_title="Mastery", legend=dict(orientation="h"))
    return _base(fig, title="Learning Overview")


def _track_profile(row: pd.Series) -> list[float]:
    """Per-track progress (0–1), blended lightly with current mastery."""
    pos = C.topological_position()
    cur_pos = pos[C.concept_index(row["Concept_Current"])]
    cm = float(row["Current_Mastery"])
    out = []
    for t in _TRACKS:
        ids = [c["id"] for c in C.CONCEPTS if c["track"] == t]
        done = sum(1 for i in ids if pos[i] < cur_pos)
        frac = done / len(ids) if ids else 0.0
        out.append(round(min(1.0, frac * (0.5 + 0.5 * cm)), 3))
    return out


def mastery_radar_figure(row: pd.Series) -> go.Figure:
    """P2: a student's mastery profile across the four curriculum tracks."""
    vals = _track_profile(row)
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=_TRACKS + [_TRACKS[0]], fill="toself",
        line=dict(color=COLORS["primary"]),
        fillcolor="rgba(94,106,210,0.25)",
    ))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 1], gridcolor=COLORS["hairline"], tickfont=dict(color=COLORS["ink_subtle"])),
        angularaxis=dict(gridcolor=COLORS["hairline"], tickfont=dict(color=COLORS["ink_muted"])),
    ))
    return _base(fig, height=380, title="Mastery Profile by Track")


def progress_timeline_figure(row: pd.Series) -> go.Figure:
    """P2/P6: the student's path so far with an implied mastery ramp."""
    order = C.topological_order()
    cur_pos = C.topological_position()[C.concept_index(row["Concept_Current"])]
    path_ids = order[:cur_pos + 1]
    names = [C.concept_name(c) for c in path_ids]
    cm = float(row["Current_Mastery"])
    n = len(path_ids)
    ramp = [round(cm * (i + 1) / n, 3) for i in range(n)]  # smooth ramp to current
    fig = go.Figure(go.Scatter(x=list(range(n)), y=ramp, mode="lines+markers",
                               line=dict(color=COLORS["primary"], width=2.5),
                               marker=dict(size=7, color=COLORS["primary_hover"]),
                               text=names, hovertemplate="%{text}<br>mastery %{y:.2f}<extra></extra>"))
    fig.update_layout(xaxis_title="Concepts completed (in order)", yaxis_title="Mastery")
    return _base(fig, title="Progress Timeline")


def epsilon_explore_pie_figure(epsilon: float) -> go.Figure:
    """P4: explore vs exploit split implied by the chosen epsilon."""
    fig = go.Figure(go.Pie(
        labels=["Exploit", "Explore"], values=[1 - epsilon, epsilon], hole=0.6,
        marker=dict(colors=[COLORS["ink_subtle"], COLORS["primary_hover"]]),
        textinfo="label+percent", sort=False,
    ))
    return _base(fig, height=360, title=f"Explore vs Exploit (ε = {epsilon:.2f})")


def learning_speed_area_figure(df: pd.DataFrame) -> go.Figure:
    """P6: mastery-growth distribution as a filled area per learning speed."""
    bins = np.linspace(0, 1, 21)
    centers = (bins[:-1] + bins[1:]) / 2
    fig = go.Figure()
    palette = {
        "Slow": (COLORS["ink_subtle"], "rgba(138,143,152,0.18)"),
        "Average": (COLORS["primary"], "rgba(94,106,210,0.18)"),
        "Fast": (COLORS["primary_hover"], "rgba(130,143,255,0.18)"),
    }
    for speed in ["Slow", "Average", "Fast"]:
        sub = df[df["Learning_Speed"] == speed]["Mastery_Growth"]
        counts, _ = np.histogram(sub, bins=bins)
        line_c, fill_c = palette[speed]
        fig.add_trace(go.Scatter(x=centers, y=counts, mode="lines", name=speed,
                                 line=dict(color=line_c, width=2), fill="tozeroy",
                                 fillcolor=fill_c))
    fig.update_layout(xaxis_title="Mastery growth", yaxis_title="Students",
                      legend=dict(orientation="h"))
    return _base(fig, title="Learning Speed Distribution")


def cohort_comparison_figure(df: pd.DataFrame) -> go.Figure:
    """P6: current-mastery spread per difficulty-preference cohort (box plot)."""
    fig = go.Figure()
    for i, pref in enumerate(["Easy", "Medium", "Hard"]):
        sub = df[df["Difficulty_Preference"] == pref]["Current_Mastery"]
        fig.add_trace(go.Box(y=sub, name=pref, marker_color=CHART_SEQUENCE[i],
                             boxmean=True))
    fig.update_layout(xaxis_title="Difficulty preference", yaxis_title="Current mastery",
                      showlegend=False)
    return _base(fig, title="Cohort Comparison")


def mastery_by_track_figure(df: pd.DataFrame) -> go.Figure:
    """P6: mean current mastery by curriculum track (horizontal bar)."""
    track_of = {c["id"]: c["track"] for c in C.CONCEPTS}
    tmp = df.copy()
    tmp["_track"] = tmp["Concept_Current"].map(lambda n: track_of[C.concept_index(n)])
    grouped = tmp.groupby("_track")["Current_Mastery"].mean().reindex(_TRACKS)
    fig = go.Figure(go.Bar(
        x=grouped.values, y=grouped.index, orientation="h",
        marker_color=CHART_SEQUENCE[:len(grouped)],
        text=[f"{v:.2f}" for v in grouped.values], textposition="outside",
    ))
    fig.update_layout(xaxis_title="Mean current mastery", yaxis_title="Track")
    return _base(fig, title="Mastery by Track")


def mcdm_weight_radar_figure(weights: dict[str, float]) -> go.Figure:
    """P7: the current MCDM weight vector as a radar."""
    total = sum(weights.values()) or 1.0
    vals = [weights[c] / total for c in MCDM_CRITERIA]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=MCDM_CRITERIA + [MCDM_CRITERIA[0]], fill="toself",
        line=dict(color=COLORS["primary"]), fillcolor="rgba(94,106,210,0.25)",
    ))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, max(vals) * 1.2 if vals else 1], gridcolor=COLORS["hairline"],
                        tickfont=dict(color=COLORS["ink_subtle"])),
        angularaxis=dict(gridcolor=COLORS["hairline"], tickfont=dict(color=COLORS["ink_muted"])),
    ))
    return _base(fig, height=400, title="MCDM Weight Distribution")


def correlation_heatmap_figure(df: pd.DataFrame) -> go.Figure:
    """P8: pairwise correlations among the dataset's numeric features."""
    cols = ["Age", "Prior_Knowledge", "Current_Mastery", "Quiz_Score",
            "Assignment_Score", "Study_Time_Hours", "Interest_Level",
            "Mastery_Growth", "Completion_Time"]
    corr = df[cols].corr().round(2)
    short = [c.replace("_", " ") for c in cols]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=short, y=short, zmin=-1, zmax=1,
        colorscale=[[0.0, COLORS["danger"]], [0.5, COLORS["surface_2"]],
                    [1.0, COLORS["primary"]]],
        colorbar=dict(title="r"),
        hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(xaxis=dict(tickangle=-45))
    return _base(fig, height=560, title="Metric Correlations")


def metrics_training_figure(history: dict) -> go.Figure:
    """P8: trackable training signals over episodes (reward, epsilon, objective rate)."""
    rewards = np.asarray(history["episode_rewards"], dtype=float)
    window = max(10, len(rewards) // 50)
    ma = _moving_average(rewards, window)
    norm_reward = (ma - ma.min()) / (ma.ptp() + 1e-9)
    eps = np.asarray(history["epsilon_curve"], dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(window - 1, window - 1 + len(norm_reward)),
                             y=norm_reward, mode="lines", name="Reward (normalised MA)",
                             line=dict(color=COLORS["primary"], width=2.5)))
    fig.add_trace(go.Scatter(y=eps, mode="lines", name="Epsilon",
                             line=dict(color=COLORS["warning"], width=2)))
    fig.update_layout(xaxis_title="Episode", yaxis_title="Normalised signal",
                      legend=dict(orientation="h"))
    return _base(fig, title="Training Signals Over Time")
