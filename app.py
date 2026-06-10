"""
Personalized Learning Path Recommender — Streamlit dashboard.

An examination-grade RDMU prototype demonstrating four concepts:
  A. Markov Decision Processes        -> rl/environment.py (StudentEnv)
  B. Exploration vs Exploitation      -> rl/q_learning.py  (epsilon-greedy)
  C. Tabular Q-Learning               -> rl/q_learning.py + rl/train.py
  D. Multi-Criteria Decision Making   -> mcdm/scoring.py   (weighted scoring)

The recommendation pipeline (utils/recommender.py) is the spine: Q-learning
ranks valid next concepts -> top-k by Q-value -> MCDM re-ranks with user weights
-> MCDM winner is the final recommendation.

Run:  uv run streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from models.concept_graph import load_graph
from rl.environment import (
    Calibration,
    StudentEnv,
    StudentProfile,
    MASTERY_BANDS,
    STUDY_TIME_BANDS,
    discretize_mastery,
    discretize_study_time,
)
from rl.train import load_bundle, save_bundle, train
from utils import concepts as C
from utils.config import (
    GRAPH_PICKLE,
    HYPERPARAMS,
    MCDM_CRITERIA,
    MCDM_DEFAULT_WEIGHTS,
    Q_TABLE_PICKLE,
    REWARD_WEIGHTS,
    STUDENTS_CSV,
    THEME_CSS,
    TOP_K,
    state_space_size,
)
from utils.data_gen import write_csv
from utils.recommender import profile_from_row, recommend
from visualizations import charts

# ---------------------------------------------------------------------------
# Page config + theme.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Learning Path Recommender — RDMU",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_css() -> None:
    if os.path.exists(THEME_CSS):
        with open(THEME_CSS, "r", encoding="utf-8") as fh:
            st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)


_inject_css()


# ---------------------------------------------------------------------------
# Cached loaders / bootstrap.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Generating student dataset…")
def load_students() -> pd.DataFrame:
    if not os.path.exists(STUDENTS_CSV):
        write_csv()
    return pd.read_csv(STUDENTS_CSV)


@st.cache_resource(show_spinner="Building concept graph…")
def get_graph():
    return load_graph(GRAPH_PICKLE)


@st.cache_resource(show_spinner=False)
def get_calibration() -> Calibration:
    return Calibration.from_csv()


def new_env() -> StudentEnv:
    """A fresh environment (cheap) — avoids cross-page mutation of a shared one."""
    return StudentEnv(calibration=get_calibration())


@st.cache_resource(show_spinner=False)
def get_bundle():
    if not os.path.exists(Q_TABLE_PICKLE):
        return None
    return load_bundle(Q_TABLE_PICKLE)


@st.cache_data(show_spinner="Computing performance metrics…")
def cached_metrics(trained_at: str) -> dict:
    """Metrics keyed on the model's training timestamp so retrains invalidate it."""
    from utils.metrics import compute_all

    df = pd.read_csv(STUDENTS_CSV)
    bundle = load_bundle(Q_TABLE_PICKLE)
    return compute_all(df, bundle, new_env(), n_rollouts=300)


# ---------------------------------------------------------------------------
# Small HTML helpers (KPI cards, badges) — match the Linear card spec.
# ---------------------------------------------------------------------------
def kpi_card(label: str, value: str, sub: str = "", accent: bool = False) -> str:
    value_cls = "kpi-value kpi-accent" if accent else "kpi-value"
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (f'<div class="linear-card kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="{value_cls}">{value}</div>{sub_html}</div>')


def render_kpis(items: list[tuple], per_row: int = 4) -> None:
    """items: list of (label, value, sub[, accent])."""
    for start in range(0, len(items), per_row):
        chunk = items[start:start + per_row]
        cols = st.columns(len(chunk))
        for col, item in zip(cols, chunk):
            label, value = item[0], item[1]
            sub = item[2] if len(item) > 2 else ""
            accent = item[3] if len(item) > 3 else False
            col.markdown(kpi_card(label, value, sub, accent), unsafe_allow_html=True)


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def ensure_weights() -> dict[str, float]:
    if "mcdm_weights" not in st.session_state:
        st.session_state["mcdm_weights"] = dict(MCDM_DEFAULT_WEIGHTS)
    return st.session_state["mcdm_weights"]


def model_ready() -> bool:
    return get_bundle() is not None


def _active_learner_count(df: pd.DataFrame) -> int:
    """Students mid-curriculum: past the first concept, not yet at the objective."""
    first = C.concept_name(0)
    objective = C.concept_name(C.OBJECTIVE_CONCEPT_ID)
    return int((~df["Concept_Current"].isin([first, objective])).sum())


def training_status_badge() -> str:
    bundle = get_bundle()
    if bundle is None:
        return '<span class="badge">Model: not trained</span>'
    meta = bundle["metadata"]
    return (f'<span class="badge badge-success">Model trained</span>'
            f'<span class="badge">{meta["episodes"]} episodes</span>'
            f'<span class="badge">{meta["trained_at"]}</span>')


def run_training(episodes: int, alpha: float, gamma: float, epsilon: float) -> None:
    """Train and persist, then clear caches so the app picks up the new model."""
    progress = st.progress(0.0, text="Training Q-learning agent…")
    bundle = train(episodes=episodes, alpha=alpha, gamma=gamma, epsilon=epsilon,
                   progress_cb=lambda f: progress.progress(min(1.0, f),
                                                           text=f"Training… {int(f*100)}%"))
    save_bundle(bundle)
    progress.empty()
    get_bundle.clear()
    cached_metrics.clear()
    st.success("Training complete — model updated.")


# ===========================================================================
# PAGES
# ===========================================================================
def page_overview(df: pd.DataFrame) -> None:
    eyebrow("RDMU · Reinforcement Learning")
    st.title("Personalized Learning Path Recommender")
    st.markdown(
        '<div style="color:#d0d6e0;font-size:18px;max-width:760px;line-height:1.5">'
        "An adaptive tutor that learns <b>which concept a student should study next</b>. "
        "A Q-learning agent is trained against a simulated student (an MDP), and a "
        "Multi-Criteria re-ranker lets an instructor steer the recommendation with "
        "interpretable weights.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f'<div style="margin:14px 0">{training_status_badge()}</div>',
                unsafe_allow_html=True)

    st.markdown("###")
    active = _active_learner_count(df)
    completion = "—"
    if model_ready():
        completion = f"{cached_metrics(get_bundle()['metadata']['trained_at'])['Path Completion Rate']*100:.0f}%"
    render_kpis([
        ("Total students", f"{len(df):,}", "synthetic, seed 42"),
        ("Active learners", f"{active:,}", "mid-curriculum"),
        ("Avg mastery", f"{df['Current_Mastery'].mean():.2f}", "across cohort"),
        ("Path completion", completion, "greedy → objective", True),
    ])

    st.markdown("## Learning overview")
    oc1, oc2 = st.columns(2)
    oc1.plotly_chart(charts.mastery_histogram_figure(df), use_container_width=True)
    oc2.plotly_chart(charts.learning_overview_figure(df), use_container_width=True)

    st.markdown("## The four RDMU concepts")
    c1, c2 = st.columns(2)
    cards = [
        ("A · Markov Decision Process",
         "State = (concept, mastery band, study-time band). Action = next concept. "
         "Reward = weighted sum of 5 learning signals. Lives in <code>rl/environment.py</code>."),
        ("B · Exploration vs Exploitation",
         "Epsilon-greedy action selection with linear epsilon decay — explore early, "
         "exploit late. Lives in <code>rl/q_learning.py</code>."),
        ("C · Tabular Q-Learning",
         "A 288×24 Q-table learned by temporal-difference updates over many simulated "
         "students. Trained in <code>rl/train.py</code>."),
        ("D · Multi-Criteria Decision Making",
         "Weighted scoring re-ranks the top-k RL candidates over 5 instructor-tunable "
         "criteria. Lives in <code>mcdm/scoring.py</code>."),
    ]
    for i, (title, body) in enumerate(cards):
        col = c1 if i % 2 == 0 else c2
        col.markdown(
            f'<div class="linear-card" style="margin-bottom:16px"><b style="font-size:17px">{title}</b>'
            f'<div style="color:#d0d6e0;margin-top:8px;font-size:14px;line-height:1.55">{body}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("## Recommendation pipeline")
    p1, p2, p3, p4 = st.columns(4)
    steps = [
        ("1 · Q-values", "Score every graph-valid next concept with the learned Q-table."),
        ("2 · Top-k", f"Keep the top {TOP_K} candidates by Q-value (the RL ranking)."),
        ("3 · MCDM re-rank", "Re-score the shortlist with the instructor's criterion weights."),
        ("4 · Final pick", "The MCDM winner is the recommended next concept."),
    ]
    for col, (t, b) in zip([p1, p2, p3, p4], steps):
        col.markdown(
            f'<div class="linear-card"><div class="eyebrow">{t}</div>'
            f'<div style="color:#d0d6e0;margin-top:8px;font-size:13px;line-height:1.5">{b}</div></div>',
            unsafe_allow_html=True,
        )


def _student_selector(df: pd.DataFrame, key: str) -> pd.Series:
    """Shared student picker; remembers selection in session_state."""
    ids = df["Student_ID"].tolist()
    default = st.session_state.get("student_id", ids[0])
    if default not in ids:
        default = ids[0]
    chosen = st.selectbox("Student", ids, index=ids.index(default), key=key)
    st.session_state["student_id"] = chosen
    return df[df["Student_ID"] == chosen].iloc[0]


def page_student_profile(df: pd.DataFrame) -> None:
    eyebrow("Student")
    st.title("Student Profile")
    left, right = st.columns([1, 2])
    with left:
        row = _student_selector(df, key="profile_select")
        if st.button("🎲 Random student", key="rand_profile"):
            st.session_state["student_id"] = df.sample(1).iloc[0]["Student_ID"]
            st.rerun()

    current_id = C.concept_index(row["Concept_Current"])
    m_band = MASTERY_BANDS[discretize_mastery(row["Current_Mastery"])]
    s_band = STUDY_TIME_BANDS[discretize_study_time(row["Study_Time_Hours"])]

    with right:
        render_kpis([
            ("Current concept", row["Concept_Current"], f"difficulty {C.difficulty_of(current_id)}/5", True),
            ("Mastery band", m_band, f"score {row['Current_Mastery']:.2f}"),
            ("Study-time band", s_band, f"{row['Study_Time_Hours']:.1f} h/week"),
        ], per_row=3)

    st.markdown("### Traits & assessment")
    render_kpis([
        ("Age", f"{int(row['Age'])}"),
        ("Prior knowledge", f"{row['Prior_Knowledge']:.2f}"),
        ("Quiz score", f"{row['Quiz_Score']:.0f}"),
        ("Assignment score", f"{row['Assignment_Score']:.0f}"),
        ("Difficulty preference", row["Difficulty_Preference"]),
        ("Learning speed", row["Learning_Speed"]),
        ("Interest level", f"{int(row['Interest_Level'])}/5"),
        ("Mastery growth", f"{row['Mastery_Growth']:.2f}"),
    ])

    st.markdown("### Learning profile")
    pc1, pc2 = st.columns(2)
    pc1.plotly_chart(charts.mastery_radar_figure(row), use_container_width=True)
    pc2.plotly_chart(charts.progress_timeline_figure(row), use_container_width=True)

    st.markdown("### Concept completion history")
    order = C.topological_order()
    cur_pos = C.topological_position()[current_id]
    history_rows = []
    for cid in order[:cur_pos]:
        history_rows.append({"Concept": C.concept_name(cid),
                             "Track": C.concept_by_id(cid)["track"],
                             "Difficulty": f"{C.difficulty_of(cid)}/5",
                             "Status": "Completed"})
    history_rows.append({"Concept": row["Concept_Current"],
                         "Track": C.concept_by_id(current_id)["track"],
                         "Difficulty": f"{C.difficulty_of(current_id)}/5",
                         "Status": f"In progress · {m_band}"})
    st.dataframe(pd.DataFrame(history_rows), hide_index=True, use_container_width=True,
                 height=280)


def page_concept_graph(df: pd.DataFrame) -> None:
    eyebrow("Curriculum")
    st.title("Concept Dependency Graph")
    st.markdown(
        '<div style="color:#d0d6e0;max-width:720px">A directed acyclic graph of 24 '
        "concepts. An edge means the source is a prerequisite of the target. Hover a "
        "node for difficulty, track, prerequisites and the average mastery observed in "
        "the dataset.</div>", unsafe_allow_html=True)

    graph = get_graph()
    names = C.all_names()
    highlight_name = st.selectbox("Highlight a concept", ["(none)"] + names)
    highlight = None if highlight_name == "(none)" else C.concept_index(highlight_name)
    st.plotly_chart(charts.concept_graph_figure(graph, highlight=highlight),
                    use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.difficulty_distribution_figure(df), use_container_width=True)
    with right:
        if highlight is not None:
            data = graph.nodes[highlight]
            prereqs = ", ".join(C.concept_name(p) for p in data["prerequisites"]) or "none"
            unlocks = [C.concept_name(s) for s in graph.successors(highlight)] or ["—"]
            st.markdown(
                f'<div class="linear-card"><b style="font-size:18px">{data["name"]}</b>'
                f'<div style="margin-top:10px;color:#d0d6e0;font-size:14px;line-height:1.7">'
                f'Difficulty: <b>{data["difficulty"]}/5</b><br>Track: <b>{data["track"]}</b><br>'
                f'Prerequisites: {prereqs}<br>Unlocks: {", ".join(unlocks)}<br>'
                f'Avg mastery (dataset): {data["avg_mastery"]:.2f}</div></div>',
                unsafe_allow_html=True)
        else:
            st.info("Select a concept above to see its prerequisites and what it unlocks.")


def _recommendation_confidence(rec) -> float:
    """Softmax probability mass on the top RL candidate — a confidence proxy."""
    import numpy as np

    qs = np.array([q for _, q in rec.rl_ranking], dtype=float)
    if qs.size == 0:
        return 0.0
    if qs.size == 1:
        return 1.0
    e = np.exp(qs - qs.max())
    return float((e / e.sum()).max())


def _render_recommendation(rec, weights) -> None:
    if rec.final_concept is None:
        st.warning("This student has already completed all available concepts — "
                   "no valid next step.")
        return
    final = rec.final_concept
    rl_top = rec.rl_ranking[0][0] if rec.rl_ranking else None
    changed = (rl_top is not None and rl_top != final)
    st.markdown(
        f'<div class="reco-hero"><div class="eyebrow">Recommended next concept</div>'
        f'<div class="reco-name">{C.concept_name(final)}</div>'
        f'<div class="reco-meta">Difficulty {C.difficulty_of(final)}/5 · '
        f'{"MCDM overrode the RL top pick" if changed else "RL and MCDM agree"}</div></div>',
        unsafe_allow_html=True)

    st.markdown("####")
    left, right = st.columns(2)
    with left:
        st.markdown("**RL ranking** (by Q-value)")
        rl_df = pd.DataFrame([
            {"#": i + 1, "Concept": C.concept_name(cid), "Q-value": round(q, 4)}
            for i, (cid, q) in enumerate(rec.rl_ranking[:TOP_K])
        ])
        st.dataframe(rl_df, hide_index=True, use_container_width=True)
    with right:
        st.markdown("**MCDM re-ranking** (by weighted score)")
        mcdm_df = pd.DataFrame([
            {"#": i + 1, "Concept": r.name, "MCDM score": round(r.score, 4),
             "RL rank": r.rl_rank}
            for i, r in enumerate(rec.mcdm_results)
        ])
        st.dataframe(mcdm_df, hide_index=True, use_container_width=True)

    st.plotly_chart(charts.rl_vs_mcdm_figure(rec.rl_ranking, rec.mcdm_results),
                    use_container_width=True)
    st.plotly_chart(charts.mcdm_contributions_figure(rec.mcdm_results),
                    use_container_width=True)


def page_recommendation(df: pd.DataFrame) -> None:
    eyebrow("Decision")
    st.title("Learning Path Recommendation")
    if not model_ready():
        st.warning("The Q-table has not been trained yet. Go to **RL Policy** and click "
                   "*Train / Retrain*, or run `uv run python rl/train.py`.")
        return

    weights = ensure_weights()
    sel_col, eps_col = st.columns([2, 2])
    with sel_col:
        row = _student_selector(df, key="reco_select")
    with eps_col:
        epsilon = st.slider("Exploration rate ε", 0.0, 1.0,
                            HYPERPARAMS["epsilon"]["default"], 0.01, key="reco_epsilon")
    student, current_id, mastery = profile_from_row(row)

    st.markdown(
        f'<div style="color:#8a8f98;font-size:13px">Current: '
        f'<b style="color:#f7f8f8">{row["Concept_Current"]}</b> · mastery '
        f'{mastery:.2f} · study {row["Study_Time_Hours"]:.1f} h/wk · '
        f'interest {int(row["Interest_Level"])}/5 · '
        f'prefers {row["Difficulty_Preference"]}</div>', unsafe_allow_html=True)
    st.caption("Adjust the MCDM weights on the **MCDM Settings** page to steer the final pick.")

    bundle = get_bundle()
    rec = recommend(bundle["q_table"], new_env(), student, current_id, mastery, weights)

    # 3 KPIs: recommended next concept, exploration %, confidence (Q-softmax margin).
    confidence = _recommendation_confidence(rec)
    next_name = C.concept_name(rec.final_concept) if rec.final_concept is not None else "—"
    render_kpis([
        ("Recommended next concept", next_name, "MCDM winner", True),
        ("Exploration", f"{epsilon*100:.0f}%", f"exploit {100-epsilon*100:.0f}%"),
        ("Confidence", f"{confidence*100:.0f}%", "Q-value margin"),
    ], per_row=3)

    _render_recommendation(rec, weights)

    bl, br = st.columns([2, 3])
    bl.plotly_chart(charts.epsilon_explore_pie_figure(epsilon), use_container_width=True)
    with br:
        st.markdown("### Where it leads")
        st.plotly_chart(
            charts.concept_graph_figure(get_graph(), highlight=rec.final_concept),
            use_container_width=True)


def page_rl_policy(df: pd.DataFrame) -> None:
    eyebrow("Agent")
    st.title("RL Policy & Training")

    with st.expander("⚙️ Train / Retrain the Q-learning agent", expanded=not model_ready()):
        alpha = HYPERPARAMS["alpha"]["default"]
        gamma = HYPERPARAMS["gamma"]["default"]
        c1, c2 = st.columns(2)
        episodes = c1.slider("Episodes", HYPERPARAMS["episodes"]["min"],
                             HYPERPARAMS["episodes"]["max"], HYPERPARAMS["episodes"]["default"],
                             HYPERPARAMS["episodes"]["step"])
        epsilon = c2.slider("Epsilon floor ε", HYPERPARAMS["epsilon"]["min"],
                            HYPERPARAMS["epsilon"]["max"], HYPERPARAMS["epsilon"]["default"],
                            HYPERPARAMS["epsilon"]["step"])
        st.caption(f"Learning rate α = {alpha} and discount γ = {gamma} are held constant "
                   "(per the design spec). Epsilon starts at 1.0 (full exploration) and "
                   "decays linearly to the floor over the first 60% of episodes.")
        if st.button("🚀 Train / Retrain", type="primary"):
            run_training(episodes, alpha, gamma, epsilon)
            st.rerun()

    if not model_ready():
        st.info("No trained model yet — train above to populate this page.")
        return

    bundle = get_bundle()
    meta, hist = bundle["metadata"], bundle["history"]
    render_kpis([
        ("Episodes", f"{meta['episodes']:,}"),
        ("α / γ", f"{meta['alpha']} / {meta['gamma']}"),
        ("Epsilon floor", f"{meta['epsilon_floor']}"),
        ("Final policy stability", f"{(meta['final_policy_stability'] or 0)*100:.0f}%", "", True),
    ])

    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.reward_curve_figure(hist), use_container_width=True)
    c2.plotly_chart(charts.learning_curve_figure(hist), use_container_width=True)
    c3, c4 = st.columns(2)
    c3.plotly_chart(charts.epsilon_decay_figure(hist), use_container_width=True)
    c4.plotly_chart(charts.explore_exploit_figure(hist), use_container_width=True)

    st.plotly_chart(charts.policy_evolution_figure(hist), use_container_width=True)
    st.plotly_chart(charts.q_heatmap_figure(bundle["q_table"]), use_container_width=True)


def page_mastery_analytics(df: pd.DataFrame) -> None:
    eyebrow("Analytics")
    st.title("Mastery Progress Analytics")
    render_kpis([
        ("Mean current mastery", f"{df['Current_Mastery'].mean():.2f}"),
        ("Mean mastery growth", f"{df['Mastery_Growth'].mean():.2f}"),
        ("Mean study time", f"{df['Study_Time_Hours'].mean():.1f} h"),
        ("Mean completion time", f"{df['Completion_Time'].mean():.1f} h"),
    ])

    st.markdown("### Mastery over time")
    row = _student_selector(df, key="analytics_select")
    st.plotly_chart(charts.progress_timeline_figure(row), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.mastery_by_track_figure(df), use_container_width=True)
    c2.plotly_chart(charts.cohort_comparison_figure(df), use_container_width=True)
    c3, c4 = st.columns(2)
    c3.plotly_chart(charts.learning_speed_area_figure(df), use_container_width=True)
    c4.plotly_chart(charts.satisfaction_figure(df), use_container_width=True)


def page_mcdm_settings(df: pd.DataFrame) -> None:
    eyebrow("Multi-Criteria Decision Making")
    st.title("MCDM Decision Settings")
    st.markdown(
        '<div style="color:#d0d6e0;max-width:760px">Set how much each criterion matters. '
        "Weights are normalised, then used to re-rank the RL shortlist live. Watch the "
        "final recommendation change as you move the sliders.</div>",
        unsafe_allow_html=True)

    weights = ensure_weights()
    cols = st.columns(len(MCDM_CRITERIA))
    new_weights = {}
    for col, crit in zip(cols, MCDM_CRITERIA):
        new_weights[crit] = col.slider(crit, 0.0, 1.0, float(weights[crit]), 0.05,
                                       key=f"w_{crit}")
    st.session_state["mcdm_weights"] = new_weights

    cc1, cc2 = st.columns([1, 3])
    if cc1.button("↺ Reset to default weights"):
        st.session_state["mcdm_weights"] = dict(MCDM_DEFAULT_WEIGHTS)
        st.rerun()
    total = sum(new_weights.values())
    cc2.markdown(
        f'<div style="padding-top:8px;color:#8a8f98">Normalised: ' +
        " · ".join(f"{c} {(w/total if total else 0):.0%}" for c, w in new_weights.items()) +
        "</div>", unsafe_allow_html=True)

    wl, wr = st.columns([2, 3])
    wl.plotly_chart(charts.mcdm_weight_radar_figure(new_weights), use_container_width=True)
    wr.markdown(
        '<div class="linear-card" style="margin-top:20px"><b>How it works</b>'
        '<div style="color:#d0d6e0;margin-top:8px;font-size:14px;line-height:1.6">'
        'The RL agent shortlists the top-k next concepts by Q-value. Each criterion '
        'above is scored per candidate, min-max normalised, oriented (e.g. lower '
        'difficulty scores higher), then combined with these weights. The highest '
        'weighted score becomes the final recommendation below.</div></div>',
        unsafe_allow_html=True)

    st.divider()
    if not model_ready():
        st.warning("Train the model (RL Policy page) to see live re-ranking.")
        return

    row = _student_selector(df, key="mcdm_select")
    student, current_id, mastery = profile_from_row(row)
    rec = recommend(get_bundle()["q_table"], new_env(), student, current_id, mastery,
                    new_weights)
    _render_recommendation(rec, new_weights)


def page_metrics(df: pd.DataFrame) -> None:
    eyebrow("Evaluation")
    st.title("Performance Metrics")
    if not model_ready():
        st.warning("Train the model (RL Policy page) to compute metrics.")
        return

    bundle = get_bundle()
    metrics = cached_metrics(bundle["metadata"]["trained_at"])

    render_kpis([
        ("Learning efficiency", f"{metrics['Learning Efficiency']:.3f}", "mastery / study hour"),
        ("Mastery rate", f"{metrics['Mastery Rate']*100:.0f}%", "≥ Proficient band"),
        ("Average reward", f"{metrics['Average Reward']:.2f}", "mean episodic"),
        ("Reward (last 200)", f"{metrics['Average Reward (last 200)']:.2f}", "converged"),
    ])
    render_kpis([
        ("Policy stability", f"{metrics['Policy Stability']*100:.0f}%", "last checkpoints", True),
        ("Path completion", f"{metrics['Path Completion Rate']*100:.0f}%", "reach objective"),
        ("Reco. accuracy", f"{metrics['Recommendation Accuracy']*100:.0f}%", "RL≡MCDM proxy"),
        ("Satisfaction", f"{metrics['Student Satisfaction']:.2f} / 5", "preference match"),
    ])

    with st.expander("📖 Metric definitions"):
        st.markdown(
            "- **Learning Efficiency** — mastery gained per study hour "
            "(mean of Mastery_Growth ÷ Study_Time_Hours).\n"
            "- **Mastery Rate** — % of students at or above the Proficient mastery band.\n"
            "- **Average Reward** — mean episodic reward during training.\n"
            "- **Policy Stability** — % of states whose greedy action is unchanged across "
            "the last training checkpoints.\n"
            "- **Path Completion Rate** — % of greedy simulated paths that reach the "
            "objective (Deep Learning).\n"
            "- **Recommendation Accuracy** — % agreement between the greedy RL action and "
            "the MCDM-optimal action (a proxy; there is no external ground truth).\n"
            "- **Student Satisfaction** — preference-match reward component, averaged over "
            "recommendations.")

    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.metrics_training_figure(bundle["history"]), use_container_width=True)
    c2.plotly_chart(charts.policy_evolution_figure(bundle["history"]), use_container_width=True)

    st.plotly_chart(charts.correlation_heatmap_figure(df), use_container_width=True)

    st.markdown("### Detailed student metrics")
    detail = df[["Student_ID", "Concept_Current", "Current_Mastery", "Mastery_Growth",
                 "Study_Time_Hours", "Completion_Time", "Interest_Level"]].copy()
    detail["Efficiency"] = (detail["Mastery_Growth"] /
                            detail["Study_Time_Hours"].clip(lower=0.1)).round(3)
    st.dataframe(detail, hide_index=True, use_container_width=True, height=360)


# ===========================================================================
# Router
# ===========================================================================
PAGES = {
    "Project Overview": page_overview,
    "Student Profile": page_student_profile,
    "Concept Dependency Graph": page_concept_graph,
    "Learning Path Recommendation": page_recommendation,
    "RL Policy": page_rl_policy,
    "Mastery Progress Analytics": page_mastery_analytics,
    "MCDM Decision Settings": page_mcdm_settings,
    "Performance Metrics": page_metrics,
}


def main() -> None:
    df = load_students()

    with st.sidebar:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
            '<div style="width:22px;height:22px;border-radius:6px;background:#5e6ad2"></div>'
            '<b style="font-size:16px;letter-spacing:-0.4px">Path Recommender</b></div>',
            unsafe_allow_html=True)
        st.caption("RDMU · MDP · Q-Learning · MCDM")
        st.markdown("---")
        choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.markdown("---")
        st.markdown(
            '<div style="font-size:11px;color:#62666d;line-height:1.6">Seed 42 · '
            f'{state_space_size()} states · {C.NUM_CONCEPTS} concepts<br>'
            'Pre-train: <code>uv run python rl/train.py</code></div>',
            unsafe_allow_html=True)

    PAGES[choice](df)


if __name__ == "__main__":
    main()
