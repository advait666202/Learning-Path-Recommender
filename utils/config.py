"""
Central configuration for the RDMU prototype.

Everything that another module might want to tune lives here: the global random
seed, the state-discretisation band edges, the MDP reward weights, the
Q-learning hyperparameters (defaults + UI slider ranges), the MCDM criteria
weights, the canonical filesystem paths, and the Linear design tokens used by
the Streamlit CSS and the Plotly template.

Keeping these as plain module-level constants (rather than a config file) means
the env, the trainer and the dashboard all import the *same* numbers, which is
exactly what an examination-grade reproducibility story needs.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Reproducibility.
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Filesystem paths (all relative to the project root).
# ---------------------------------------------------------------------------
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")
ASSETS_DIR: str = os.path.join(PROJECT_ROOT, "assets")

STUDENTS_CSV: str = os.path.join(DATA_DIR, "students.csv")
GRAPH_PICKLE: str = os.path.join(DATA_DIR, "concept_graph.gpickle")
Q_TABLE_PICKLE: str = os.path.join(MODELS_DIR, "q_table.pkl")
THEME_CSS: str = os.path.join(ASSETS_DIR, "theme.css")

NUM_STUDENTS: int = 2200  # > 2000 as required

# ---------------------------------------------------------------------------
# State discretisation.
#
# State = (current_concept_index, mastery_band, study_time_band)
#   mastery_band      : 4 bins over a 0..1 mastery score
#   study_time_band   : 3 bins over weekly study hours
# ---------------------------------------------------------------------------
MASTERY_BANDS: list[str] = ["Novice", "Developing", "Proficient", "Mastered"]
# Upper-edge thresholds on a 0..1 mastery score. mastery <= edge -> that band.
MASTERY_BAND_EDGES: list[float] = [0.25, 0.50, 0.75, 1.01]
NUM_MASTERY_BANDS: int = len(MASTERY_BANDS)

STUDY_TIME_BANDS: list[str] = ["Low", "Medium", "High"]
# Upper-edge thresholds on weekly study hours. hours <= edge -> that band.
STUDY_TIME_BAND_EDGES: list[float] = [5.0, 12.0, 1e9]
NUM_STUDY_TIME_BANDS: int = len(STUDY_TIME_BANDS)

# The mastery band a student must reach for a concept to count as "mastered"
# in the Mastery-Rate and Path-Completion metrics.
TARGET_MASTERY_BAND_INDEX: int = MASTERY_BANDS.index("Proficient")  # >= Proficient

# Derived: total number of discrete states.
# (filled in lazily to avoid importing concepts at module load if not needed)
def state_space_size() -> int:
    from utils.concepts import NUM_CONCEPTS

    return NUM_CONCEPTS * NUM_MASTERY_BANDS * NUM_STUDY_TIME_BANDS


# ---------------------------------------------------------------------------
# MDP reward weights (sum to 1.0). Exposed so the trainer/dashboard agree.
# ---------------------------------------------------------------------------
REWARD_WEIGHTS: dict[str, float] = {
    "mastery_gain": 0.30,
    "completion_speed": 0.20,
    "preference_match": 0.20,
    "difficulty_suitability": 0.20,
    "time_efficiency": 0.10,
}

# ---------------------------------------------------------------------------
# Q-learning hyperparameters: defaults + UI slider ranges (min, max, step).
# ---------------------------------------------------------------------------
# Per the design spec, only epsilon and episodes are user-adjustable; alpha and
# gamma are held constant.
HYPERPARAMS = {
    "epsilon": {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01},
    "alpha":   {"default": 0.10, "min": 0.01, "max": 1.0, "step": 0.01},
    "gamma":   {"default": 0.90, "min": 0.50, "max": 0.999, "step": 0.001},
    "episodes": {"default": 2000, "min": 100, "max": 5000, "step": 100},
}

# Epsilon is decayed linearly from EPSILON_START to the chosen floor across
# training so the agent explores early and exploits late (Exploration vs
# Exploitation). The floor defaults to the configured epsilon.
EPSILON_START: float = 1.0
EPSILON_DECAY_FRACTION: float = 0.6  # reach the floor after 60% of episodes

# Number of training checkpoints used for the policy-stability metric.
NUM_CHECKPOINTS: int = 20
# Max steps (concept transitions) per training/simulation episode.
# The objective (Deep Learning) has 19 required ancestors, so the minimum path
# is 20 steps; a tight budget of 22 leaves only 2 steps of slack. A policy that
# wanders into the 4 off-path concepts overruns the budget and fails to reach
# the objective, whereas an efficient learned policy succeeds — this is what
# gives Q-learning a real, visible optimisation signal.
MAX_EPISODE_STEPS: int = 22

# Reward shaping that turns the curriculum into a real optimisation problem:
#   * each step carries a time cost, so wandering through off-path concepts hurts;
#   * reaching the objective pays a terminal bonus.
# A learned policy therefore reaches the objective via well-suited concepts and
# fewer wasted steps than a random one, producing a visibly rising learning curve.
REWARD_STEP_COST: float = 0.6      # subtracted from every step's component reward
REWARD_OBJECTIVE_BONUS: float = 10.0  # paid once, on reaching the objective concept

# ---------------------------------------------------------------------------
# MCDM criteria. Default weights are equal (0.20 each) and user-adjustable.
# "direction" = +1 means higher raw value is better, -1 means lower is better
# (applied before normalisation in mcdm/scoring.py).
# ---------------------------------------------------------------------------
MCDM_CRITERIA: list[str] = [
    "Mastery Goal",
    "Difficulty",
    "Time Available",
    "Student Interest",
    "Expected Learning Gain",
]
# Non-equal defaults per the design spec (user-adjustable on the MCDM page).
MCDM_DEFAULT_WEIGHTS: dict[str, float] = {
    "Mastery Goal": 0.30,
    "Difficulty": 0.20,
    "Time Available": 0.20,
    "Student Interest": 0.15,
    "Expected Learning Gain": 0.15,
}
MCDM_DIRECTION: dict[str, int] = {
    "Mastery Goal": +1,            # concepts that close the gap to the goal score higher
    "Difficulty": -1,             # gentler difficulty preferred (re-weightable)
    "Time Available": +1,          # better fit to available time scores higher
    "Student Interest": +1,        # higher interest match scores higher
    "Expected Learning Gain": +1,  # higher expected mastery gain scores higher
}

# Number of top RL candidates handed to the MCDM re-ranker.
TOP_K: int = 5

# ---------------------------------------------------------------------------
# Linear design tokens (subset used by CSS + Plotly). Mirrors
# DESIGN-linear.app.md so the dashboard speaks the same visual language.
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#5e6ad2",
    "primary_hover": "#828fff",
    "on_primary": "#ffffff",
    "ink": "#f7f8f8",
    "ink_muted": "#d0d6e0",
    "ink_subtle": "#8a8f98",
    "ink_tertiary": "#62666d",
    "canvas": "#010102",
    "surface_1": "#0f1011",
    "surface_2": "#141516",
    "surface_3": "#18191a",
    "hairline": "#23252a",
    "hairline_strong": "#34343a",
    "success": "#27a644",
    "warning": "#e5a50a",
    "danger": "#e5484d",
}

# Ordered categorical palette for charts — the design spec's colorway:
# lavender hero + hover, then the three semantic colors and a neutral.
CHART_SEQUENCE: list[str] = [
    COLORS["primary"],        # lavender
    COLORS["primary_hover"],  # lavender hover
    COLORS["success"],        # green
    COLORS["warning"],        # amber
    COLORS["danger"],         # red
    COLORS["ink_subtle"],     # neutral grey
]
