# Personalized Learning Path Recommender (RDMU)

An examination-grade prototype that recommends **which concept a learner should
study next**. A tabular **Q-Learning** agent is trained against a simulated
student modelled as a **Markov Decision Process**, using **epsilon-greedy**
exploration; a **Multi-Criteria Decision Making** re-ranker then lets an
instructor steer the final recommendation with interpretable weights.

The UI is a Streamlit dashboard themed after the Linear design system (dark
canvas, lavender accent — see `DESIGN-linear.app.md`).

---

## The four RDMU concepts and where each lives

| Concept | Where in the code |
|---|---|
| **A · Markov Decision Process** | `rl/environment.py` — `StudentEnv` (state, action, reward, transition) |
| **B · Exploration vs Exploitation** | `rl/q_learning.py` — `select_action` (epsilon-greedy) + epsilon decay in `rl/train.py` |
| **C · Tabular Q-Learning** | `rl/q_learning.py` (`QLearningAgent`) trained by `rl/train.py` |
| **D · Multi-Criteria Decision Making** | `mcdm/scoring.py` — `rerank` (weighted scoring) |

The pipeline that ties RL and MCDM together is `utils/recommender.py`:
Q-values → top-k by Q → MCDM re-rank → MCDM winner is the final recommendation.

---

## MDP specification

**State** `S = (current_concept_index, mastery_band, study_time_band)`
* `mastery_band ∈ {Novice, Developing, Proficient, Mastered}` (4 bins)
* `study_time_band ∈ {Low, Medium, High}` (3 bins)

**State-space size = 24 concepts × 4 mastery bands × 3 study-time bands = 288 states.**
The action space is the 24 concepts, masked per state to the graph-valid
candidates (prerequisites satisfied, not already completed).

**Reward** = weighted sum (weights in `utils/config.REWARD_WEIGHTS`):

| Component | Weight |
|---|---|
| mastery_gain | 0.30 |
| completion_speed | 0.20 |
| preference_match | 0.20 |
| difficulty_suitability | 0.20 |
| time_efficiency | 0.10 |

**Transition** — after studying a concept, mastery moves up a band with a
probability that rises with difficulty suitability and study time; that
probability is **calibrated from the dataset** (mean `Mastery_Growth` per
study-time band, `rl/environment.Calibration`).

**Default hyperparameters** — epsilon floor `0.1` (decays from `1.0`), learning
rate `α = 0.1`, discount `γ = 0.9`, `2000` episodes. Per the design spec, only
**epsilon and episodes** are user-adjustable (episodes slider `100–5000`); α and
γ are held constant. The *RL Policy* page and the `rl/train.py` CLI expose them.

**MCDM criteria** (user-adjustable, defaults from the design spec): Mastery Goal
`0.30`, Difficulty `0.20`, Time Available `0.20`, Student Interest `0.15`,
Expected Learning Gain `0.15`.

---

## Dataset

`utils/data_gen.py` generates a synthetic, internally-consistent cohort of
**2,200 students** (seed 42) to `data/students.csv`. Higher study time yields
higher mastery growth on average; quiz/assignment scores track current mastery;
the (completed → current → next) concept triples are prerequisite-valid.

Columns: `Student_ID, Age, Prior_Knowledge, Current_Mastery, Quiz_Score,
Assignment_Score, Study_Time_Hours, Difficulty_Preference, Learning_Speed,
Interest_Level, Concept_Completed, Concept_Current, Concept_Next,
Mastery_Growth, Completion_Time, Learning_Objective`.

---

## Setup & run

This machine uses **[uv](https://docs.astral.sh/uv/)** (the working Python
toolchain here). A plain-`pip` fallback is given below.

### With uv (recommended)

```bash
uv python install 3.11
uv venv --python 3.11
uv pip install -r requirements.txt

# 1) generate the dataset
uv run python utils/data_gen.py
# 2) build the concept graph
uv run python models/concept_graph.py
# 3) pre-train the Q-table  -> models/q_table.pkl
uv run python rl/train.py
# 4) launch the dashboard
uv run streamlit run app.py
```

### With plain pip

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python utils/data_gen.py
python models/concept_graph.py
python rl/train.py
streamlit run app.py
```

> First run convenience: if you skip steps 1–2, the app generates the dataset
> and graph automatically on launch. The Q-table is **not** auto-trained — use
> the *Train / Retrain* button on the **RL Policy** page, or run `rl/train.py`.
> The app loads the pickled Q-table via `st.cache_resource` and never retrains
> on ordinary interactions.

---

## Dashboard (8 pages)

1. **Project Overview** — concept summary, KPIs, pipeline.
2. **Student Profile** — per-student traits + position in the curriculum.
3. **Concept Dependency Graph** — interactive DAG with tooltips.
4. **Learning Path Recommendation** — RL ranking vs MCDM re-ranking, final pick.
5. **RL Policy** — reward/learning curves, epsilon decay, explore/exploit,
   policy stability, Q-value heatmap, and training controls.
6. **Mastery Progress Analytics** — mastery bands, growth-vs-study, satisfaction.
7. **MCDM Decision Settings** — live weight sliders that re-rank in real time.
8. **Performance Metrics** — the KPI set below.

### Metrics (precise definitions)

* **Learning Efficiency** — total mastery gained per total study hour invested
  across the cohort (`sum(Mastery_Growth) / sum(Study_Time_Hours)`).
* **Mastery Rate** — % of students who have mastered at least 50% of the
  concepts. The dataset stores one current concept per student; since advancing
  past a concept implies mastering it, the mastered fraction is proxied by the
  student's topological position over the curriculum (≥ 0.5 counts).
* **Average Reward** — mean episodic reward during training.
* **Policy Stability** — % of states whose greedy action is unchanged across the
  last training checkpoints.
* **Path Completion Rate** — % of greedy simulated paths reaching the objective.
* **Recommendation Accuracy** — % agreement between the greedy RL action and the
  MCDM-optimal action (a stated proxy; no external ground truth exists).
* **Student Satisfaction** — preference-match reward component, reported on a
  1–5 scale.

---

## Project structure

```
app.py                  # 8-page Streamlit dashboard
.streamlit/config.toml  # Linear dark theme
assets/theme.css        # injected Linear-style CSS
data/                   # students.csv, concept_graph.gpickle (generated)
models/                 # concept_graph.py, q_table.pkl (generated)
rl/                     # environment.py, q_learning.py, train.py
mcdm/                   # scoring.py
utils/                  # concepts.py, config.py, data_gen.py, recommender.py, metrics.py
visualizations/         # charts.py (Plotly)
```

---

## Deployment

Deployable to **Streamlit Community Cloud**: push the repo, set the main file to
`app.py`, and pin Python 3.11 (`runtime.txt` with `python-3.11` or the app
settings). Commit a pre-trained `models/q_table.pkl` so the cloud app starts
instantly; otherwise train once via the *RL Policy* page. `requirements.txt`
already pins every dependency.

Reproducibility: `RANDOM_SEED = 42` is used across data generation, the
environment, and training (`utils/config.py`).
