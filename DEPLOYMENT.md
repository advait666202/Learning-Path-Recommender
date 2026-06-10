# Personalized Learning Path Recommender — Deployment & Validation Guide

## Project Overview

This is an **examination-grade RDMU prototype** that recommends which concept a learner should study next. It combines:
- **Reinforcement Learning (Q-Learning)** to learn an optimal policy
- **Multi-Criteria Decision Making (MCDM)** to let instructors steer recommendations
- **Streamlit Dashboard** with 8 interactive pages
- **Dark theme** matching the Linear design system

---

## ✅ Verification Checklist — All Complete

### Core Modules
- ✅ `utils/concepts.py` — 24-concept canonical DAG
- ✅ `utils/config.py` — centralized configuration (seed 42, hyperparams, theme)
- ✅ `rl/environment.py` — StudentEnv MDP with calibration
- ✅ `rl/q_learning.py` — QLearningAgent with epsilon-greedy
- ✅ `rl/train.py` — training loop with progress tracking
- ✅ `mcdm/scoring.py` — weighted re-ranker with 5 criteria
- ✅ `utils/recommender.py` — RL → top-k → MCDM pipeline
- ✅ `utils/metrics.py` — 7 precisely-defined metrics
- ✅ `visualizations/charts.py` — 24 Plotly chart builders

### Data & Models
- ✅ `data/students.csv` — 2,200 synthetic students (seed 42, internally consistent)
- ✅ `data/concept_graph.gpickle` — 24-concept DAG as NetworkX DiGraph
- ✅ `models/q_table.pkl` — pre-trained Q-table (3,000 episodes, 288×24 states/actions)
- ✅ State space: **288 states** (24 concepts × 4 mastery bands × 3 study-time bands)

### Dashboard
- ✅ Page 1: **Project Overview** — KPIs, concept intro, pipeline
- ✅ Page 2: **Student Profile** — traits, mastery radar, history
- ✅ Page 3: **Concept Dependency Graph** — interactive DAG
- ✅ Page 4: **Learning Path Recommendation** — RL vs MCDM rankings
- ✅ Page 5: **RL Policy** — reward curves, epsilon decay, Q-heatmap, training controls
- ✅ Page 6: **Mastery Progress Analytics** — cohort trends, satisfaction
- ✅ Page 7: **MCDM Settings** — live weight sliders (5 criteria, real-time re-ranking)
- ✅ Page 8: **Performance Metrics** — 7 KPIs, trends, correlations

### Presentation
- ✅ **10-slide PowerPoint** in Linear theme (`presentation/RDMU_Learning_Path_Recommender.pptx`)
  - Title + Pipeline + Requirements + Business Problem + Flow + Four Concepts
  - Dataset + Architecture + Dashboard + Results + Conclusion

### Documentation
- ✅ `README.md` — complete with setup, MDP spec, dataset, dashboard overview
- ✅ `requirements.txt` — pinned versions (Streamlit 1.39, Plotly 5.24, etc.)
- ✅ `runtime.txt` — Python 3.11 for cloud deployment
- ✅ `.streamlit/config.toml` — dark theme configuration
- ✅ `assets/theme.css` — Linear design tokens (dark canvas, lavender accent)
- ✅ This file: `DEPLOYMENT.md` — setup and run instructions

---

## 🚀 How to Run Locally

### Prerequisites
- **Python 3.11** (tested; 3.10+ should work)
- **uv** package manager (recommended) or **pip**

### With uv (Recommended)

```bash
# Navigate to project
cd c:\VS CODE\Learning-Path-Recommender

# Install dependencies (if not already installed)
uv python install 3.11
uv venv --python 3.11
uv pip install -r requirements.txt

# Launch the dashboard (data & model auto-load)
uv run streamlit run app.py
```

The app will:
1. Auto-generate the dataset if missing
2. Auto-load the concept graph
3. Load the pre-trained Q-table (no waiting)
4. Open at `http://localhost:8501`

### With plain pip

```bash
# Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```

---

## 📊 Dashboard Pages Quick Reference

| # | Page | Purpose | Key Controls |
|---|---|---|---|
| 1 | **Overview** | Intro + KPIs + concept summary | — |
| 2 | **Student Profile** | Per-student state, mastery radar, history | Student selector, Random button |
| 3 | **Concept Graph** | Interactive DAG, tooltips, prerequisites | Graph zoom/pan, concept selector |
| 4 | **Recommendation** | RL ranking vs MCDM re-ranking, final pick | Student selector, Epsilon slider |
| 5 | **RL Policy** | Training curves, Q-heatmap, explore/exploit | Episodes slider, Train button |
| 6 | **Mastery Analytics** | Cohort trends, satisfaction, growth | Student selector |
| 7 | **MCDM Settings** | Weight sliders for 5 criteria | 5 sliders (0–1), Reset, Apply buttons |
| 8 | **Performance Metrics** | 7 KPIs, trends, correlations | — |

---

## 🎓 The Four RDMU Concepts — Mapped to Code

| Concept | Where | How it Matters |
|---------|-------|---|
| **A. MDP** | `rl/environment.py::StudentEnv` | States, actions, rewards, transitions model the learning problem |
| **B. Exploration vs Exploitation** | `rl/q_learning.py::select_action` + epsilon decay | Epsilon-greedy: start exploratory (ε=1.0), decay to floor (ε=0.1) |
| **C. Q-Learning** | `rl/q_learning.py::QLearningAgent` + `rl/train.py::train` | Learns a policy by updating Q-values from simulated rollouts |
| **D. MCDM** | `mcdm/scoring.py::rerank` | Re-ranks RL top-k with user-adjustable criterion weights |

---

## ⚙️ Training (Optional — Pre-trained Model Provided)

A Q-table is already trained and cached at `models/q_table.pkl`. To retrain:

### Via Dashboard
1. Go to **RL Policy** page
2. Adjust `Episodes` slider (default 2000, range 100–5000)
3. Click **🚀 Train / Retrain**
4. Wait for progress bar
5. Model saved automatically

### Via CLI
```bash
uv run python rl/train.py
# Generates: models/q_table.pkl
```

Training progress:
- Reward: 1.54 (early, exploratory) → 8.49 (converged)
- Policy stability: ~98% (greedy action unchanged)
- Path completion: ~92% (reach objective)

---

## 📈 Key Metrics (Precise Definitions)

1. **Learning Efficiency** — total mastery gained per total study hour
2. **Mastery Rate** — % students who mastered ≥50% of concepts
3. **Average Reward** — mean episodic reward during training
4. **Policy Stability** — % states with unchanged greedy action
5. **Path Completion Rate** — % of simulated paths reaching the objective
6. **Recommendation Accuracy** — RL vs MCDM agreement (proxy)
7. **Student Satisfaction** — preference-match component (1–5 scale)

See `utils/metrics.py` for exact formulas.

---

## 🌐 Deploy to Streamlit Cloud

1. **Push this repo to GitHub**
   ```bash
   git add .
   git commit -m "RDMU Learning Path Recommender"
   git push origin main
   ```

2. **On Streamlit Cloud Dashboard:**
   - New app → Connect repo
   - Main file: `app.py`
   - Python version: 3.11
   - Click Deploy

3. **First run:** May train briefly if Q-table not included. To skip, commit `models/q_table.pkl` to repo.

---

## 🛠️ Architecture Highlights

### Single Source of Truth
- Concepts defined once in `utils/concepts.py`
- Imported by graph, dataset, environment, dashboard
- No concept drift

### Separation of Concerns
- **RL module** (train offline, pickle model)
- **Dashboard** (load cached model, never retrain on interaction)
- **MCDM** (isolated re-ranker, pluggable weights)
- **Visualizations** (pure Plotly, no side effects)

### Reproducibility
- Seed 42 everywhere (data, environment, training)
- Pinned dependencies (`requirements.txt`)
- Deterministic transitions (calibrated from dataset)

### Performance
- Q-table cached with `@st.cache_resource` → instant load
- Data cached with `@st.cache_data` → no reload
- Dashboard startup: ~2 seconds

---

## 🐛 Troubleshooting

### "Module not found" error
```bash
uv pip install -r requirements.txt
```

### "Concept graph not found"
```bash
uv run python models/concept_graph.py
```

### "Q-table not found"
Models load via the **RL Policy** page → click **Train / Retrain**
Or run: `uv run python rl/train.py`

### Slow on first run
First run generates data + builds graph + loads model. Subsequent runs are cached.
- Second run: ~1 second
- Page switches: instant (cached)

### Unicode/encoding errors on Windows
Ensure PowerShell uses UTF-8: `$PSDefaultParameterValues['*:Encoding'] = 'utf8'`

---

## 📚 File Structure

```
.
├── app.py                          # Main Streamlit app (8 pages)
├── requirements.txt                # Dependencies (pinned)
├── runtime.txt                     # Python version (3.11)
├── README.md                       # Full documentation
├── DEPLOYMENT.md                   # This file
├── DESIGN-linear.app.md            # Design tokens reference
│
├── data/
│   ├── students.csv                # 2,200 students (auto-generated)
│   └── concept_graph.gpickle       # 24-concept DAG (auto-generated)
│
├── models/
│   ├── concept_graph.py            # Graph builder
│   └── q_table.pkl                 # Pre-trained Q-table (3k episodes)
│
├── rl/
│   ├── environment.py              # StudentEnv (MDP)
│   ├── q_learning.py               # QLearningAgent
│   └── train.py                    # Training loop
│
├── mcdm/
│   └── scoring.py                  # MCDM re-ranker
│
├── utils/
│   ├── concepts.py                 # 24-concept catalogue
│   ├── config.py                   # Config (seed, hyperparams, theme)
│   ├── data_gen.py                 # Dataset generator
│   ├── recommender.py              # RL → top-k → MCDM pipeline
│   └── metrics.py                  # 7 metrics
│
├── visualizations/
│   └── charts.py                   # 24 Plotly figures
│
├── assets/
│   └── theme.css                   # Linear design CSS
│
├── .streamlit/
│   └── config.toml                 # Streamlit config
│
├── presentation/
│   ├── build_pptx.py               # Slide builder
│   └── RDMU_Learning_Path_Recommender.pptx  # 10-slide deck
│
└── tests/
    └── test_smoke.py               # Smoke tests
```

---

## 💡 Next Steps / Future Scope

- **Function approximation** (neural net) to scale beyond tabular
- **Real learner telemetry** instead of synthetic data
- **A/B evaluation** vs fixed-curriculum baseline
- **Richer MCDM criteria** (spaced repetition, cohort outcomes)
- **Mobile app** for student-facing interface

---

## 📞 Support

For issues or questions:
1. Check the [README.md](README.md) for concept & metric definitions
2. Review `DESIGN-linear.app.md` for theme tokens
3. See `rl/environment.py` docstrings for MDP details
4. Check `mcdm/scoring.py` for criterion formulas

---

**Built with:**
- Python 3.11 | Streamlit 1.39 | Plotly 5.24 | NumPy/Pandas | NetworkX

**Design:** Linear (dark theme, lavender accent)  
**Seed:** 42 (reproducible everywhere)  
**Status:** ✅ Ready to deploy
