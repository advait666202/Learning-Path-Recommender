"""
Synthetic student dataset generator.

Produces ``data/students.csv`` with NUM_STUDENTS (>2000) internally-consistent
rows, seeded with RANDOM_SEED for full reproducibility. The concept columns
reference the canonical catalogue in :mod:`utils.concepts`, and a topological
ordering of that DAG is used so every (completed -> current -> next) triple is
prerequisite-valid.

Consistency rules baked into the generative model (so analytics tell a true
story rather than noise):
  * Higher Study_Time_Hours  -> higher Mastery_Growth on average.
  * Higher Current_Mastery    -> higher Quiz_Score and Assignment_Score.
  * Matching Difficulty_Preference to the current concept's difficulty raises
    Mastery_Growth (difficulty suitability).
  * Faster Learning_Speed      -> lower Completion_Time.

Run directly to (re)generate the CSV:
    uv run python utils/data_gen.py
"""

from __future__ import annotations

import os
import sys

# Make the project root importable when run as a script (python utils/data_gen.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from utils import concepts as C
from utils.config import NUM_STUDENTS, RANDOM_SEED, STUDENTS_CSV, DATA_DIR


DIFFICULTY_PREFERENCES = ["Easy", "Medium", "Hard"]
# Map a difficulty preference onto the concept difficulty (1..5) it suits best.
PREF_TO_DIFFICULTY = {"Easy": 1.5, "Medium": 3.0, "Hard": 4.5}
LEARNING_SPEEDS = ["Slow", "Average", "Fast"]
SPEED_FACTOR = {"Slow": 1.4, "Average": 1.0, "Fast": 0.7}


def generate(num_students: int = NUM_STUDENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build the synthetic student DataFrame (does not write to disk)."""
    rng = np.random.default_rng(seed)
    order = C.topological_order()
    objective_name = C.concept_name(C.OBJECTIVE_CONCEPT_ID)

    rows = []
    for i in range(num_students):
        # --- demographics & traits -------------------------------------------------
        age = int(np.clip(rng.normal(24, 6), 16, 55))
        prior_knowledge = float(np.clip(rng.beta(2, 3), 0.0, 1.0))
        study_time = float(np.clip(rng.gamma(shape=2.2, scale=4.0), 0.5, 30.0))
        difficulty_pref = rng.choice(DIFFICULTY_PREFERENCES, p=[0.35, 0.45, 0.20])
        learning_speed = rng.choice(LEARNING_SPEEDS, p=[0.30, 0.45, 0.25])
        interest = int(rng.integers(1, 6))  # 1..5

        # --- position along the learning path -------------------------------------
        # Further progress for students with more prior knowledge + study time.
        progress_drive = 0.55 * prior_knowledge + 0.45 * min(study_time / 20.0, 1.0)
        pos = int(np.clip(rng.binomial(C.NUM_CONCEPTS - 1, np.clip(progress_drive, 0.05, 0.95)),
                          0, C.NUM_CONCEPTS - 1))
        current_id = order[pos]
        completed_id = order[pos - 1] if pos > 0 else None
        next_id = order[pos + 1] if pos + 1 < C.NUM_CONCEPTS else C.OBJECTIVE_CONCEPT_ID

        current_difficulty = C.difficulty_of(current_id)

        # --- mastery & assessment scores ------------------------------------------
        # Current mastery rises with prior knowledge and study time, falls a little
        # with concept difficulty.
        base_mastery = (
            0.12
            + 0.40 * prior_knowledge
            + 0.50 * min(study_time / 20.0, 1.0)
            - 0.06 * (current_difficulty - 3)
            + rng.normal(0, 0.08)
        )
        current_mastery = float(np.clip(base_mastery, 0.02, 0.99))

        quiz = float(np.clip(100 * current_mastery + rng.normal(0, 7), 0, 100))
        assignment = float(np.clip(100 * current_mastery + rng.normal(0, 9), 0, 100))

        # --- difficulty suitability (preference vs concept difficulty) ------------
        pref_target = PREF_TO_DIFFICULTY[str(difficulty_pref)]
        # 1.0 when the concept difficulty matches the student's sweet spot, decaying
        # with the gap. Range ~ [0, 1].
        suitability = float(np.exp(-0.5 * ((current_difficulty - pref_target) / 1.6) ** 2))

        # --- mastery growth (KEY consistency: rises with study time) --------------
        growth = (
            0.10
            + 0.030 * study_time          # study time is the dominant driver
            + 0.18 * suitability          # learning at the right difficulty helps
            + 0.05 * (interest - 3) / 2   # interest nudges growth
            + rng.normal(0, 0.05)
        )
        mastery_growth = float(np.clip(growth, 0.0, 1.0))

        # --- completion time (hours to finish current concept) --------------------
        completion_time = float(np.clip(
            (3 + 1.8 * current_difficulty) * SPEED_FACTOR[str(learning_speed)]
            / (0.4 + mastery_growth) + rng.normal(0, 1.5),
            1.0, 60.0,
        ))

        rows.append({
            "Student_ID": f"S{i + 1:05d}",
            "Age": age,
            "Prior_Knowledge": round(prior_knowledge, 3),
            "Current_Mastery": round(current_mastery, 3),
            "Quiz_Score": round(quiz, 1),
            "Assignment_Score": round(assignment, 1),
            "Study_Time_Hours": round(study_time, 1),
            "Difficulty_Preference": str(difficulty_pref),
            "Learning_Speed": str(learning_speed),
            "Interest_Level": interest,
            "Concept_Completed": C.concept_name(completed_id) if completed_id is not None else "None",
            "Concept_Current": C.concept_name(current_id),
            "Concept_Next": C.concept_name(next_id),
            "Mastery_Growth": round(mastery_growth, 3),
            "Completion_Time": round(completion_time, 1),
            "Learning_Objective": objective_name,
        })

    return pd.DataFrame(rows)


def write_csv(path: str = STUDENTS_CSV) -> pd.DataFrame:
    """Generate and persist the dataset, returning the DataFrame."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df = generate()
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df = write_csv()
    print(f"Wrote {len(df)} students to {STUDENTS_CSV}")
    # Quick consistency check: mean growth should increase across study-time terciles.
    bands = pd.qcut(df["Study_Time_Hours"], 3, labels=["Low", "Medium", "High"])
    print("Mean Mastery_Growth by study-time tercile:")
    print(df.groupby(bands, observed=True)["Mastery_Growth"].mean().round(3).to_string())
