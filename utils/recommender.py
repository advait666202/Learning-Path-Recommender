"""
The recommendation pipeline: how RL and MCDM relate.

This module is the glue that turns a learned Q-table plus a student context into
a final recommendation, exactly as specified:

    1. Q-learning produces Q-values for the graph-valid next concepts.
    2. Take the top-k candidates by Q-value (the RL ranking).
    3. MCDM re-ranks those top-k using the user-adjustable criterion weights.
    4. The MCDM winner is the final recommendation.

Both the RL ranking and the MCDM re-ranking are returned so the dashboard can
show the effect of changing weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mcdm.scoring import MCDMResult, rerank
from rl.environment import (
    StudentEnv,
    StudentProfile,
    discretize_mastery,
    encode_state,
)
from utils import concepts as C
from utils.config import TOP_K


@dataclass
class Recommendation:
    state: int
    current_concept: int
    candidates: list[int]                 # all valid next concepts
    rl_ranking: list[tuple[int, float]]   # (concept_id, q_value) best-first
    top_k: list[int]                      # top-k concept ids by Q
    mcdm_results: list[MCDMResult]        # MCDM re-ranking of the top-k
    final_concept: int | None             # the MCDM winner


def profile_from_row(row: pd.Series) -> tuple[StudentProfile, int, float]:
    """Build a (StudentProfile, current_concept_id, mastery) from a dataset row."""
    current_id = C.concept_index(row["Concept_Current"])
    profile = StudentProfile(
        prior_knowledge=float(row["Prior_Knowledge"]),
        study_time_hours=float(row["Study_Time_Hours"]),
        difficulty_preference=str(row["Difficulty_Preference"]),
        learning_speed=str(row["Learning_Speed"]),
        interest_level=int(row["Interest_Level"]),
        preferred_track=C.concept_by_id(current_id)["track"],
    )
    return profile, current_id, float(row["Current_Mastery"])


def _completed_before(concept_id: int) -> set[int]:
    """Topologically-prior concepts, treated as already completed for the state."""
    order = C.topological_order()
    pos = order.index(concept_id)
    return set(order[:pos])


def configure_env(env: StudentEnv, student: StudentProfile, current_concept: int,
                  mastery: float) -> int:
    """Place ``env`` into the (student, current concept, mastery) situation."""
    env.student = student
    env.current_concept = current_concept
    env.mastery = float(mastery)
    env.completed = _completed_before(current_concept)
    env.steps = 0
    return env.state


def rank_by_q(q_table: np.ndarray, state: int, candidates: list[int]) -> list[tuple[int, float]]:
    """Valid candidates sorted by Q-value, descending (ties -> lower concept id)."""
    scored = [(cid, float(q_table[state, cid])) for cid in candidates]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored


def recommend(
    q_table: np.ndarray,
    env: StudentEnv,
    student: StudentProfile,
    current_concept: int,
    mastery: float,
    weights: dict[str, float] | None = None,
    top_k: int = TOP_K,
) -> Recommendation:
    """Run the full RL -> top-k -> MCDM pipeline and return the breakdown."""
    state = configure_env(env, student, current_concept, mastery)
    candidates = env.valid_actions()

    rl_ranking = rank_by_q(q_table, state, candidates)
    top_ids = [cid for cid, _ in rl_ranking[:top_k]]
    mcdm_results = rerank(top_ids, env, weights, student)
    final = mcdm_results[0].concept_id if mcdm_results else None

    return Recommendation(
        state=state,
        current_concept=current_concept,
        candidates=candidates,
        rl_ranking=rl_ranking,
        top_k=top_ids,
        mcdm_results=mcdm_results,
        final_concept=final,
    )


def greedy_policy_fn(q_table: np.ndarray):
    """A ``policy_fn(state, valid_actions) -> action`` that exploits the Q-table."""
    def _fn(state: int, valid_actions: list[int]) -> int:
        scored = [(cid, q_table[state, cid]) for cid in valid_actions]
        return max(scored, key=lambda t: (t[1], -t[0]))[0]
    return _fn
