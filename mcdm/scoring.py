"""
Multi-Criteria Decision Making re-ranker.

This is concept **D (Multi-Criteria Decision Making — weighted scoring)** of the
RDMU course, and the second half of the recommendation pipeline:

    1. Q-learning ranks the graph-valid next concepts by Q-value.
    2. The top-k candidates are handed to this module.
    3. ``rerank`` scores each candidate with a weighted sum over five
       user-adjustable criteria and returns them best-first.
    4. The top of that list is the final recommendation.

Each criterion is mapped to a raw value derived from the environment's reward
model (so RL and MCDM share one consistent notion of the student), min-max
normalised across the candidate set, oriented by its ``direction`` (higher- or
lower-is-better), then combined with the normalised criterion weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl.environment import StudentEnv, StudentProfile
from utils import concepts as C
from utils.config import (
    MCDM_CRITERIA,
    MCDM_DEFAULT_WEIGHTS,
    MCDM_DIRECTION,
)


@dataclass
class MCDMResult:
    concept_id: int
    name: str
    score: float
    raw: dict[str, float]            # raw criterion values
    normalized: dict[str, float]     # oriented + normalised criterion values
    contributions: dict[str, float]  # weight * normalised (sums to score)
    rl_rank: int                     # 1-based rank coming in from the RL stage


def _criterion_raw(env: StudentEnv, concept_id: int, student: StudentProfile) -> dict[str, float]:
    """Raw value of each MCDM criterion for ``concept_id`` (pre-normalisation)."""
    return {
        "Mastery Goal": env.completion_speed(concept_id, student),        # progress toward objective
        "Difficulty": float(C.difficulty_of(concept_id)),                 # 1..5
        "Time Available": env.time_efficiency(concept_id, student),       # fit to time budget
        "Student Interest": env.preference_match(concept_id, student),    # interest/track match
        "Expected Learning Gain": env.expected_mastery_gain(concept_id, student),
    }


def _normalize(values: list[float], direction: int) -> list[float]:
    """Min-max normalise to [0,1] and orient by direction (+1 high-good, -1 low-good)."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        norm = [0.5 for _ in values]  # all equal -> neutral
    else:
        norm = [(v - lo) / (hi - lo) for v in values]
    if direction < 0:
        norm = [1.0 - n for n in norm]
    return norm


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, weights.get(c, 0.0)) for c in MCDM_CRITERIA)
    if total <= 1e-12:
        return {c: 1.0 / len(MCDM_CRITERIA) for c in MCDM_CRITERIA}
    return {c: max(0.0, weights.get(c, 0.0)) / total for c in MCDM_CRITERIA}


def rerank(
    candidates: list[int],
    env: StudentEnv,
    weights: dict[str, float] | None = None,
    student: StudentProfile | None = None,
) -> list[MCDMResult]:
    """
    Re-rank ``candidates`` (concept ids, already RL-ranked best-first) by the
    MCDM weighted score. Returns a list of :class:`MCDMResult`, best score first.
    """
    if not candidates:
        return []
    student = student or env.student
    weights = _normalize_weights(weights or MCDM_DEFAULT_WEIGHTS)

    raw_by_concept = {cid: _criterion_raw(env, cid, student) for cid in candidates}

    # Normalise each criterion across the candidate set.
    normalized_by_criterion: dict[str, list[float]] = {}
    for crit in MCDM_CRITERIA:
        col = [raw_by_concept[cid][crit] for cid in candidates]
        normalized_by_criterion[crit] = _normalize(col, MCDM_DIRECTION[crit])

    results: list[MCDMResult] = []
    for idx, cid in enumerate(candidates):
        normalized = {crit: normalized_by_criterion[crit][idx] for crit in MCDM_CRITERIA}
        contributions = {crit: weights[crit] * normalized[crit] for crit in MCDM_CRITERIA}
        score = float(sum(contributions.values()))
        results.append(MCDMResult(
            concept_id=cid,
            name=C.concept_name(cid),
            score=score,
            raw=raw_by_concept[cid],
            normalized=normalized,
            contributions=contributions,
            rl_rank=idx + 1,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def mcdm_optimal_action(
    candidates: list[int],
    env: StudentEnv,
    weights: dict[str, float] | None = None,
    student: StudentProfile | None = None,
) -> int | None:
    """Convenience: the single MCDM-winning concept id (or None if no candidates)."""
    ranked = rerank(candidates, env, weights, student)
    return ranked[0].concept_id if ranked else None
