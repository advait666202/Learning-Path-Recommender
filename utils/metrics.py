"""
Precise metric definitions for the Performance page (P8).

Every metric here has an unambiguous formula (the prompt explicitly forbids a
hand-wavy "accuracy"). Where a metric needs the agent, it is computed from
greedy roll-outs of the learned Q-table against the simulated environment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mcdm.scoring import mcdm_optimal_action
from rl.environment import (
    StudentEnv,
    discretize_mastery,
    simulate_path,
)
from utils import concepts as C
from utils.config import RANDOM_SEED, TARGET_MASTERY_BAND_INDEX, TOP_K
from utils.recommender import (
    configure_env,
    greedy_policy_fn,
    profile_from_row,
    rank_by_q,
)


# ---------------------------------------------------------------------------
# Dataset-derived metrics.
# ---------------------------------------------------------------------------
def learning_efficiency(df: pd.DataFrame) -> float:
    """Mastery gained per study hour: mean(Mastery_Growth / Study_Time_Hours)."""
    eff = df["Mastery_Growth"] / df["Study_Time_Hours"].clip(lower=0.1)
    return float(eff.mean())


def mastery_rate(df: pd.DataFrame) -> float:
    """% of students whose Current_Mastery reaches at least the target band."""
    bands = df["Current_Mastery"].apply(discretize_mastery)
    return float((bands >= TARGET_MASTERY_BAND_INDEX).mean())


# ---------------------------------------------------------------------------
# Training-history metrics.
# ---------------------------------------------------------------------------
def average_reward(history: dict) -> float:
    """Mean episodic reward across all training episodes."""
    return float(np.mean(history["episode_rewards"])) if history["episode_rewards"] else 0.0

def average_reward_last(history: dict, n: int = 200) -> float:
    """Mean episodic reward over the last ``n`` episodes (converged performance)."""
    rewards = history["episode_rewards"]
    return float(np.mean(rewards[-n:])) if rewards else 0.0


def policy_stability(history: dict) -> float:
    """% of states whose greedy action is unchanged across the last checkpoints."""
    ps = history.get("policy_stability") or []
    return float(ps[-1]) if ps else 0.0


# ---------------------------------------------------------------------------
# Roll-out metrics (greedy Q-policy vs the environment / MCDM).
# ---------------------------------------------------------------------------
def path_completion_rate(q_table: np.ndarray, env: StudentEnv, n_paths: int = 300,
                         seed: int = RANDOM_SEED) -> float:
    """% of greedy simulated paths that reach the learning objective."""
    policy = greedy_policy_fn(q_table)
    env.rng = np.random.default_rng(seed)
    reached = 0
    for _ in range(n_paths):
        result = simulate_path(env, policy)
        reached += int(result["reached_objective"])
    return reached / n_paths


def _sample_rows(df: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    """Sample dataset rows so metrics span real students at their actual concepts."""
    n = min(n_samples, len(df))
    return df.sample(n, random_state=seed)


def recommendation_accuracy(q_table: np.ndarray, env: StudentEnv, df: pd.DataFrame,
                            n_samples: int = 300, seed: int = RANDOM_SEED) -> float:
    """
    Proxy accuracy: % agreement between the greedy RL action and the
    MCDM-optimal action (default weights) over the same valid candidate set,
    measured across real students placed at their current concept.

    There is no external ground truth for the "correct" next concept, so this
    measures how often the two decision rules already concur.
    """
    agree = 0
    counted = 0
    for _, row in _sample_rows(df, n_samples, seed).iterrows():
        student, current, mastery = profile_from_row(row)
        state = configure_env(env, student, current, mastery)
        valid = env.valid_actions()
        if not valid:
            continue
        ranked = rank_by_q(q_table, state, valid)
        rl_action = ranked[0][0]
        top_ids = [cid for cid, _ in ranked[:TOP_K]]
        mcdm_action = mcdm_optimal_action(top_ids, env)
        counted += 1
        agree += int(rl_action == mcdm_action)
    return agree / counted if counted else 0.0


def student_satisfaction(q_table: np.ndarray, env: StudentEnv, df: pd.DataFrame,
                         n_samples: int = 300, seed: int = RANDOM_SEED) -> float:
    """
    Mean preference-match component of the greedy RL recommendation across real
    sampled students (the reward model's "satisfaction" signal aggregated).
    """
    vals = []
    for _, row in _sample_rows(df, n_samples, seed).iterrows():
        student, current, mastery = profile_from_row(row)
        state = configure_env(env, student, current, mastery)
        valid = env.valid_actions()
        if not valid:
            continue
        rl_action = rank_by_q(q_table, state, valid)[0][0]
        vals.append(env.preference_match(rl_action, student))
    return float(np.mean(vals)) if vals else 0.0


def compute_all(df: pd.DataFrame, bundle: dict, env: StudentEnv,
                n_rollouts: int = 300) -> dict[str, float]:
    """Compute the full KPI set for the Performance page."""
    q_table = bundle["q_table"]
    history = bundle["history"]
    return {
        "Learning Efficiency": learning_efficiency(df),
        "Mastery Rate": mastery_rate(df),
        "Average Reward": average_reward(history),
        "Average Reward (last 200)": average_reward_last(history),
        "Policy Stability": policy_stability(history),
        "Path Completion Rate": path_completion_rate(q_table, env, n_rollouts),
        "Recommendation Accuracy": recommendation_accuracy(q_table, env, df, n_rollouts),
        "Student Satisfaction": student_satisfaction(q_table, env, df, n_rollouts),
    }
