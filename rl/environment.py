"""
The Markov Decision Process: a simulated student environment.

This is concept **A (Markov Decision Processes)** of the RDMU course made
concrete. Q-learning trains *online against this environment*, never directly on
the CSV. The CSV is used only to *calibrate* the transition probabilities (via
:class:`Calibration`) and to supply student profiles for the dashboard.

MDP definition
--------------
State  S = (current_concept_index, mastery_band, study_time_band)
       encoded to a single integer in [0, state_space_size).
         * mastery_band     in {Novice, Developing, Proficient, Mastered}  (4)
         * study_time_band  in {Low, Medium, High}                          (3)
       study_time_band is a property of the *student* and is therefore fixed
       within an episode; mastery_band evolves as the student studies.

Action A = choose the next concept to study, restricted to the graph-valid
       candidates (all prerequisites completed, not already completed).

Reward R = weighted sum of five components, each normalised to ~[0, 1]:
       mastery_gain, difficulty_suitability, preference_match, time_efficiency,
       completion_speed. Weights live in utils.config.REWARD_WEIGHTS.

Transition T : after choosing concept ``a`` the student studies it; their
       mastery moves probabilistically toward a higher band, with the
       probability of moving up a band increasing in difficulty suitability and
       study time (calibrated from dataset Mastery_Growth aggregates).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils import concepts as C
from utils.config import (
    MASTERY_BAND_EDGES,
    MASTERY_BANDS,
    MAX_EPISODE_STEPS,
    NUM_MASTERY_BANDS,
    NUM_STUDY_TIME_BANDS,
    RANDOM_SEED,
    REWARD_OBJECTIVE_BONUS,
    REWARD_STEP_COST,
    REWARD_WEIGHTS,
    STUDENTS_CSV,
    STUDY_TIME_BAND_EDGES,
    STUDY_TIME_BANDS,
    TARGET_MASTERY_BAND_INDEX,
)

# Same suitability mapping the dataset generator uses, so the env's notion of
# "the right difficulty for this student" matches the data it was calibrated on.
PREF_TO_DIFFICULTY = {"Easy": 1.5, "Medium": 3.0, "Hard": 4.5}
DIFFICULTY_PREFERENCES = ["Easy", "Medium", "Hard"]
LEARNING_SPEEDS = ["Slow", "Average", "Fast"]
SPEED_FACTOR = {"Slow": 1.4, "Average": 1.0, "Fast": 0.7}


# ---------------------------------------------------------------------------
# State (de)serialisation.
# ---------------------------------------------------------------------------
def discretize_mastery(mastery: float) -> int:
    """Map a 0..1 mastery score to a band index (0..3)."""
    for i, edge in enumerate(MASTERY_BAND_EDGES):
        if mastery <= edge:
            return i
    return NUM_MASTERY_BANDS - 1


def discretize_study_time(hours: float) -> int:
    """Map weekly study hours to a band index (0..2)."""
    for i, edge in enumerate(STUDY_TIME_BAND_EDGES):
        if hours <= edge:
            return i
    return NUM_STUDY_TIME_BANDS - 1


def encode_state(concept_idx: int, mastery_band: int, study_band: int) -> int:
    """Flatten (concept, mastery_band, study_band) into a single state index."""
    return (concept_idx * NUM_MASTERY_BANDS + mastery_band) * NUM_STUDY_TIME_BANDS + study_band


def decode_state(state: int) -> tuple[int, int, int]:
    """Inverse of :func:`encode_state`."""
    study_band = state % NUM_STUDY_TIME_BANDS
    rest = state // NUM_STUDY_TIME_BANDS
    mastery_band = rest % NUM_MASTERY_BANDS
    concept_idx = rest // NUM_MASTERY_BANDS
    return concept_idx, mastery_band, study_band


# ---------------------------------------------------------------------------
# Student profile (the "context" the reward/transition depend on).
# ---------------------------------------------------------------------------
@dataclass
class StudentProfile:
    prior_knowledge: float
    study_time_hours: float
    difficulty_preference: str
    learning_speed: str
    interest_level: int          # 1..5
    preferred_track: str

    @property
    def study_band(self) -> int:
        return discretize_study_time(self.study_time_hours)

    @property
    def pref_difficulty(self) -> float:
        return PREF_TO_DIFFICULTY[self.difficulty_preference]


# ---------------------------------------------------------------------------
# Calibration of the transition model from dataset aggregates.
# ---------------------------------------------------------------------------
@dataclass
class Calibration:
    """Per study-time-band multiplier on the probability of a mastery up-move."""

    growth_by_band: dict[int, float]  # band index -> mean Mastery_Growth (0..1)

    @classmethod
    def from_csv(cls, path: str = STUDENTS_CSV) -> "Calibration":
        if not os.path.exists(path):
            # Reasonable monotone defaults if the dataset has not been generated.
            return cls(growth_by_band={0: 0.30, 1: 0.45, 2: 0.62})
        df = pd.read_csv(path)
        bands = df["Study_Time_Hours"].apply(discretize_study_time)
        means = df.assign(_band=bands).groupby("_band")["Mastery_Growth"].mean()
        growth = {int(b): float(means.get(b, 0.4)) for b in range(NUM_STUDY_TIME_BANDS)}
        return cls(growth_by_band=growth)

    def band_multiplier(self, study_band: int) -> float:
        return self.growth_by_band.get(study_band, 0.4)


# ---------------------------------------------------------------------------
# The environment.
# ---------------------------------------------------------------------------
class StudentEnv:
    """A Gym-style MDP over the concept graph for a single simulated student."""

    def __init__(
        self,
        calibration: Calibration | None = None,
        reward_weights: dict[str, float] | None = None,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.calibration = calibration or Calibration.from_csv()
        self.reward_weights = reward_weights or dict(REWARD_WEIGHTS)
        self.rng = np.random.default_rng(seed)
        self.topo_pos = C.topological_position()
        self.max_pos = max(self.topo_pos.values())

        # Episode state (set in reset).
        self.student: StudentProfile | None = None
        self.current_concept: int = 0
        self.mastery: float = 0.0
        self.completed: set[int] = set()
        self.steps: int = 0

    # -- student sampling -----------------------------------------------------
    def sample_student(self) -> StudentProfile:
        rng = self.rng
        return StudentProfile(
            prior_knowledge=float(np.clip(rng.beta(2, 3), 0, 1)),
            study_time_hours=float(np.clip(rng.gamma(2.2, 4.0), 0.5, 30.0)),
            difficulty_preference=str(rng.choice(DIFFICULTY_PREFERENCES, p=[0.35, 0.45, 0.20])),
            learning_speed=str(rng.choice(LEARNING_SPEEDS, p=[0.30, 0.45, 0.25])),
            interest_level=int(rng.integers(1, 6)),
            preferred_track=str(rng.choice(["Foundations", "Programming", "CS Theory", "Data/ML"])),
        )

    # -- lifecycle ------------------------------------------------------------
    def reset(self, student: StudentProfile | None = None) -> int:
        self.student = student or self.sample_student()
        self.current_concept = 0  # everyone starts at "Python Basics"
        self.mastery = float(np.clip(0.15 + 0.4 * self.student.prior_knowledge, 0, 0.95))
        self.completed = set()
        self.steps = 0
        return self.state

    @property
    def state(self) -> int:
        return encode_state(
            self.current_concept,
            discretize_mastery(self.mastery),
            self.student.study_band,
        )

    def valid_actions(self, completed: set[int] | None = None) -> list[int]:
        """Graph-valid next concepts. The current concept counts as completed."""
        done_set = set(self.completed if completed is None else completed)
        done_set.add(self.current_concept)
        return C.valid_next_concepts(done_set)

    # -- reward model ---------------------------------------------------------
    def difficulty_suitability(self, concept_id: int, student: StudentProfile) -> float:
        gap = C.difficulty_of(concept_id) - student.pref_difficulty
        return float(np.exp(-0.5 * (gap / 1.6) ** 2))

    def prob_mastery_up(self, concept_id: int, student: StudentProfile) -> float:
        """Probability the mastery band moves up after studying ``concept_id``."""
        suit = self.difficulty_suitability(concept_id, student)
        band_mult = self.calibration.band_multiplier(student.study_band)
        p = 0.15 + 0.45 * suit + 0.45 * band_mult
        return float(np.clip(p, 0.05, 0.95))

    def expected_mastery_gain(self, concept_id: int, student: StudentProfile) -> float:
        """Expected normalised mastery improvement (reused by MCDM)."""
        return self.prob_mastery_up(concept_id, student)

    def preference_match(self, concept_id: int, student: StudentProfile) -> float:
        track = C.concept_by_id(concept_id)["track"]
        base = student.interest_level / 5.0
        track_bonus = 0.2 if track == student.preferred_track else 0.0
        return float(np.clip(0.6 * base + track_bonus + 0.2, 0, 1))

    def time_efficiency(self, concept_id: int, student: StudentProfile) -> float:
        diff = C.difficulty_of(concept_id)
        growth = max(0.05, self.expected_mastery_gain(concept_id, student))
        hours = (3 + 1.8 * diff) * SPEED_FACTOR[student.learning_speed] / (0.4 + growth)
        return float(np.clip(1 - hours / 30.0, 0, 1))

    def completion_speed(self, concept_id: int, student: StudentProfile) -> float:
        """Progress toward the objective via topological position (0..1)."""
        return float(self.topo_pos[concept_id] / self.max_pos)

    def reward_components(self, concept_id: int, student: StudentProfile | None = None) -> dict[str, float]:
        """The five normalised reward components for moving to ``concept_id``."""
        student = student or self.student
        return {
            "mastery_gain": self.expected_mastery_gain(concept_id, student),
            "difficulty_suitability": self.difficulty_suitability(concept_id, student),
            "preference_match": self.preference_match(concept_id, student),
            "time_efficiency": self.time_efficiency(concept_id, student),
            "completion_speed": self.completion_speed(concept_id, student),
        }

    def reward(self, concept_id: int, student: StudentProfile | None = None) -> float:
        comps = self.reward_components(concept_id, student)
        return float(sum(self.reward_weights[k] * v for k, v in comps.items()))

    # -- transition -----------------------------------------------------------
    def step(self, action: int) -> tuple[int, float, bool, dict]:
        """Apply ``action`` (move to concept ``action``) and advance the MDP."""
        if self.student is None:
            raise RuntimeError("Call reset() before step().")
        valid = self.valid_actions()
        if action not in valid:
            # Illegal action: small penalty, state unchanged, terminate episode.
            return self.state, -1.0, True, {"illegal": True, "valid_actions": valid}

        student = self.student
        # Instantaneous "learning value" of the choice (weighted components, ~0..1)
        # minus the per-step time cost. The objective bonus is added below.
        quality = self.reward(action, student)
        reward = quality - REWARD_STEP_COST

        # Mark the concept we are leaving as completed; move to the new concept.
        self.completed.add(self.current_concept)
        self.current_concept = action
        self.steps += 1

        # Transition: simulate studying the new concept. Mastery starts modest and
        # moves up a band with the calibrated probability.
        p_up = self.prob_mastery_up(action, student)
        base = 0.20 + 0.10 * student.prior_knowledge
        if self.rng.random() < p_up:
            base += 0.30 + 0.20 * self.rng.random()  # a clear up-move
        else:
            base += 0.05 * self.rng.random()
        self.mastery = float(np.clip(base + self.rng.normal(0, 0.05), 0.02, 0.99))

        # Reaching the objective concept ends the episode and pays the bonus.
        reached_objective = action == C.OBJECTIVE_CONCEPT_ID
        if reached_objective:
            reward += REWARD_OBJECTIVE_BONUS

        mastered = discretize_mastery(self.mastery) >= TARGET_MASTERY_BAND_INDEX
        no_more_actions = len(self.valid_actions()) == 0
        out_of_steps = self.steps >= MAX_EPISODE_STEPS
        done = bool(reached_objective or no_more_actions or out_of_steps)

        info = {
            "reached_objective": reached_objective,
            "mastered_current": mastered,
            "p_up": p_up,
            "quality": quality,
            "components": self.reward_components(action, student),
        }
        return self.state, reward, done, info


def simulate_path(env: StudentEnv, policy_fn, student: StudentProfile | None = None,
                  max_steps: int = MAX_EPISODE_STEPS) -> dict:
    """
    Roll out a full path under a greedy ``policy_fn(state, valid_actions) -> action``.

    Returns a summary used by the Path-Completion and analytics metrics:
    the concept sequence, total reward, and whether the objective was reached.
    """
    state = env.reset(student)
    path = [env.current_concept]
    total_reward = 0.0
    reached = False
    for _ in range(max_steps):
        valid = env.valid_actions()
        if not valid:
            break
        action = policy_fn(state, valid)
        state, reward, done, info = env.step(action)
        path.append(env.current_concept)
        total_reward += reward
        if info.get("reached_objective"):
            reached = True
        if done:
            break
    return {
        "path": path,
        "path_names": [C.concept_name(c) for c in path],
        "total_reward": total_reward,
        "reached_objective": reached,
        "length": len(path),
    }
