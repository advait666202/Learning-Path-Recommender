"""
Tabular Q-Learning agent.

This is concept **C (Reinforcement Learning via tabular Q-Learning)** and
concept **B (Exploration vs Exploitation)** of the RDMU course.

The agent holds a dense Q-table of shape ``(num_states, num_concepts)`` and
updates it with the standard temporal-difference rule:

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

Action selection is **epsilon-greedy with action masking**: with probability
epsilon the agent explores a random *valid* concept, otherwise it exploits the
highest-Q *valid* concept. Invalid concepts (prerequisites unmet / already
learned) are never selected. Epsilon is decayed over training so exploration is
high early and low late.
"""

from __future__ import annotations

import numpy as np

from utils import concepts as C
from utils.config import RANDOM_SEED
from utils.config import state_space_size


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        seed: int = RANDOM_SEED,
        num_actions: int = C.NUM_CONCEPTS,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.num_states = state_space_size()
        self.rng = np.random.default_rng(seed)
        # Q-table initialised to zeros (optimistic-free, tabular).
        self.q = np.zeros((self.num_states, self.num_actions), dtype=np.float64)

    # -- action selection -----------------------------------------------------
    def _masked_q(self, state: int, valid_actions: list[int]) -> np.ndarray:
        """Return a Q-row with invalid actions set to -inf."""
        row = np.full(self.num_actions, -np.inf)
        row[valid_actions] = self.q[state, valid_actions]
        return row

    def greedy_action(self, state: int, valid_actions: list[int]) -> int:
        """Exploit: the valid action with the highest Q-value (ties -> lowest id)."""
        masked = self._masked_q(state, valid_actions)
        return int(np.argmax(masked))

    def select_action(self, state: int, valid_actions: list[int]) -> tuple[int, bool]:
        """
        Epsilon-greedy choice among ``valid_actions``.

        Returns ``(action, explored)`` where ``explored`` is True when the action
        was a random exploratory pick (used for the exploration/exploitation
        analytics on the dashboard).
        """
        if not valid_actions:
            raise ValueError("No valid actions available for selection.")
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_actions)), True
        return self.greedy_action(state, valid_actions), False

    # -- learning -------------------------------------------------------------
    def update(self, state: int, action: int, reward: float,
               next_state: int, next_valid: list[int], done: bool) -> float:
        """Apply the TD update and return the TD error (for diagnostics)."""
        if done or not next_valid:
            target = reward
        else:
            future = np.max(self.q[next_state, next_valid])
            target = reward + self.gamma * future
        td_error = target - self.q[state, action]
        self.q[state, action] += self.alpha * td_error
        return float(td_error)

    # -- policy snapshot (for the policy-stability metric) --------------------
    def greedy_policy(self) -> np.ndarray:
        """
        Greedy action per state ignoring masking (raw argmax over the Q-row).

        Used only to measure how much the policy *changes* between checkpoints;
        because the comparison is self-consistent, the unmasked argmax is a fine
        stability proxy and avoids needing per-state valid sets here.
        """
        return np.argmax(self.q, axis=1)
