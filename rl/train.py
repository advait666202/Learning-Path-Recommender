"""
Train the tabular Q-Learning agent against the simulated student environment
and pickle the result to ``models/q_table.pkl``.

The pickle bundles three things the dashboard needs:
    * ``q_table``  : the learned (num_states x num_concepts) array
    * ``metadata`` : hyperparameters, state-space size, timestamps
    * ``history``  : per-episode reward, epsilon schedule, explore/exploit
                     counts, and policy-stability checkpoints (for the
                     convergence and policy-evolution charts)

Run directly:
    uv run python rl/train.py
    uv run python rl/train.py --episodes 8000 --epsilon 0.15
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

# Make the project root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rl.environment import Calibration, StudentEnv
from rl.q_learning import QLearningAgent
from utils.config import (
    EPSILON_DECAY_FRACTION,
    EPSILON_START,
    HYPERPARAMS,
    MODELS_DIR,
    NUM_CHECKPOINTS,
    Q_TABLE_PICKLE,
    RANDOM_SEED,
    state_space_size,
)


def _epsilon_schedule(episode: int, total: int, floor: float) -> float:
    """Linear decay from EPSILON_START to ``floor`` over the first fraction."""
    decay_end = max(1, int(total * EPSILON_DECAY_FRACTION))
    if episode >= decay_end:
        return floor
    frac = episode / decay_end
    return float(EPSILON_START + (floor - EPSILON_START) * frac)


def train(
    episodes: int = HYPERPARAMS["episodes"]["default"],
    alpha: float = HYPERPARAMS["alpha"]["default"],
    gamma: float = HYPERPARAMS["gamma"]["default"],
    epsilon: float = HYPERPARAMS["epsilon"]["default"],
    seed: int = RANDOM_SEED,
    progress_cb=None,
) -> dict:
    """
    Run Q-learning and return a result bundle (q_table, metadata, history).

    ``epsilon`` here is the exploration *floor*: training starts fully
    exploratory (EPSILON_START) and decays linearly to this floor.
    ``progress_cb(frac)`` is an optional callback in [0,1] for a Streamlit
    progress bar.
    """
    env = StudentEnv(calibration=Calibration.from_csv(), seed=seed)
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)

    episode_rewards: list[float] = []
    epsilon_curve: list[float] = []
    explore_count = 0
    exploit_count = 0
    objective_hits = 0

    checkpoint_every = max(1, episodes // NUM_CHECKPOINTS)
    policy_snapshots: list[np.ndarray] = []
    policy_stability: list[float] = []  # % unchanged vs previous checkpoint

    for ep in range(episodes):
        agent.epsilon = _epsilon_schedule(ep, episodes, epsilon)
        epsilon_curve.append(agent.epsilon)

        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            valid = env.valid_actions()
            if not valid:
                break
            action, explored = agent.select_action(state, valid)
            explore_count += int(explored)
            exploit_count += int(not explored)

            next_state, reward, done, info = env.step(action)
            next_valid = env.valid_actions()
            agent.update(state, action, reward, next_state, next_valid, done)

            state = next_state
            total_reward += reward
            if info.get("reached_objective"):
                objective_hits += 1

        episode_rewards.append(total_reward)

        # Policy-stability checkpoint.
        if (ep + 1) % checkpoint_every == 0 or ep == episodes - 1:
            snap = agent.greedy_policy()
            if policy_snapshots:
                unchanged = float(np.mean(snap == policy_snapshots[-1]))
                policy_stability.append(unchanged)
            policy_snapshots.append(snap)
            if progress_cb:
                progress_cb((ep + 1) / episodes)

    history = {
        "episode_rewards": episode_rewards,
        "epsilon_curve": epsilon_curve,
        "explore_count": explore_count,
        "exploit_count": exploit_count,
        "objective_hits": objective_hits,
        "policy_stability": policy_stability,
        "policy_snapshots": [s.tolist() for s in policy_snapshots],
        "checkpoint_every": checkpoint_every,
    }
    metadata = {
        "episodes": episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_floor": epsilon,
        "epsilon_start": EPSILON_START,
        "seed": seed,
        "num_states": state_space_size(),
        "num_actions": agent.num_actions,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "final_policy_stability": policy_stability[-1] if policy_stability else None,
    }
    return {"q_table": agent.q, "metadata": metadata, "history": history}


def save_bundle(bundle: dict, path: str = Q_TABLE_PICKLE) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_bundle(path: str = Q_TABLE_PICKLE) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Q-learning agent.")
    p.add_argument("--episodes", type=int, default=HYPERPARAMS["episodes"]["default"])
    p.add_argument("--alpha", type=float, default=HYPERPARAMS["alpha"]["default"])
    p.add_argument("--gamma", type=float, default=HYPERPARAMS["gamma"]["default"])
    p.add_argument("--epsilon", type=float, default=HYPERPARAMS["epsilon"]["default"])
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(f"State-space size: {state_space_size()} states "
          f"(24 concepts x 4 mastery-bands x 3 study-time-bands).")
    print(f"Training Q-learning for {args.episodes} episodes "
          f"(alpha={args.alpha}, gamma={args.gamma}, epsilon-floor={args.epsilon}) ...")
    t0 = time.time()
    bundle = train(args.episodes, args.alpha, args.gamma, args.epsilon, args.seed)
    save_bundle(bundle)
    hist = bundle["history"]
    rewards = hist["episode_rewards"]
    print(f"Done in {time.time() - t0:.1f}s. Saved to {Q_TABLE_PICKLE}")
    print(f"Mean reward (first 100 eps): {np.mean(rewards[:100]):.3f}")
    print(f"Mean reward (last 100 eps):  {np.mean(rewards[-100:]):.3f}")
    print(f"Explore/exploit: {hist['explore_count']} / {hist['exploit_count']}")
    print(f"Final policy stability: {bundle['metadata']['final_policy_stability']}")
