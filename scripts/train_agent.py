"""
scripts/train_agent.py
----------------------
Minimal training loop showing how to connect a policy to Thermos-RL.

Includes:
  - RandomAgent      : random baseline
  - SimpleRLAgent    : epsilon-greedy Q-table agent (tabular, for demonstration)

Usage
-----
    python scripts/train_agent.py --task burst_management --episodes 200
"""
from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from typing import List, Tuple

import requests

BASE_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# HTTP helpers (no openai dependency needed for training)
# ---------------------------------------------------------------------------
def env_reset(task_id: str, seed: int | None = None) -> dict:
    payload = {"task_id": task_id}
    if seed is not None:
        payload["seed"] = seed
    r = requests.post(f"{BASE_URL}/reset", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()["observation"]


def env_step(delta_freq: float) -> Tuple[dict, float, bool, bool]:
    r = requests.post(
        f"{BASE_URL}/step",
        json={"action": {"delta_freq": delta_freq}},
        timeout=10,
    )
    r.raise_for_status()
    d = r.json()
    return d["observation"], d["reward"]["value"], d["done"], d["truncated"]


def env_grade() -> dict:
    r = requests.get(f"{BASE_URL}/grade", timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
ACTIONS = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]   # Discrete action set


class RandomAgent:
    """Fully random policy — serves as a lower-bound baseline."""
    def select_action(self, obs: dict) -> float:
        return random.choice(ACTIONS)

    def update(self, *args, **kwargs):
        pass


class SimpleRLAgent:
    """
    Tabular Q-learning agent with epsilon-greedy exploration.

    State discretisation
    --------------------
    Bucket cpu_temp into 5-degree bins and cpu_load into 0.2 bins.
    This gives a small enough state space for a tabular demo.
    """
    def __init__(self, lr: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.05):
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.q: defaultdict = defaultdict(lambda: [0.0] * len(ACTIONS))

    def _discretise(self, obs: dict) -> Tuple[int, int, int]:
        temp_bin  = int(obs["cpu_temp"]  // 5)
        load_bin  = int(obs["cpu_load"]  // 0.2)
        freq_bin  = int(obs["curr_freq"] // 0.5)
        return (temp_bin, load_bin, freq_bin)

    def select_action(self, obs: dict) -> float:
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        state  = self._discretise(obs)
        q_vals = self.q[state]
        return ACTIONS[q_vals.index(max(q_vals))]

    def update(self, obs: dict, action: float,
               reward: float, next_obs: dict, done: bool):
        state      = self._discretise(obs)
        next_state = self._discretise(next_obs)
        a_idx      = ACTIONS.index(min(ACTIONS, key=lambda a: abs(a - action)))

        target = reward
        if not done:
            target += self.gamma * max(self.q[next_state])

        self.q[state][a_idx] += self.lr * (target - self.q[state][a_idx])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(task_id: str, episodes: int = 100, agent_type: str = "rl",
          verbose: bool = True) -> List[float]:

    agent = SimpleRLAgent() if agent_type == "rl" else RandomAgent()
    episode_scores: List[float] = []

    for ep in range(1, episodes + 1):
        obs  = env_reset(task_id=task_id, seed=ep)
        done = False
        truncated = False
        ep_reward = 0.0

        while not (done or truncated):
            action              = agent.select_action(obs)
            next_obs, rew, done, truncated = env_step(action)
            agent.update(obs, action, rew, next_obs, done or truncated)
            obs        = next_obs
            ep_reward += rew

        grade = env_grade()
        score = grade["score"]
        episode_scores.append(score)

        if hasattr(agent, "decay_epsilon"):
            agent.decay_epsilon()

        if verbose and (ep % 10 == 0 or ep == 1):
            eps_str = f"  ε={agent.epsilon:.3f}" if hasattr(agent, "epsilon") else ""
            print(f"  Ep {ep:>4}/{episodes}  reward={ep_reward:>8.2f}  "
                  f"score={score:.4f}{eps_str}")

    return episode_scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thermos-RL Training Loop")
    parser.add_argument("--task",     default="idle_stability",
                        choices=["idle_stability", "burst_management", "eco_endurance"])
    parser.add_argument("--episodes", type=int,  default=100)
    parser.add_argument("--agent",    default="rl", choices=["rl", "random"])
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args()

    print(f"\nTraining {args.agent.upper()} agent on task='{args.task}' "
          f"for {args.episodes} episodes ...\n")

    scores = train(
        task_id=args.task,
        episodes=args.episodes,
        agent_type=args.agent,
        verbose=not args.quiet,
    )

    print(f"\n--- Results ---")
    print(f"  Mean score (last 10 episodes): "
          f"{sum(scores[-10:]) / min(10, len(scores)):.4f}")
    print(f"  Best score: {max(scores):.4f}")