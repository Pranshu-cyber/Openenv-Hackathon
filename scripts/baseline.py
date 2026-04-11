"""
scripts/baseline.py
--------------------
Baseline PID-controller agent for Thermos-RL.

Uses session_id from /reset to stay isolated from other parallel runs.

Usage
-----
    python scripts/baseline.py --task idle_stability --seed 42
    python scripts/baseline.py --task burst_management
    python scripts/baseline.py --task eco_endurance --quiet
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

import requests

BASE_URL = "http://localhost:8000"

TASK_TARGETS = {
    "idle_stability":   40.0,
    "burst_management": 70.0,
    "eco_endurance":    65.0,
}


# ---------------------------------------------------------------------------
# PID Controller
# ---------------------------------------------------------------------------
class PIDController:
    def __init__(self, kp: float, ki: float, kd: float,
                 output_min: float = -1.0, output_max: float = 1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = output_min, output_max
        self._integral   = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0

    def compute(self, setpoint: float, measured: float) -> float:
        error            = measured - setpoint
        self._integral  += error
        derivative       = error - self._prev_error
        self._prev_error = error
        output = -((self.kp * error) + (self.ki * self._integral) + (self.kd * derivative))
        return max(self.out_min, min(self.out_max, output))


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def post(endpoint: str, payload: dict, session_id: Optional[str] = None) -> dict:
    url = f"{BASE_URL}{endpoint}"
    if session_id:
        url += f"?session_id={session_id}"
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def get(endpoint: str, session_id: Optional[str] = None) -> dict:
    url = f"{BASE_URL}{endpoint}"
    if session_id:
        url += f"?session_id={session_id}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------
def run_baseline(task_id: str, seed: Optional[int] = None, verbose: bool = True) -> dict:
    t_target = TASK_TARGETS.get(task_id, 65.0)
    pid = PIDController(kp=0.05, ki=0.002, kd=0.01)
    pid.reset()

    # Reset — get session_id for isolation
    reset_payload = {"task_id": task_id}
    if seed is not None:
        reset_payload["seed"] = seed

    resp       = post("/reset", reset_payload)
    session_id = resp["info"]["session_id"]   # ← unique session for this run
    obs        = resp["observation"]
    total_rew  = 0.0
    step_n     = 0

    if verbose:
        print(f"\n{'='*62}")
        print(f"  Task : {task_id}  |  Target T : {t_target}°C  |  Seed : {seed}")
        print(f"  Session: {session_id[:8]}...")
        print(f"{'='*62}")
        print(f"{'Step':>5}  {'Temp':>7}  {'Freq':>6}  {'Load':>6}  "
              f"{'Power':>7}  {'Bat%':>6}  {'Reward':>8}")
        print(f"{'-'*62}")

    done = truncated = False

    while not (done or truncated):
        delta_freq = pid.compute(setpoint=t_target, measured=obs["cpu_temp"])

        # Pass session_id so this step goes to the right env instance
        resp      = post("/step", {"action": {"delta_freq": delta_freq}}, session_id)
        obs       = resp["observation"]
        reward    = resp["reward"]
        done      = resp["done"]
        truncated = resp["truncated"]

        total_rew += reward["value"]
        step_n    += 1

        if verbose and (step_n % 50 == 0 or done or truncated):
            print(f"{step_n:>5}  "
                  f"{obs['cpu_temp']:>7.2f}  "
                  f"{obs['curr_freq']:>6.3f}  "
                  f"{obs['cpu_load']:>6.3f}  "
                  f"{obs['power_draw']:>7.2f}  "
                  f"{obs['battery_pc']:>6.2f}  "
                  f"{reward['value']:>8.3f}")

    grade = get("/grade", session_id)
    if verbose:
        print(f"\n{'='*62}")
        print(f"  Episode finished after {step_n} steps")
        print(f"  Total reward  : {total_rew:.3f}")
        print(f"  Grader score  : {grade['score']:.4f}")
        print(f"  Breakdown     : {json.dumps(grade['breakdown'], indent=4)}")
        print(f"{'='*62}\n")

    return {"total_reward": total_rew, "steps": step_n, "grade": grade}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thermos-RL PID Baseline")
    parser.add_argument("--task",  default="idle_stability",
                        choices=list(TASK_TARGETS.keys()))
    parser.add_argument("--seed",  type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = run_baseline(task_id=args.task, seed=args.seed, verbose=not args.quiet)
    print(f"Final score: {result['grade']['score']:.4f}")