"""
inference.py  —  Thermos-RL OpenEnv Inference Script
-----------------------------------------------------
Mandatory submission script. Uses OpenAI-compatible client to drive
the Thermos-RL environment through all 3 tasks.

Structured stdout logs follow [START] / [STEP] / [END] format exactly
as required by the OpenEnv evaluation harness.

Environment variables required:
  API_BASE_URL   LLM API endpoint  (e.g. https://api.openai.com/v1)
  MODEL_NAME     Model identifier  (e.g. gpt-4o-mini)
  HF_TOKEN       HuggingFace / API key

Optional:
  ENV_BASE_URL   Thermos-RL Space URL (default: http://localhost:8000)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables (mandatory per submission spec)
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN:     str = os.environ.get("HF_TOKEN",     "dummy")
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------
TASKS          = ["idle_stability", "burst_management", "eco_endurance"]
TASK_NAME      = "thermos-rl-all-tasks"
BENCHMARK      = "Thermos-RL"
MAX_STEPS      = 500
MAX_TOTAL_REWARD = 1000.0
SUCCESS_SCORE_THRESHOLD = 0.5
SEED           = 42

# ---------------------------------------------------------------------------
# Structured log helpers  —  [START] / [STEP] / [END]
# MUST match exactly — field names, ordering, formatting
# ---------------------------------------------------------------------------
def log_start(*, task: str, env: str, model: str) -> None:
    print(json.dumps({
        "type":  "START",
        "task":  task,
        "env":   env,
        "model": model,
    }), flush=True)


def log_step(*,
             step:   int,
             action: float,
             reward: float,
             done:   bool,
             error:  Optional[str] = None) -> None:
    payload = {
        "type":   "STEP",
        "step":   step,
        "action": round(action, 6),
        "reward": round(reward, 6),
        "done":   done,
    }
    if error is not None:
        payload["error"] = error
    print(json.dumps(payload), flush=True)


def log_end(*,
            success: bool,
            steps:   int,
            score:   float,
            rewards: List[float]) -> None:
    print(json.dumps({
        "type":    "END",
        "success": success,
        "steps":   steps,
        "score":   round(score, 6),
        "rewards": [round(r, 6) for r in rewards],
    }), flush=True)


# ---------------------------------------------------------------------------
# OpenEnv HTTP helpers
# ---------------------------------------------------------------------------
def env_reset(task_id: str, seed: int) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_step(delta_freq: float, session_id: str) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/step?session_id={session_id}",
        json={"action": {"delta_freq": delta_freq}},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_grade(session_id: str) -> dict:
    r = requests.get(
        f"{ENV_BASE_URL}/grade?session_id={session_id}",
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# LLM agent — uses OpenAI client as required
# ---------------------------------------------------------------------------
def get_model_action(
    client:     OpenAI,
    step:       int,
    obs:        dict,
    last_reward: float,
    history:    List[str],
    task_id:    str,
) -> float:
    """
    Ask the LLM to decide the next delta_freq action.
    Falls back to PID heuristic if the model call fails.
    Returns a float in [-1, 1].
    """
    system_prompt = (
        "You are a CPU thermal governor agent controlling a simulated CPU. "
        "Your goal is to optimise Instructions Per Watt while keeping the CPU "
        "within safe thermal and battery limits.\n\n"
        "Given the current CPU state, respond with ONLY a JSON object like:\n"
        '{"delta_freq": 0.3}\n\n'
        "Rules:\n"
        "- delta_freq must be a float between -1.0 and 1.0\n"
        "- Negative = decrease frequency (cooler, less power)\n"
        "- Positive = increase frequency (hotter, more work)\n"
        "- If cpu_temp > 80, decrease frequency urgently\n"
        "- If battery_pc < 5, decrease frequency to save power\n"
        "- Respond with ONLY the JSON, no explanation."
    )

    user_msg = (
        f"Task: {task_id} | Step: {step}\n"
        f"cpu_temp:   {obs['cpu_temp']:.2f} °C\n"
        f"cpu_load:   {obs['cpu_load']:.3f}\n"
        f"curr_freq:  {obs['curr_freq']:.3f} GHz\n"
        f"power_draw: {obs['power_draw']:.2f} W\n"
        f"battery_pc: {obs['battery_pc']:.2f} %\n"
        f"last_reward:{last_reward:.4f}\n"
        f"recent_history: {history[-3:] if history else []}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=32,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        delta = float(parsed["delta_freq"])
        return max(-1.0, min(1.0, delta))

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # PID fallback — temperature-based heuristic
        TARGETS = {
            "idle_stability":   40.0,
            "burst_management": 70.0,
            "eco_endurance":    65.0,
        }
        error = obs["cpu_temp"] - TARGETS.get(task_id, 65.0)
        return max(-1.0, min(1.0, -(error * 0.05)))


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------
def run_task(client: OpenAI, task_id: str) -> dict:
    print(f"[DEBUG] Starting task: {task_id}", flush=True)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        reset_resp  = env_reset(task_id, seed=SEED)
        session_id  = reset_resp["info"]["session_id"]
        obs         = reset_resp["observation"]
        last_reward = 0.0
        done        = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Get action from LLM (or fallback PID)
            action = get_model_action(
                client, step, obs, last_reward, history, task_id
            )

            # Step the environment
            error = None
            try:
                step_resp   = env_step(action, session_id)
                obs         = step_resp["observation"]
                reward      = step_resp["reward"]["value"]
                done        = step_resp["done"] or step_resp["truncated"]
            except Exception as exc:
                error  = str(exc)
                reward = 0.0
                done   = True
                print(f"[DEBUG] Step error: {exc}", flush=True)

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(
                step=step,
                action=action,
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                f"Step {step}: delta={action:+.3f} -> reward {reward:+.4f}"
            )

            if done:
                break

        # Grade the episode
        try:
            grade  = env_grade(session_id)
            score  = float(grade["score"])
        except Exception as exc:
            print(f"[DEBUG] Grade error: {exc}", flush=True)
            total_r = sum(rewards)
            score   = min(max(total_r / MAX_TOTAL_REWARD, 0.0), 1.0)

        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task error: {exc}", flush=True)
        score   = 0.0
        success = False

    log_end(
        success=success,
        steps=steps_taken,
        score=score,
        rewards=rewards,
    )

    print(
        f"[DEBUG] Task {task_id} finished — "
        f"score={score:.4f} success={success} steps={steps_taken}",
        flush=True,
    )

    return {"task_id": task_id, "score": score, "success": success}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"[DEBUG] Thermos-RL inference starting", flush=True)
    print(f"[DEBUG] API_BASE_URL = {API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME   = {MODEL_NAME}",   flush=True)
    print(f"[DEBUG] ENV_BASE_URL = {ENV_BASE_URL}", flush=True)

    # Validate env vars
    missing = [v for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN")
               if not os.environ.get(v)]
    if missing:
        print(f"[DEBUG] WARNING: env vars not set: {missing}", flush=True)

    # OpenAI client — mandatory per submission spec
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    results = []
    start_time = time.time()

    for task_id in TASKS:
        result = run_task(client, task_id)
        results.append(result)

        elapsed = time.time() - start_time
        print(
            f"[DEBUG] Elapsed: {elapsed:.1f}s / 1200s limit",
            flush=True,
        )

        # Safety check — stop if approaching 20 min limit
        if elapsed > 1100:
            print("[DEBUG] Approaching time limit — stopping early", flush=True)
            break

    # Final summary
    print("\n[DEBUG] ===== FINAL RESULTS =====", flush=True)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"[DEBUG] {r['task_id']:<22} score={r['score']:.4f}  {status}",
            flush=True,
        )

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"[DEBUG] Average score: {avg_score:.4f}", flush=True)
    print("[DEBUG] Thermos-RL inference complete", flush=True)


if __name__ == "__main__":
    main()