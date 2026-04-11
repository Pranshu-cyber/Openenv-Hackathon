"""
tests/test_api.py
Integration tests for the FastAPI endpoints using TestClient (no server needed).
Run with:  pytest tests/test_api.py -v
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health & Meta
# ---------------------------------------------------------------------------
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_tasks():
    r = client.get("/tasks")
    assert r.status_code == 200
    ids = [t["id"] for t in r.json()["tasks"]]
    assert "idle_stability"   in ids
    assert "burst_management" in ids
    assert "eco_endurance"    in ids


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
def test_reset_idle_stability():
    r = client.post("/reset", json={"task_id": "idle_stability", "seed": 0})
    assert r.status_code == 200
    body = r.json()
    assert "observation" in body
    obs = body["observation"]
    for key in ("cpu_temp", "cpu_load", "curr_freq", "power_draw", "battery_pc", "time_step"):
        assert key in obs
    assert obs["time_step"] == 0


def test_reset_unknown_task_returns_422():
    r = client.post("/reset", json={"task_id": "fake_task"})
    assert r.status_code == 422


def test_reset_all_tasks():
    for task_id in ("idle_stability", "burst_management", "eco_endurance"):
        r = client.post("/reset", json={"task_id": task_id})
        assert r.status_code == 200, f"Failed for task_id={task_id}"


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------
def test_step_after_reset():
    client.post("/reset", json={"task_id": "idle_stability", "seed": 1})
    r = client.post("/step", json={"action": {"delta_freq": 0.0}})
    assert r.status_code == 200
    body = r.json()
    assert "observation" in body
    assert "reward"      in body
    assert "done"        in body
    assert "truncated"   in body
    assert body["observation"]["time_step"] == 1


def test_step_action_bounds():
    client.post("/reset", json={"task_id": "idle_stability", "seed": 0})
    # Should accept boundary values
    for delta in (-1.0, -0.5, 0.0, 0.5, 1.0):
        r = client.post("/step", json={"action": {"delta_freq": delta}})
        assert r.status_code == 200


def test_step_invalid_action_rejected():
    client.post("/reset", json={"task_id": "idle_stability", "seed": 0})
    r = client.post("/step", json={"action": {"delta_freq": 2.0}})  # out of range
    assert r.status_code == 422


def test_step_without_reset_returns_400():
    # Force done state by exhausting a short episode
    # (Rely on the fact that the env may already be done)
    # We directly probe: if done, server must return 400
    from server.app import _env
    _env._done = True
    r = client.post("/step", json={"action": {"delta_freq": 0.0}})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def test_state_returns_observation():
    client.post("/reset", json={"task_id": "idle_stability", "seed": 0})
    r = client.get("/state")
    assert r.status_code == 200
    body = r.json()
    assert "observation" in body
    assert "task_id"     in body
    assert "step"        in body
    assert "done"        in body


def test_state_does_not_advance_step():
    client.post("/reset", json={"task_id": "idle_stability", "seed": 0})
    s1 = client.get("/state").json()["step"]
    s2 = client.get("/state").json()["step"]
    assert s1 == s2 == 0


# ---------------------------------------------------------------------------
# Grade
# ---------------------------------------------------------------------------
def test_grade_returns_score():
    client.post("/reset", json={"task_id": "idle_stability", "seed": 0})
    for _ in range(5):
        client.post("/step", json={"action": {"delta_freq": 0.0}})
    r = client.get("/grade")
    assert r.status_code == 200
    body = r.json()
    assert "score"     in body
    assert "task_id"   in body
    assert "breakdown" in body
    assert 0.0 <= body["score"] <= 1.0


# ---------------------------------------------------------------------------
# Reward structure
# ---------------------------------------------------------------------------
def test_reward_components_present():
    client.post("/reset", json={"task_id": "idle_stability", "seed": 0})
    r = client.post("/step", json={"action": {"delta_freq": 0.1}})
    rew = r.json()["reward"]
    assert "value"      in rew
    assert "components" in rew
    for key in ("precision", "efficiency", "jitter_penalty"):
        assert key in rew["components"]