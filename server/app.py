"""
server/app.py
FastAPI application exposing the Thermos-RL environment via REST.

Endpoints
---------
POST /reset          – start / restart an episode (returns session_id)
POST /step           – submit one action, receive next obs + reward
GET  /state          – peek at current observation without stepping
GET  /grade          – get the grader score for the current episode
GET  /tasks          – list available tasks
GET  /health         – liveness probe

Session isolation
-----------------
Each call to /reset creates a new isolated environment instance identified
by a session_id (UUID). Pass session_id in subsequent /step, /state, /grade
calls. This allows multiple agents to run in parallel without interference.

Backwards compatibility
-----------------------
If no session_id is provided to /step, /state, /grade, the server falls back
to the default single shared session (key="default") so existing scripts
without session support continue to work.
"""
from __future__ import annotations

import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from env.environment import ThermosEnv, TASK_PROFILES
from env.models import (
    CPUObservation, CPUReward,
    ResetRequest, ResetResponse,
    StepRequest, StepResponse,
    StateResponse, GradeResponse,
)

app = FastAPI(
    title="Thermos-RL",
    description="CPU Thermal & Power Management RL Environment — OpenEnv v1.0",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session registry — each session is an independent ThermosEnv instance
# ---------------------------------------------------------------------------
_sessions: Dict[str, ThermosEnv] = {
    "default": ThermosEnv()
}


def _get_session(session_id: Optional[str]) -> ThermosEnv:
    key = session_id or "default"
    if key not in _sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{key}' not found. Call POST /reset first."
        )
    return _sessions[key]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health():
    return {
        "status":          "ok",
        "env":             "Thermos-RL",
        "version":         "1.2.0",
        "active_sessions": len(_sessions),
    }


# ---------------------------------------------------------------------------
# Task listing
# ---------------------------------------------------------------------------

@app.get("/tasks", tags=["meta"])
def list_tasks():
    return {
        "tasks": [
            {
                "id":           t.task_id,
                "difficulty":   t.difficulty,
                "description":  t.description,
                "t_target":     t.t_target,
                "init_battery": t.init_battery,
                "max_steps":    t.max_steps,
            }
            for t in TASK_PROFILES.values()
        ]
    }


# ---------------------------------------------------------------------------
# Core RL loop
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=ResetResponse, tags=["env"])
def reset(req: ResetRequest):
    """
    Reset the environment and begin a new episode.
    Returns session_id in info — pass it to /step, /state, /grade.
    """
    session_id = str(uuid.uuid4())
    env = ThermosEnv()
    _sessions[session_id] = env
    _sessions["default"]  = env   # keep default in sync for backwards compat

    try:
        obs_dict = env.reset(task_id=req.task_id, seed=req.seed)
    except ValueError as exc:
        del _sessions[session_id]
        raise HTTPException(status_code=422, detail=str(exc))

    return ResetResponse(
        observation=CPUObservation(**obs_dict),
        task_id=req.task_id,
        info={
            "message":    f"Episode started — task: {req.task_id}",
            "session_id": session_id,
        },
    )


@app.post("/step", response_model=StepResponse, tags=["env"])
def step(
    req: StepRequest,
    session_id: Optional[str] = Query(None, description="Session ID from /reset")
):
    """Submit one action. Pass session_id query param for parallel isolation."""
    env = _get_session(session_id)

    if env.is_done():
        raise HTTPException(
            status_code=400,
            detail="Episode is finished. Call POST /reset to start a new one."
        )

    try:
        obs_dict, rew_dict, done, truncated, info = env.step(req.action.delta_freq)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return StepResponse(
        observation=CPUObservation(**obs_dict),
        reward=CPUReward(**rew_dict),
        done=done,
        truncated=truncated,
        info=info,
    )


@app.get("/state", response_model=StateResponse, tags=["env"])
def state(
    session_id: Optional[str] = Query(None, description="Session ID from /reset")
):
    """Peek at current observation without stepping."""
    env = _get_session(session_id)
    obs_dict = env.state()
    return StateResponse(
        observation=CPUObservation(**obs_dict),
        task_id=env.get_task_id(),
        step=obs_dict["time_step"],
        done=env.is_done(),
        info={},
    )


@app.get("/grade", response_model=GradeResponse, tags=["env"])
def grade(
    session_id: Optional[str] = Query(None, description="Session ID from /reset")
):
    """Get grader score for this session's current episode."""
    env = _get_session(session_id)
    result = env.grade()
    return GradeResponse(**result)


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
# Entry point — allows running via: python server/app.py
# Port 7860 matches HuggingFace Spaces requirement
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )