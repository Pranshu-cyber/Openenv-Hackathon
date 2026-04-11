"""
thermos_env/models.py
Typed Pydantic models for the Thermos-RL OpenEnv environment.
All API request / response bodies are validated against these schemas.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class CPUAction(BaseModel):
    """
    Action submitted by the agent each step.

    delta_freq : continuous value in [-1, 1]
        -1  =>  decrease frequency as fast as possible
        +1  =>  increase frequency as fast as possible
         0  =>  hold current frequency
    """
    delta_freq: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Normalised frequency delta in [-1, 1]"
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class CPUObservation(BaseModel):
    """Full observable state returned to the agent after every step."""
    cpu_temp:   float = Field(..., description="CPU temperature in °C")
    cpu_load:   float = Field(..., description="Current workload demand [0, 1]")
    curr_freq:  float = Field(..., description="Current CPU frequency in GHz [0.8, 5.0]")
    power_draw: float = Field(..., description="Total system power draw in Watts")
    battery_pc: float = Field(..., description="Remaining battery percentage [0, 100]")
    time_step:  int   = Field(..., description="Step index within the current episode")


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class CPUReward(BaseModel):
    """Scalar reward plus a named breakdown for transparency / debugging."""
    value:      float            = Field(..., description="Scalar reward this step")
    components: Dict[str, float] = Field(
        ...,
        description="Sub-component breakdown: precision, efficiency, jitter, penalty"
    )


# ---------------------------------------------------------------------------
# API Request / Response wrappers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """
    All fields are optional so the validator can POST an empty body {}.
    Defaults: task_id="idle_stability", seed=None
    """
    task_id: Optional[str] = Field(
        "idle_stability",
        description="One of: idle_stability | burst_management | eco_endurance"
    )
    seed: Optional[int] = Field(None, description="RNG seed for reproducibility")

    @property
    def resolved_task_id(self) -> str:
        return self.task_id or "idle_stability"


class ResetResponse(BaseModel):
    observation: CPUObservation
    task_id:     str
    info:        Dict[str, Any] = {}


class StepRequest(BaseModel):
    """
    action is optional — defaults to delta_freq=0.0 (hold frequency).
    Allows the validator to POST an empty body {}.
    """
    action: Optional[CPUAction] = Field(
        None,
        description="If omitted, defaults to delta_freq=0.0 (hold)"
    )

    @property
    def resolved_action(self) -> CPUAction:
        return self.action or CPUAction(delta_freq=0.0)


class StepResponse(BaseModel):
    observation: CPUObservation
    reward:      CPUReward
    done:        bool
    truncated:   bool
    info:        Dict[str, Any] = {}


class StateResponse(BaseModel):
    observation: CPUObservation
    task_id:     str
    step:        int
    done:        bool
    info:        Dict[str, Any] = {}


class GradeResponse(BaseModel):
    task_id:   str
    score:     float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, Any] = {}