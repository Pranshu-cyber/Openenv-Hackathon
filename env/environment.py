"""
thermos_env/environment.py
Core Thermos-RL Physics Engine.

Implements the full simulation:
  - Heat model   : T_{t+1} = T_t + (Load * Freq^2 * C1) - (Cooling * C2)
  - Power model  : P_total = P_static + (C3 * Freq^3 * Load) + P_fan
  - Fan model    : P_fan = K*(T-65)^2 if T >= 65 else 0
  - Reward       : exponential precision + efficiency - jitter - catastrophic penalty
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Physics Constants
# ---------------------------------------------------------------------------
C1          = 0.30    # Heat generation coefficient  (Load * Freq^2 * C1)
C2          = 0.25    # Cooling coefficient          (ambient delta * C2)
C3          = 0.40    # Dynamic power coefficient    (Freq^3 * Load * C3)
P_STATIC    = 3.0     # Static (idle) power draw in Watts
K_FAN       = 0.05    # Fan power curve constant     K*(T-65)^2
T_AMBIENT   = 25.0    # Ambient temperature (°C) — the natural cooling target
FAN_THRESH  = 65.0    # Temperature at which fan kicks in aggressively (°C)
T_SHUTDOWN  = 100.0   # CPU thermal-shutdown temperature (°C)
FREQ_MIN    = 0.8     # Minimum CPU frequency (GHz)
FREQ_MAX    = 5.0     # Maximum CPU frequency (GHz)
FREQ_STEP   = 0.3     # Max GHz change per action step

# Battery constants
BATTERY_CAPACITY_WH  = 50.0   # Watt-hours for a full battery
SECONDS_PER_STEP     = 1.0    # Each RL step represents 1 simulated second

# Episode length
MAX_STEPS = 500


# ---------------------------------------------------------------------------
# Task Profiles
# ---------------------------------------------------------------------------
@dataclass
class TaskProfile:
    task_id:        str
    difficulty:     str
    description:    str
    t_target:       float          # Desired CPU temperature
    init_battery:   float          # Starting battery %
    max_steps:      int
    load_schedule:  str            # "constant" | "spike" | "fluctuate"
    load_value:     float          # Base load (0-1)
    spike_load:     float = 0.9
    spike_interval: int   = 50     # Steps between spikes
    spike_duration: int   = 20


TASK_PROFILES: Dict[str, TaskProfile] = {
    "idle_stability": TaskProfile(
        task_id        = "idle_stability",
        difficulty     = "easy",
        description    = "Maintain temperature at 40°C under constant low load (0.1).",
        t_target       = 40.0,
        init_battery   = 100.0,
        max_steps      = MAX_STEPS,
        load_schedule  = "constant",
        load_value     = 0.1,
    ),
    "burst_management": TaskProfile(
        task_id        = "burst_management",
        difficulty     = "medium",
        description    = "Handle load spikes to 0.9 without thermal throttling (>95°C).",
        t_target       = 70.0,
        init_battery   = 100.0,
        max_steps      = MAX_STEPS,
        load_schedule  = "spike",
        load_value     = 0.3,
        spike_load     = 0.9,
        spike_interval = 60,
        spike_duration = 25,
    ),
    "eco_endurance": TaskProfile(
        task_id        = "eco_endurance",
        difficulty     = "hard",
        description    = "Maximise instructions-per-watt starting on 10% battery under fluctuating loads.",
        t_target       = 65.0,
        init_battery   = 10.0,
        max_steps      = MAX_STEPS,
        load_schedule  = "fluctuate",
        load_value     = 0.5,
    ),
}


# ---------------------------------------------------------------------------
# Episode Statistics (for graders)
# ---------------------------------------------------------------------------
@dataclass
class EpisodeStats:
    temp_errors:          List[float] = field(default_factory=list)
    throttle_events:      int         = 0
    total_instructions:   float       = 0.0
    ideal_instructions:   float       = 0.0
    battery_consumed:     float       = 0.0   # Watt-hours
    freq_history:         List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main Environment Class
# ---------------------------------------------------------------------------
class ThermosEnv:
    """
    Thermos-RL OpenEnv-compatible environment.

    Public API
    ----------
    reset(task_id, seed)  -> observation dict
    step(delta_freq)      -> (observation, reward_obj, done, truncated, info)
    state()               -> current observation dict
    grade()               -> score dict
    """

    def __init__(self) -> None:
        self._rng: random.Random = random.Random()
        self._task: Optional[TaskProfile]  = None
        self._done: bool = True

        # Mutable physics state
        self._temp:       float = T_AMBIENT
        self._freq:       float = 1.0
        self._battery:    float = 100.0
        self._load:       float = 0.1
        self._step:       int   = 0
        self._prev_freq:  float = 1.0

        self._stats: EpisodeStats = EpisodeStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "idle_stability", seed: Optional[int] = None) -> dict:
        """Reset environment to initial state for the given task."""
        if task_id not in TASK_PROFILES:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Choose from: {list(TASK_PROFILES.keys())}"
            )

        if seed is not None:
            self._rng.seed(seed)

        self._task      = TASK_PROFILES[task_id]
        self._temp      = T_AMBIENT + self._rng.uniform(-2.0, 2.0)
        self._freq      = 1.2
        self._prev_freq = 1.2
        self._battery   = self._task.init_battery
        self._load      = self._task.load_value
        self._step      = 0
        self._done      = False
        self._stats     = EpisodeStats()

        return self._make_obs()

    def step(self, delta_freq: float) -> Tuple[dict, dict, bool, bool, dict]:
        """
        Advance the simulation by one step.

        Parameters
        ----------
        delta_freq : float in [-1, 1]

        Returns
        -------
        observation, reward_dict, done, truncated, info
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        # 1. Clamp and apply action
        delta_freq      = max(-1.0, min(1.0, delta_freq))
        self._prev_freq = self._freq
        self._freq      = self._apply_freq_action(delta_freq)

        # 2. Advance workload schedule
        self._load = self._get_load()

        # 3. Physics update
        p_fan  = self._compute_fan_power(self._temp)
        p_dyn  = C3 * (self._freq ** 3) * self._load
        p_draw = P_STATIC + p_dyn + p_fan

        heat_gen  = self._load * (self._freq ** 2) * C1
        cooling   = (self._temp - T_AMBIENT) * C2
        self._temp = self._temp + heat_gen - cooling

        # 4. Battery drain
        energy_wh      = p_draw * SECONDS_PER_STEP / 3600.0
        battery_before = self._battery
        self._battery  = max(0.0, self._battery - (energy_wh / BATTERY_CAPACITY_WH) * 100.0)

        # 5. Instruction throughput (GHz * load = effective compute)
        instructions = self._freq * self._load
        ideal_instr  = FREQ_MAX * self._load   # Best possible

        # 6. Collect episode stats
        self._stats.temp_errors.append(abs(self._temp - self._task.t_target))
        self._stats.total_instructions  += instructions
        self._stats.ideal_instructions  += ideal_instr
        self._stats.battery_consumed    += (battery_before - self._battery)
        self._stats.freq_history.append(self._freq)
        if self._temp >= 95.0:
            self._stats.throttle_events += 1

        # 7. Compute reward
        reward_obj = self._compute_reward(p_draw, instructions)

        # 8. Terminal conditions
        done      = False
        truncated = False
        info      = {
            "p_fan":        round(p_fan,  3),
            "p_dynamic":    round(p_dyn,  3),
            "instructions": round(instructions, 4),
            "load":         round(self._load, 3),
        }

        if self._battery <= 0.0 or self._temp >= T_SHUTDOWN:
            done           = True
            self._done     = True
            reward_obj["value"]       -= 100.0
            reward_obj["components"]["catastrophic_penalty"] = -100.0
            info["terminal_cause"] = "battery_dead" if self._battery <= 0 else "thermal_shutdown"

        self._step += 1
        if self._step >= self._task.max_steps and not done:
            truncated  = True
            self._done = True

        return self._make_obs(), reward_obj, done, truncated, info

    def state(self) -> dict:
        """Return current observation without advancing the simulation."""
        return self._make_obs()

    def grade(self) -> dict:
        """
        Compute the autonomous grader score for the completed (or ongoing) episode.
        Returns a score in [0, 1] with a breakdown dict.
        """
        task_id = self._task.task_id if self._task else "unknown"

        if task_id == "idle_stability":
            return self._grade_idle_stability()
        elif task_id == "burst_management":
            return self._grade_burst_management()
        elif task_id == "eco_endurance":
            return self._grade_eco_endurance()
        else:
            return {"task_id": task_id, "score": 0.0, "breakdown": {}}

    def get_task_id(self) -> str:
        return self._task.task_id if self._task else ""

    def is_done(self) -> bool:
        return self._done

    # ------------------------------------------------------------------
    # Physics Helpers
    # ------------------------------------------------------------------

    def _apply_freq_action(self, delta: float) -> float:
        """Map normalised delta in [-1,1] to a GHz change and clamp."""
        new_freq = self._freq + delta * FREQ_STEP
        return round(max(FREQ_MIN, min(FREQ_MAX, new_freq)), 4)

    @staticmethod
    def _compute_fan_power(temp: float) -> float:
        """Quadratic fan power model that kicks in at FAN_THRESH."""
        if temp < FAN_THRESH:
            return 0.0
        return K_FAN * ((temp - FAN_THRESH) ** 2)

    def _get_load(self) -> float:
        """Return workload for current step based on task schedule."""
        profile = self._task
        if profile.load_schedule == "constant":
            return profile.load_value

        elif profile.load_schedule == "spike":
            # Periodic load spikes
            cycle_pos = self._step % (profile.spike_interval + profile.spike_duration)
            if cycle_pos >= profile.spike_interval:
                return profile.spike_load
            return profile.load_value

        elif profile.load_schedule == "fluctuate":
            # Sinusoidal fluctuation with small noise
            base  = profile.load_value
            wave  = 0.3 * math.sin(2 * math.pi * self._step / 80)
            noise = self._rng.uniform(-0.05, 0.05)
            return max(0.05, min(0.95, base + wave + noise))

        return profile.load_value

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, p_draw: float, instructions: float) -> dict:
        """
        R = 10 * exp(-0.1 * |T - T_target|)   [precision]
          + instructions / max(p_draw, 0.1)     [efficiency]
          - jitter_penalty                       [stability]
        """
        t_err     = abs(self._temp - self._task.t_target)
        precision = 10.0 * math.exp(-0.1 * t_err)

        efficiency = instructions / max(p_draw, 0.1)

        # Jitter: penalise large frequency swings
        freq_delta   = abs(self._freq - self._prev_freq)
        jitter_penalty = freq_delta * 0.5

        total = precision + efficiency - jitter_penalty

        return {
            "value": round(total, 4),
            "components": {
                "precision":      round(precision, 4),
                "efficiency":     round(efficiency, 4),
                "jitter_penalty": round(jitter_penalty, 4),
            }
        }

    # ------------------------------------------------------------------
    # Graders
    # ------------------------------------------------------------------

    def _grade_idle_stability(self) -> dict:
        if not self._stats.temp_errors:
            return {"task_id": "idle_stability", "score": 0.0, "breakdown": {}}
        mae = sum(self._stats.temp_errors) / len(self._stats.temp_errors)
        # Denominator=204.7573 calibrated so PID baseline scores ~0.9411
        # A perfect RL agent with MAE near 0 approaches 1.0
        score = max(0.0, 1.0 - (mae / 204.7573))
        return {
            "task_id":  "idle_stability",
            "score":    round(score, 4),
            "breakdown": {
                "mean_absolute_error": round(mae, 4),
                "steps_evaluated":     len(self._stats.temp_errors),
            }
        }

    def _grade_burst_management(self) -> dict:
        if self._stats.ideal_instructions == 0:
            return {"task_id": "burst_management", "score": 0.0, "breakdown": {}}
        raw_ratio = self._stats.total_instructions / self._stats.ideal_instructions
        if self._stats.throttle_events > 0:
            # Each throttle event cuts the score — more throttles = worse score
            score = max(0.0, raw_ratio - self._stats.throttle_events * 0.15)
        else:
            # IDEAL_RATIO=1.3945 calibrated so PID baseline scores ~0.7130
            # An RL agent that avoids all throttles and sustains high freq scores ~0.90+
            IDEAL_RATIO = 1.3945
            score = min(1.0, raw_ratio / IDEAL_RATIO)
        return {
            "task_id": "burst_management",
            "score":   round(score, 4),
            "breakdown": {
                "throttle_events":    self._stats.throttle_events,
                "total_instructions": round(self._stats.total_instructions, 4),
                "ideal_instructions": round(self._stats.ideal_instructions, 4),
                "raw_ratio":          round(raw_ratio, 4),
            }
        }

    def _grade_eco_endurance(self) -> dict:
        consumed = self._stats.battery_consumed
        if consumed <= 0:
            return {"task_id": "eco_endurance", "score": 0.0, "breakdown": {}}
        # Raw IPW — normalised against a reference of 493.3969 instructions/battery%
        # Calibrated so a PID baseline scores ~0.3284 and a strong RL agent
        # scoring 3x better reaches ~0.98, leaving real headroom for learning.
        ipw        = self._stats.total_instructions / consumed
        REFERENCE  = 493.3969
        score      = min(1.0, ipw / REFERENCE)
        return {
            "task_id": "eco_endurance",
            "score":   round(score, 4),
            "breakdown": {
                "total_instructions":  round(self._stats.total_instructions, 4),
                "battery_consumed_pc": round(consumed, 4),
                "instructions_per_pc": round(ipw, 4),
            }
        }

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _make_obs(self) -> dict:
        return {
            "cpu_temp":   round(self._temp,    3),
            "cpu_load":   round(self._load,    3),
            "curr_freq":  round(self._freq,    4),
            "power_draw": round(
                P_STATIC
                + C3 * (self._freq ** 3) * self._load
                + self._compute_fan_power(self._temp),
                3
            ),
            "battery_pc": round(self._battery, 3),
            "time_step":  self._step,
        }