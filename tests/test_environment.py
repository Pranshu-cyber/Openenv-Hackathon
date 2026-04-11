"""
tests/test_environment.py
Full test-suite for the Thermos-RL physics engine, reward function, and graders.
Run with:  pytest tests/test_environment.py -v
"""
from __future__ import annotations

import math
import pytest

from env.environment import (
    ThermosEnv, TASK_PROFILES,
    C1, C2, C3, P_STATIC, K_FAN, FAN_THRESH, T_SHUTDOWN, FREQ_MIN, FREQ_MAX,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def env():
    return ThermosEnv()


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
class TestReset:
    def test_reset_returns_valid_observation(self, env):
        obs = env.reset("idle_stability", seed=0)
        assert "cpu_temp"   in obs
        assert "cpu_load"   in obs
        assert "curr_freq"  in obs
        assert "power_draw" in obs
        assert "battery_pc" in obs
        assert "time_step"  in obs
        assert obs["time_step"] == 0

    def test_reset_battery_idle(self, env):
        obs = env.reset("idle_stability")
        assert obs["battery_pc"] == pytest.approx(100.0, abs=0.1)

    def test_reset_battery_eco(self, env):
        obs = env.reset("eco_endurance")
        assert obs["battery_pc"] == pytest.approx(10.0, abs=0.1)

    def test_reset_unknown_task_raises(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset("nonexistent_task")

    def test_reset_is_reproducible_with_seed(self, env):
        obs1 = env.reset("burst_management", seed=42)
        obs2 = env.reset("burst_management", seed=42)
        assert obs1["cpu_temp"] == pytest.approx(obs2["cpu_temp"], abs=1e-6)

    def test_done_flag_cleared_after_reset(self, env):
        env.reset("idle_stability")
        assert not env.is_done()

    def test_step_after_done_without_reset_raises(self, env):
        # is_done starts True before first reset
        with pytest.raises(RuntimeError):
            env.step(0.0)


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
class TestPhysics:
    def test_fan_power_zero_below_threshold(self, env):
        env.reset("idle_stability", seed=0)
        env._temp = 60.0
        # Access internal helper
        p_fan = env._compute_fan_power(60.0)
        assert p_fan == 0.0

    def test_fan_power_nonzero_above_threshold(self, env):
        p_fan = env._compute_fan_power(75.0)
        expected = K_FAN * ((75.0 - FAN_THRESH) ** 2)
        assert p_fan == pytest.approx(expected, rel=1e-6)

    def test_freq_clamped_at_max(self, env):
        env.reset("idle_stability", seed=0)
        env._freq = FREQ_MAX
        env.step(1.0)          # try to go higher
        obs = env.state()
        assert obs["curr_freq"] <= FREQ_MAX

    def test_freq_clamped_at_min(self, env):
        env.reset("idle_stability", seed=0)
        env._freq = FREQ_MIN
        env.step(-1.0)
        obs = env.state()
        assert obs["curr_freq"] >= FREQ_MIN

    def test_battery_drains_each_step(self, env):
        obs0 = env.reset("idle_stability", seed=0)
        bat0 = obs0["battery_pc"]
        obs1, _, _, _, _ = env.step(0.0)
        assert obs1["battery_pc"] < bat0

    def test_temperature_rises_under_high_load(self, env):
        env.reset("burst_management", seed=0)
        env._load = 0.95
        env._freq = FREQ_MAX
        temp_before = env._temp
        env.step(1.0)
        assert env._temp > temp_before

    def test_temperature_falls_at_zero_load(self, env):
        env.reset("idle_stability", seed=0)
        env._temp = 80.0
        env._load = 0.0
        env._freq = FREQ_MIN
        env.step(-1.0)
        assert env._temp < 80.0


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
class TestReward:
    def test_reward_has_required_keys(self, env):
        env.reset("idle_stability", seed=0)
        _, rew, _, _, _ = env.step(0.0)
        assert "value"      in rew
        assert "components" in rew
        for key in ("precision", "efficiency", "jitter_penalty"):
            assert key in rew["components"]

    def test_reward_precision_max_at_target(self, env):
        env.reset("idle_stability", seed=0)
        env._temp = TASK_PROFILES["idle_stability"].t_target
        _, rew, _, _, _ = env.step(0.0)
        # At target the precision component should be close to 10.0
        assert rew["components"]["precision"] == pytest.approx(10.0, abs=0.5)

    def test_catastrophic_penalty_on_battery_death(self, env):
        env.reset("idle_stability", seed=0)
        env._battery = 0.0001    # almost dead
        env._freq    = FREQ_MAX
        _, rew, done, _, info = env.step(1.0)
        assert done
        assert "catastrophic_penalty" in rew["components"]
        assert rew["components"]["catastrophic_penalty"] == -100.0

    def test_catastrophic_penalty_on_thermal_shutdown(self, env):
        env.reset("idle_stability", seed=0)
        env._temp = T_SHUTDOWN - 0.1
        env._load = 1.0
        env._freq = FREQ_MAX
        _, rew, done, _, _ = env.step(1.0)
        if done:
            assert "catastrophic_penalty" in rew["components"]


# ---------------------------------------------------------------------------
# Step flow
# ---------------------------------------------------------------------------
class TestStep:
    def test_step_increments_time_step(self, env):
        env.reset("idle_stability", seed=0)
        for i in range(1, 6):
            obs, _, _, _, _ = env.step(0.0)
            assert obs["time_step"] == i

    def test_episode_truncates_at_max_steps(self, env):
        env.reset("idle_stability", seed=0)
        done = truncated = False
        steps = 0
        while not (done or truncated):
            _, _, done, truncated, _ = env.step(0.0)
            steps += 1
        assert truncated or done
        assert steps <= TASK_PROFILES["idle_stability"].max_steps + 1

    def test_state_matches_last_obs(self, env):
        env.reset("idle_stability", seed=1)
        obs, _, _, _, _ = env.step(0.2)
        state = env.state()
        assert state["cpu_temp"]  == pytest.approx(obs["cpu_temp"],  abs=1e-6)
        assert state["curr_freq"] == pytest.approx(obs["curr_freq"], abs=1e-6)


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------
class TestGraders:
    def _run_full_episode(self, env, task_id, action=0.0, seed=0):
        env.reset(task_id, seed=seed)
        done = truncated = False
        while not (done or truncated):
            _, _, done, truncated, _ = env.step(action)
        return env.grade()

    def test_idle_stability_score_range(self, env):
        result = self._run_full_episode(env, "idle_stability")
        assert 0.0 <= result["score"] <= 1.0

    def test_burst_management_score_range(self, env):
        result = self._run_full_episode(env, "burst_management")
        assert 0.0 <= result["score"] <= 1.0

    def test_eco_endurance_score_range(self, env):
        result = self._run_full_episode(env, "eco_endurance")
        assert 0.0 <= result["score"] <= 1.0

    def test_idle_stability_breakdown_present(self, env):
        result = self._run_full_episode(env, "idle_stability")
        assert "mean_absolute_error" in result["breakdown"]
        assert "steps_evaluated"     in result["breakdown"]

    def test_burst_throttle_events_counted(self, env):
        result = self._run_full_episode(env, "burst_management", action=1.0)
        assert "throttle_events" in result["breakdown"]

    def test_eco_endurance_breakdown(self, env):
        result = self._run_full_episode(env, "eco_endurance")
        assert "total_instructions"  in result["breakdown"]
        assert "battery_consumed_pc" in result["breakdown"]
        assert "instructions_per_pc" in result["breakdown"]