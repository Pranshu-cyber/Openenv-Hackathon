---
title: Thermos-RL
colorFrom: orange
colorTo: red
sdk: docker
pinned: false
license: mit
tags:
  - reinforcement-learning
  - openenv
  - cpu-thermal
  - systems-engineering
---

# 🌡️ Thermos-RL — CPU Thermal & Power Management RL Environment

**Version 1.2 · OpenEnv-compatible · Docker-deployable**

> An AI agent acts as a *Neural Governor*, learning to maximise **Instructions Per Watt (IPW)**
> while keeping the CPU within safe thermal and battery envelopes.

---

## Why Thermos-RL?

Standard OS governors (`ondemand`, `powersave`) use static heuristics that fail at the
complex, non-linear relationship between temperature, power leakage, and varying workloads.
Thermos-RL provides a physically grounded simulation where an RL agent can learn richer policies.

---

## API — Three endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new episode. Body: `{"task_id": "...", "seed": 42}` |
| `POST` | `/step`  | Submit one action. Body: `{"action": {"delta_freq": 0.5}}` |
| `GET`  | `/state` | Peek at current observation (no step taken) |
| `GET`  | `/grade` | Get the grader score for the current episode |
| `GET`  | `/tasks` | List all tasks with difficulty and description |
| `GET`  | `/health`| Liveness probe |

### Quick-start (curl)
```bash
# 1. Start an episode
curl -X POST http://localhost:8000/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "idle_stability", "seed": 42}'

# 2. Take an action (increase frequency slightly)
curl -X POST http://localhost:8000/step \
     -H "Content-Type: application/json" \
     -d '{"action": {"delta_freq": 0.3}}'

# 3. Grade the episode
curl http://localhost:8000/grade
```

---

## Observation Space

| Field        | Type  | Range        | Unit |
|-------------|-------|--------------|------|
| `cpu_temp`   | float | 0 – 120      | °C   |
| `cpu_load`   | float | 0.0 – 1.0    | —    |
| `curr_freq`  | float | 0.8 – 5.0    | GHz  |
| `power_draw` | float | 0 – ~200     | W    |
| `battery_pc` | float | 0.0 – 100.0  | %    |
| `time_step`  | int   | 0 – 500      | —    |

## Action Space

| Field        | Type  | Range      | Effect |
|-------------|-------|------------|--------|
| `delta_freq` | float | −1.0 – 1.0 | Scales CPU frequency up/down by up to 0.3 GHz/step |

---

## Tasks

### 🟢 idle_stability (Easy)
Maintain temperature at **40°C** under constant low load (0.1).
**Score** = `1 − (MAE / 10)` where MAE is mean absolute temperature error.

### 🟡 burst_management (Medium)
Handle periodic load spikes to **0.9** without thermal throttling (>95°C).
Requires balancing fan power vs CPU throughput.
**Score** = `WorkDone / IdealWork` — drops to **0** if any throttle event occurs.

> ⚠️ *The fan draws power quadratically above 65°C: `P_fan = K·(T−65)²`*
> *Getting too hot creates a **Battery Tax** that hurts the eco_endurance task.*

### 🔴 eco_endurance (Hard)
Maximise instructions-per-watt starting on only **10% battery** under sinusoidally
fluctuating loads.
**Score** = `TotalInstructions / BatteryConsumed` (normalised).

---

## Physics Engine

```
Heat model : T_{t+1} = T_t + (Load × Freq² × C₁) − (Cooling × C₂)
Power model: P_total = P_static + (C₃ × Freq³ × Load) + P_fan
Fan model  : P_fan   = K·(T − 65)²  if T ≥ 65°C, else 0
```

---

## Reward Function

```
R = 10·exp(−0.1·|T − T_target|)   ← precision signal
  + Throughput / PowerDraw         ← efficiency signal
  − JitterPenalty                  ← stability
  − 100 (if battery = 0 or T ≥ 100°C)   ← catastrophic penalty
```

---

## Running Locally

```bash
# Install
pip install -r requirements.txt

# Start server
uvicorn server.app:app --port 8000

# Run PID baseline
python scripts/baseline.py --task idle_stability

# Run training loop
python scripts/train_agent.py --task burst_management --episodes 200

# Run tests
pytest tests/ -v
```

## Docker

```bash
docker build -t thermos-rl .
docker run -p 8000:8000 thermos-rl
```

---

## File Structure

```
thermos-rl/
├── thermos_env/
│   ├── __init__.py
│   ├── environment.py   ← Physics engine + step/reset/grade
│   └── models.py        ← Pydantic schemas (OpenEnv-validated)
├── server/
│   ├── __init__.py
│   └── app.py           ← FastAPI REST layer
├── scripts/
│   ├── baseline.py      ← PID controller baseline
│   └── train_agent.py   ← Q-learning training loop
├── tests/
│   ├── test_environment.py
│   └── test_api.py
├── openenv.yaml
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## License
MIT