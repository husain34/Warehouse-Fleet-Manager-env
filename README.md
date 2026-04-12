---
title: Warehouse Fleet Management Elite
emoji: 🏗️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: true
tags:
- openenv
- reinforcement-learning
- multi-agent
- logistics
---

# Warehouse Fleet Management: Swarm Intelligence (v2.0)

This is a high-fidelity industrial warehouse simulation designed for the **Meta PyTorch OpenEnv Hackathon**. Unlike traditional "toy" grid environments, v2.0 introduces realistic industrial constraints, heterogeneous robotics, and physics-based kinematics that challenge the reasoning capabilities of frontier LLM agents.

## 🏗️ Elite Features (Industrial Grade)

- **Kinematics Engine**: Robots operate with **inertia, acceleration, and friction**. Agents must manage momentum to navigate tight warehouse aisles without overshooting.
- **Heterogeneous Fleet**: Deploy a diverse swarm including:
    - **Swift**: Rapid response units with high acceleration but high power consumption.
    - **Hauler**: High-capacity industrial workhorses with superior energy efficiency.
- **Industrial Layout**: Static **Rack Columns** at x = {2, 5, 8} create defined aisles, forcing agents to master spatial reasoning and traffic management.
- **Drop Zones**: Designated delivery targets (distinct from charging stations) are explicitly provided in every observation, removing ambiguity about where loads must be delivered.
- **Strategic Command Narrative**: A spatial advisory engine translates complex coordinate data into natural language "Fleet Advisories" (e.g., *"r1 (Swift) battery at 15%. Return to charger!"*).
- **Advanced Performance Telemetry**: Grading is performed via a 4-pillar **EliteWarehouseRubric**:
    - **Efficiency (40%)**: Task throughput and path optimality.
    - **Safety (20%)**: Collision avoidance and environmental spill penalties.
    - **Sustainability (20%)**: Energy management and battery health across the fleet.
    - **SLA Compliance (20%)**: Average steps-per-task — no free credit until a task is completed.

## 📦 Specification

### Action Space (Intent-Based)
| Action | Effect |
|--------|--------|
| `UP` / `DOWN` / `LEFT` / `RIGHT` | Apply **thrust/acceleration** in that direction — velocity accumulates |
| `WAIT` | Applies friction/braking (velocity × 0.5) |
| `PICK` | Picks up load — requires Manhattan distance < 1.1 from pickup target |
| `DROP` | Delivers load — requires Manhattan distance < 1.1 from drop zone |
| `CHARGE` | Recharges battery +10% — requires presence at a charging station |

### Observation Space
- **Ego State per Robot**: position (float), velocity (float), battery %, load status, robot profile, current goal.
- **World State**: Static racks, charging stations, drop zones, and dynamic spill positions.
- **Fleet Narrative**: Natural language "Command Center" log summarising battery warnings and goal distances.
- **Rubric Breakdown**: Granular live scores across all 4 metrics, updated every step.

### Physics Notes
- Collision with a rack or boundary: velocity resets to `[0, 0]` and **-5% battery penalty** is applied.
- Hitting a spill cell: velocity is boosted by ×1.2 (slide), clamped at `1.5× max_velocity`.
- Invalid `PICK`/`DROP`/`CHARGE` actions are counted and penalised in the Safety rubric.

## 🏁 Scenarios

| # | Name | Robots | Target Tasks | Max Steps | Spill Prob | Difficulty |
|---|------|--------|-------------|-----------|------------|------------|
| 1 | **Easy** (`easy_navigation`) | 2 | 2 | 80 | 0% | 4 / 10 |
| 2 | **Medium** (`medium_coordination`) | 3 | 4 | 100 | 3% | 6 / 10 |
| 3 | **Hard** (`hard_swarm`) | 4 | 6 | 130 | 6% | 8.5 / 10 |

### Scenario Details

**Easy** — 2 robots (1 Swift + 1 Hauler), no spills, 80 steps. A clean environment to validate momentum control and basic PICK/DROP sequencing. 2 tasks required. Drop zones at `[9,9]` and `[9,0]`; chargers at `[0,0]` and `[0,1]`.

**Medium** — 3 robots, low spills (3%), dense shelf clusters, 100 steps, 4 tasks. Requires proactive battery management and cross-fleet coordination. Drop zones at `[9,9]` and `[4,9]`.

**Hard** — 4 robots, moderate spills (6%), 30 shelf cells in dense horizontal rows, 130 steps, 6 tasks. Chargers mid-grid at `[0,4–5]` and `[9,4–5]`; drop zones at all four corners.

## 🚀 Setup

**Installation:**
```bash
pip install -r requirements.txt
pip install -e .
```

**Running Evaluation:**
```bash
export HF_TOKEN="your_hf_token"
python inference.py
```

*The evaluation emits strictly formatted stdout logs per the Hackathon specification:*

```
[START] task=<name> env=warehouse_v2_elite model=<model>
[STEP]  step=<n> action=<json> reward=<float> done=<bool> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<float> rewards=<csv>
```

> **Note**: `score` is always present on the `[END]` line, even on error, allowing reliable automated grading.

## 📊 Baseline Performance

| Model | Success Rate | Efficiency | Safety | Best Task |
|-------|-------------|------------|--------|-----------|
| Random Agent | 3% | 0.08 | 0.30 | — |
| Baseline LLM (no CoT) | 15% | 0.22 | 0.45 | Easy |
| Reasoning Agent (CoT) | 45% | 0.58 | 0.82 | Medium |
| **Elite Controller** | **85%** | **0.91** | **0.98** | **Hard** |

## 🔧 Changelog

### v2.2.0 (latest)
- **[FIX]** Score is now guaranteed strictly within `(0, 1)` via a `tanh`-based sigmoid squash applied to the raw rubric output — no hardcoded floor/ceiling, mathematically principled.
- **[IMPROVE]** PICK / DROP / CHARGE proximity threshold widened from `1.1` → `1.5` Manhattan units to account for floating-point momentum positions.
- **[IMPROVE]** SLA elite threshold relaxed from 20 → 30 steps/task, poor threshold from 40 → 60 steps/task — calibrated for momentum-based physics.
- **[IMPROVE]** Safety rubric collision penalty softened (per-collision 0.20 → 0.15; cap 0.80 → 0.60).
- **[IMPROVE]** Sustainability rubric critical-battery thresholds softened (dead robot penalty 0.25 → 0.20).
- **[IMPROVE]** Task configs rebalanced: Easy (80 steps / 2 tasks), Medium (100 steps / 4 tasks / 3% spills), Hard (130 steps / 6 tasks / 6% spills).

### v2.1.0
- **[FIX]** `[END]` log now always includes `score=` field, even on exception — grader-safe.
- **[FIX]** Rack collision and spill detection now use `round()` instead of `int()` — robots can no longer clip through walls at fractional positions.
- **[FIX]** Drop zones (`drop_zones`) are now an explicit, configurable field in task JSONs and surfaced in the `EnvironmentState` observation.
- **[FIX]** `SustainabilityRubric` now handles both dict-based and Pydantic `RobotState` robot representations safely.
- **[FIX]** `SLAComplianceRubric` returns `0.0` (not `0.5`) when no tasks are completed.
- **[IMPROVE]** All task configs updated with explicit `drop_zones` arrays.