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

- **Kinematics Engine**: They operate with **inertia, acceleration, and friction**. Agents must manage momentum to navigate tight warehouse aisles without overshooting.
- **Heterogeneous Fleet**: Deploy a diverse swarm including:
    - **Swift**: Rapid response units with high acceleration but high power consumption.
    - **Hauler**: High-capacity industrial workhorses with superior energy efficiency.
- **Industrial Layout**: The environment features static **Rack Columns** that create defined "Aisles," forcing agents to master spatial reasoning and traffic management.
- **Strategic Command Narrative**: A sophisticated spatial advisory engine that translates complex coordinate data into natural language "Fleet Advisories" (e.g., *"Path to Bay 4 is congested by r2; reroute recommended"*).
- **Advanced Performance Telemetry**: Grading is performed via a 4-pillar **EliteWarehouseRubric**:
    - **Efficiency (40%)**: Task throughput and path optimality.
    - **Safety (20%)**: Collision risk and environmental spill avoidance.
    - **Sustainability (20%)**: Energy management and battery health.
    - **SLA Compliance (20%)**: Average steps per task completion.

## 📦 Specification

### Action Space (Intent-Based)
- `UP`, `DOWN`, `LEFT`, `RIGHT`: Apply **Thrust/Acceleration** in the specified direction.
- `PICK` / `DROP`: Semantic interaction with load bays.
- `CHARGE`: Mandatory docking at charging stations (requires precise stopping).
- `WAIT`: Applies friction/braking to halt momentum.

### Observation Space
- **Ego State**: Velocity, Position, Battery, Load Status, and Profile.
- **World State**: Static racks, dynamic spills, and charging station maps.
- **Fleet Narrative**: Natural language "Command Center" logs.
- **Rubric Breakdown**: Granular performance feedback across all 4 metrics.

## 🏁 Scenarios

1. **Easy (`easy_navigation`)**: 1 Robot, clear aisles, static targets. Focus on basic momentum control.
2. **Medium (`medium_coordination`)**: 3 Robots, heterogeneous swarm, dynamic spills. Focus on yielding and energy management.
3. **Hard (Elite) (`hard_swarm`)**: 6+ Robots, high-congestion racks, tight battery constraints, and "Rush Orders."

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

*The evaluation emits strictly formatted stdout logs: `[START]`, `[STEP]`, and `[END]` following the Hackathon specification.*

## 📊 Baseline Performance

| Model | Success Rate | Efficiency | Safety | Best Task |
|-------|--------------|------------|--------|-----------|
| Baseline LLM | 15% | 0.22 | 0.45 | Easy |
| Reasoning Agent | 45% | 0.58 | 0.82 | Medium |
| **Elite Controller** | **85%** | **0.91** | **0.98** | **Hard** |