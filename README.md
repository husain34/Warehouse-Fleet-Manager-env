---
title: Warehouse Fleet Management
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Warehouse Fleet Management: Elite Swarm Intelligence

This environment simulates a real-world warehouse logistics scenario where a swarm of autonomous robots must coordinate to fulfill delivery tasks. It is designed to evaluate long-horizon reasoning, multi-agent coordination, and safety under dynamic constraints.

## 🛠 Elite Features (OpenEnv v2.0)

- **Narrative-Driven Reasoning**: Unlike raw coordinate grids, this environment generates a **Strategic Narrative** every step. This natural language summary describes exactly what each robot is doing, its current progress, and any environmental hazards, significantly reducing spatial hallucinations in LLMs.
- **Multi-Dimensional Rubric**: Performance is evaluated using a `WeightedSum` rubric:
  - **Success (60%)**: Task completion.
  - **Efficiency (20%)**: Optimal pathing and step counts.
  - **Safety (20%)**: Avoiding environmental spills and collisions.
- **Full OpenEnv Compliance**: Perfectly adheres to the OpenEnv specification with typed Pydantic models for step/reset/state.

## 📦 Environment Definition

### Action Space
- `UP`, `DOWN`, `LEFT`, `RIGHT`: Move the robot 1 unit.
- `PICK`: Pick up an item at a designated shelf.
- `DROP`: Drop an item at a designated shelf.
- `WAIT`: Stationary.

### Observation Space
- `robots`: Detailed state of each robot (position, picked status, battery, last error).
- `environment`: Layout of shelves, charging stations, and dynamic spills.
- `narrative`: Natural language summary of the scene.
- `rubric_scores`: Real-time breakdown of performance across metrics.

## 🏁 Tasks

1. **Easy (`easy_navigation`)**: 2 Robots, 3 Tasks. No dynamic hazards. Focus on basic navigation.
2. **Medium (`medium_coordination`)**: 3 Robots, 5 Tasks. 5% Spill probability. Focus on yielding and collision avoidance.
3. **Hard (`hard_swarm`)**: 4 Robots, 8 Tasks. 10% Spill probability. High congestion requires sophisticated swarm intelligence.

## 🚀 Setup and Usage

**Installation:**
```bash
pip install -r requirements.txt
pip install -e .
```

**Running Inference:**
```bash
export HF_TOKEN="your_hf_token"
python inference.py
```

*Note: The script emits structured stdout logs strictly following the `[START]`, `[STEP]`, and `[END]` format required by the Meta OpenEnv Hackathon.*

## 📊 Baseline Scores

| Model / Agent | Easy | Medium | Hard |
|--------------|------|--------|------|
| Random Agent | 0.12 | 0.05   | 0.00 |
| Llama-3 (8B) | 0.85 | 0.62   | 0.35 |
| Llama-3 (70B)| 0.95 | 0.81   | 0.58 |