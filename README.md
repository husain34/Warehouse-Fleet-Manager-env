---
title: Warehouse Fleet Management
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Warehouse Fleet Management Environment

## Environment Description & Motivation
This is an OpenEnv-compatible warehouse fleet management environment designed to accurately simulate real-world logistics and operations. The core motivation is to model real-world problem-solving scenarios by determining the most efficient paths to optimize traffic for autonomous robot swarms. By completing tasks as fast as possible while preventing collisions, anticipating bottlenecks, and tracking battery states, this environment serves to test and develop systems capable of managing complex, dynamic fleet routing.

## Action and Observation Space Definitions

**Action Space:**
The action space consists of a discrete set of commands assigned to each active robot in the swarm. The valid actions are:
- `UP`, `DOWN`, `LEFT`, `RIGHT`: Move the robot 1 unit in the specified direction.
- `PICK`: Pick up an item at a designated shelf.
- `DROP`: Drop an item at a designated shelf.
- `WAIT`: Keep the robot stationary in its current location.

Actions are submitted as a dictionary mapping each robot ID to its chosen action:
```json
{
  "actions": {
    "r1": "RIGHT",
    "r2": "UP",
    "r3": "PICK"
  }
}
```

**Observation Space:**
The environment provides a comprehensive JSON state representing grid conditions:
- `grid_size`: Dimensions of the warehouse grid (e.g., [10, 10]).
- `robots`: Dictionary of each robot's state (Position `pos`, target `goal`, inventory status `picked`, health `battery`, and `last_action_error` if blocked).
- `environment`: Static and semi-static layouts, including permanent `obstacles`, temporary `spills`, and `charging_stations`.
- `dynamic`: Real-time state such as `occupied_cells` and calculated continuous `congestion` maps.
- `task_info`: Count of `remaining_tasks` and `completed_tasks`.
- `step_info`: Current `step_count` vs `max_steps`.

## Task Descriptions & Expected Difficulty
The environment defines three difficulty tasks, configured via JSON blueprints in `/tasks`:

1. **Easy (`easy.json`)**
   - **Difficulty:** Low
   - **Environment:** 10x10 Grid, 50 Max Steps.
   - **Objective:** 2 Robots coordinating to complete 3 tasks.
   - **Challenges:** No dynamic spills (0% probability). The agent only needs minimal multi-agent reasoning to complete the tasks safely.

2. **Medium (`medium.json`)**
   - **Difficulty:** Moderate
   - **Environment:** 10x10 Grid, 75 Max Steps.
   - **Objective:** 3 Robots coordinating to complete 5 tasks.
   - **Challenges:** Introduces a 5% probability of dynamic spills. Agents must actively avoid random hazards while navigating slightly tighter spaces and yielding priority to avoid collisions.

3. **Hard (`hard.json`)**
   - **Difficulty:** High
   - **Environment:** 10x10 Grid, 100 Max Steps.
   - **Objective:** 4 Robots coordinating to complete 8 tasks.
   - **Challenges:** High congestion with 4 active robots, higher spill probability (10%), and frequent gridlock scenarios. Reaching optimal scores requires sophisticated swarm path calculation to prevent infinite loops and bottlenecks.

## Setup and Usage Instructions

**Prerequisites:**
- Python 3.9+
- Docker (optional, but supported for isolated environments via OpenEnv specification)

**Installation:**
1. Clone the repository to your local environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

**Usage via CLI Inference:**
To run the evaluation script against an LLM (such as Llama-3 or an OpenAI-compatible endpoint):
```bash
export API_BASE_URL="https://integrate.api.nvidia.com/v1" # Or custom LLM endpoint
export MODEL_NAME="openai/gpt-oss-20b"                    # Or target model
export HF_TOKEN="your_api_token"
python inference.py
```
This automatically runs through the Easy, Medium, and Hard benchmarks and logs the evaluation trace.

**Integration into OpenEnv:**
The environment wrapper `WarehouseOpenEnv` conforms to standard OpenEnv API specs (`/reset`, `/step`, `/state`). You can launch a FastAPI handler using standard OpenEnv validators.

## Baseline Scores
*Scores below are normalized grades (ranging strictly between 0 and 1) based on task completion ratio and step efficiency.*

| Model / Agent | Task: Easy | Task: Medium | Task: Hard |
|--------------|------------|--------------|------------|
| Random Agent | 0.12       | 0.05         | 0.00       |
| Llama-3 (8B) | 0.85       | 0.62         | 0.35       |
| Llama-3 (70B)| 0.95       | 0.81         | 0.58       |

*Note: Baseline outputs depend heavily on strict adherence to JSON output requirements and the anti-loop protocol. Stronger LLMs yield fewer invalid actions.*