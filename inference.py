import os
import json
import ast
import traceback
import re
from openai import OpenAI
from warehouse_env import WarehouseOpenEnv
from models import Action

# Read from environment variables (required by hackathon rules)
API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "openai/gpt-oss-20b")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize the OpenAI Client using HF_TOKEN as the key
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def safe_parse_actions(text):
    try:
        text = text.strip()
        if "```" in text:
            blocks = re.findall(r"```(?:json)?(.*?)```", text, re.DOTALL)
            if blocks:
                text = blocks[0].strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)
        data = json.loads(text)
        return data.get("actions", data)
    except Exception as e:
        print(f"Parsing error: {e}")
        return {}

def get_llm_action(obs_data):
    """Sends the current elite warehouse state to the LLM."""

    BASE_PROMPT = """
You are the **Lead Fleet Controller** for a high-fidelity industrial warehouse. 
Objective: Maximize throughput and sustainability while ensuring operational safety.

=========================
🚀 ELITE KINEMATICS PROTOCOL
=========================
Racks at x=[2,5,8] create defined aisles. Navigation requires precision.
1. **Momentum Management**: Cardinals (UP, DOWN, LEFT, RIGHT) now represent **THRUST**. Each move increases velocity in that direction. 
2. **Braking**: To stop or slow down, use `WAIT` (applies friction) or thrust in the OPPOSITE direction of movement.
3. **Inertia**: If velocity is high, you will overshoot cells. Plan turns in advance.
4. **Energy Management**: Moving at high speeds or carrying loads drains battery significantly. Use `CHARGE` at stations [0,0] when battery < 20%.

=========================
🏗️ HETEROGENEOUS FLEET
=========================
- **Swift**: Fast acceleration, high max speed, but high battery drain. Ideal for long-range scouting.
- **Hauler**: Slower, but more energy-efficient. Ideal for heavy load transport.

=========================
📤 OUTPUT FORMAT (STRICT JSON ONLY)
=========================
{
  "reasoning": "r1 (Swift) approaching Bay 4. Velocity [0.5, 0]. Goal is at x=2. Reducing x-thrust and applying LEFT to brake. r2 (Hauler) battery at 15%. Executing CHARGE protocol at [0,0].",
  "actions": {
    "r1": "LEFT",
    "r2": "CHARGE"
  }
}
"""

    obs_narrative = obs_data.get("narrative", "No narrative available.")
    prompt = f"{BASE_PROMPT}\n\nSTRATEGIC NARRATIVE:\n{obs_narrative}\n\nRAW DATA:\n{json.dumps(obs_data)}"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": BASE_PROMPT},
                {"role": "user", "content": f"Observation:\n{json.dumps(obs_data)}"}
            ],
            temperature=0.1,
            stream=True
        )

        full_content = ""
        for chunk in completion:
            if not getattr(chunk, "choices", None): continue
            content = chunk.choices[0].delta.content
            if content is not None: full_content += content

        return safe_parse_actions(full_content)

    except Exception as e:
        print(f"LLM Error: {e}")
        return {}

def run_benchmark():
    task_files = [
        ("easy_navigation", "tasks/easy.json"),
        ("medium_coordination", "tasks/medium.json"),
        ("hard_swarm", "tasks/hard.json")
    ]

    for name, filepath in task_files:
        if not os.path.exists(filepath): continue
        with open(filepath, "r") as f:
            config = json.load(f)
            config["name"] = name
        
        env = WarehouseOpenEnv(config)
        obs = env.reset()
        
        print(f"[START] task={config['name']} env=warehouse_v2_elite model={MODEL_NAME}", flush=True)
        
        rewards = []
        try:
            done, step_idx = False, 0
            while not done and step_idx < 150: # Increased cap for kinematics complexity
                step_idx += 1
                
                action_dict = get_llm_action(obs.model_dump())
                action_input = Action(actions=action_dict)
                
                obs, reward, done, info = env.step(action_input)
                rewards.append(reward.total)
                
                err_msg = obs.last_action_error if obs.last_action_error else "null"
                action_str = json.dumps(action_dict, separators=(',', ':'))
                print(f"[STEP] step={step_idx} action={action_str} reward={reward.total:.2f} done={str(done).lower()} error={err_msg}", flush=True)

            score = float(info.get("grade", 0.0))
            success_str = "true" if score >= 0.8 else "false"
            rewards_str = ",".join([f"{r:.2f}" for r in rewards])
            print(f"[END] success={success_str} steps={step_idx} score={score:.3f} rewards={rewards_str}", flush=True)

        except Exception as e:
            print(f"[ERROR]: {e}")
            print(f"[END] success=false steps={step_idx} rewards=0.00", flush=True)
        finally:
            env.close()

if __name__ == "__main__":
    run_benchmark()