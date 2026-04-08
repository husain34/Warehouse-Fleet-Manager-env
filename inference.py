import os
import json
import ast
import traceback
import re
from openai import OpenAI
from warehouse_env import WarehouseOpenEnv, Action

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

        # 1. Remove markdown blocks first (safe cleanup)
        if "```" in text:
            blocks = re.findall(r"```(?:json)?(.*?)```", text, re.DOTALL)
            if blocks:
                text = blocks[0].strip()

        # 2. Extract first valid JSON object (non-greedy)
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)

        # 3. Parse JSON
        data = json.loads(text)

        # 4. Return actions safely
        return data.get("actions", data)

    except Exception as e:
        print(f"Parsing error: {e} | Raw text: {text[:200]}")
        return {}

def get_llm_action(obs_data):
    """
    Sends the current warehouse state to the LLM and gets robot actions.
    """

    BASE_PROMPT = """
You are the **Swarm Intelligence Controller** for a warehouse fleet. Your objective is to maximize efficiency and total reward by completing PICK and DROP tasks without collisions.
Warehouse Layout: 10x10. Coordinates are [X, Y] where [0,0] is Bottom-Left. 
Shelves are PERMANENT OBSTACLES. Spills are TEMPORARY OBSTACLES.

=========================
🚨 CRITICAL FAILURES TO AVOID (STRICT RULES)
=========================
1. **THE ANTI-LOOP PROTOCOL**: You are stateless and prone to getting stuck in infinite loops. You MUST read `last_action_error`. 
   - If `last_action_error` says "{robot} blocked by boundary" or "{robot} hit obstacle/shelf", **YOU ARE STRICTLY FORBIDDEN from repeating the `last_action`.** You MUST shift to the perpendicular axis (e.g., if blocked going UP, you MUST choose LEFT or RIGHT).
   - If your `last_action` was DOWN, do NOT move UP this turn unless avoiding a collision. Sliding past obstacles is required.

2. **GRID AWARENESS [X, Y]**:
   - X axis (Left/Right): LEFT = X-1. RIGHT = X+1. Boundary: 0 to 9.
   - Y axis (Down/Up): DOWN = Y-1. UP = Y+1. Boundary: 0 to 9.
   - NEVER command LEFT if X=0, RIGHT if X=9, DOWN if Y=0, or UP if Y=9.

3. **SWARM COLLISION & YIELDING**:
   - Priority Order: r1 > r2 > r3. Higher ID robots MUST calculate their paths around lower ID robots.
   - **EVASION > WAITING**: If r1's intended next cell is r2's current cell, r2 MUST NOT WAIT. r2 MUST move to an adjacent free cell to avoid being rammed.
   - **CROSSING**: Do not let r2's next cell equal r1's next cell.

=========================
🧠 STEP-BY-STEP CALCULATION PROTOCOL
=========================
For EVERY active robot provided in the Observation's `robots` dictionary (and ONLY those robots), you must calculate the following in your reasoning:
1. **Target Action**: Check if `pos == goal`. If yes, output "PICK" (if picked=False) or "DROP" (if picked=True).
2. **Delta Calculation**: Calculate X_diff = goal[0] - pos[0], Y_diff = goal[1] - pos[1]. Choose the direction that reduces the largest difference.
3. **Safety Check**: Check the chosen direction against the 10x10 boundaries, the `occupied_cells`, `shelves`, and `obstacles`. 
4. **Error Check**: Does this direction match the `last_action_error`? If yes, change axis.
5. **Swarm Check**: Will this place the robot in a cell occupied by or targeted by a higher-priority robot?

**CRITICAL**: Do NOT output actions for robots that are not in the Observation. If you see only "r1" and "r2", you must NOT output an action for "r3" or "r4".

=========================
📤 OUTPUT FORMAT (STRICT JSON ONLY)
=========================
Return ONLY a valid JSON object. No markdown prose outside the JSON. Use the `reasoning` field to execute the 5-step protocol for EVERY robot so you do not hallucinate coordinates.

{
  "reasoning": "r1: pos [2,5], goal [8,5]. Target: RIGHT. Next: [3,5]. Valid cell, no errors. r2: pos [3,5], goal [3,9]. Target: UP, but r1 is targeting [3,5] which is r2's current pos. r2 MUST EVADE. R2 target changed to UP [3,6]. r3: pos [0,9], goal [5,9]. Target RIGHT. X+1=[1,9]. But last_action_error was 'r3 hit obstacle'. Forced to move DOWN [0,8].",
  "actions": {
    "r1": "RIGHT",
    "r2": "UP",
    "r3": "DOWN"
  }
}
"""

    prompt = f"{BASE_PROMPT}\n\nObservation:\n{json.dumps(obs_data)}"

    try:
        # Following your teammate's pattern: stream=True and handling reasoning_content
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": BASE_PROMPT},
                {"role": "user", "content": f"Observation:\n{json.dumps(obs_data)}"}
            ],
            temperature=0.2, # Lowered from 1.0 for more stable robotics logic
            stream=True
        )

        full_content = ""
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            
            # Capture reasoning if the model provides it (useful for debugging loops)
            reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
            if reasoning:
                # Optionally print reasoning to console to see the "thought process"
                pass 
            
            content = chunk.choices[0].delta.content
            if content is not None:
                full_content += content

        return safe_parse_actions(full_content)

    except Exception as e:
        print(f"LLM Error: {e}")
        traceback.print_exc()
        return {}  # fallback

def run_benchmark():
    # Define the three mandatory tasks: easy, medium, and hard
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
        
        # [START] line - Required by Meta Guidelines
        print(f"[START] task={config['name']} env=warehouse_v1 model={MODEL_NAME}", flush=True)
        
        rewards = []

        try:
            done, step_idx = False, 0
            while not done and step_idx < 100: # Added safety cap
                step_idx += 1
                
                # Get actions from LLM
                action_dict = get_llm_action(obs.model_dump())
                action_input = Action(actions=action_dict)
                
                # Step the environment
                obs, reward, done, info = env.step(action_input)
                rewards.append(reward.total)
                
                # [STEP] line - Required by Meta Guidelines
                err_msg = obs.last_action_error if obs.last_action_error else "null"
                # Strip spaces from JSON so the string doesn't break auto-grader space-splitters
                action_str = json.dumps(action_dict, separators=(',', ':'))
                print(f"[STEP] step={step_idx} action={action_str} reward={reward.total:.2f} done={str(done).lower()} error={err_msg}", flush=True)

            # [END] line - Required by Meta Guidelines
            score = info.get("grade", 0.0)
            success_str = "true" if score >= 1.0 else "false"
            rewards_str = ",".join([f"{r:.2f}" for r in rewards])
            print(f"[END] success={success_str} steps={step_idx} score={score:.2f} rewards={rewards_str}", flush=True)

        except Exception as e:
            # If the loop crashes, we MUST still print an [END] line
            print(f"[FATAL ERROR IN RUN_BENCHMARK]: {e}")
            traceback.print_exc()
            print(f"[END] success=false steps={step_idx} score=0.00 rewards=0.00", flush=True)
        finally:
            env.close()

if __name__ == "__main__":
    run_benchmark()