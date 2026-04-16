import gradio as gr
import json
import os
import time
from warehouse_env import WarehouseOpenEnv, Action
from inference import get_llm_action

# --- Constants & Config ---
TASKS_DIR = "tasks"

def load_task_config(task_name):
    path = os.path.join(TASKS_DIR, f"{task_name}.json")
    with open(path, "r") as f:
        return json.load(f)

# --- CSS for Premium UI ---
CSS = """
.container { max-width: 1200px; margin: auto; }
.grid-container {
    display: grid;
    grid-template-columns: repeat(10, 1fr);
    grid-template-rows: repeat(10, 1fr);
    gap: 4px;
    background: #1a1a1a;
    padding: 10px;
    border-radius: 12px;
    aspect-ratio: 1 / 1;
    border: 2px solid #333;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}
.cell {
    background: #2a2a2a;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    position: relative;
    transition: all 0.2s ease;
}
.cell.shelf { background: #3d3d3d; border: 1px solid #555; }
.cell.station { background: #2d4a2d; border: 1px solid #4ade80; }
.cell.spill { background: #2d2d4a; }
.cell.spill::after { content: '💧'; font-size: 1rem; opacity: 0.6; }

.robot {
    z-index: 10;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    filter: drop-shadow(0 0 5px rgba(255,255,255,0.3));
}
.robot-r1 { color: #60a5fa; }
.robot-r2 { color: #f87171; }
.robot-r3 { color: #fbbf24; }

.robot-label {
    position: absolute;
    top: -5px;
    left: -5px;
    font-size: 0.6rem;
    background: rgba(0,0,0,0.8);
    padding: 2px 4px;
    border-radius: 4px;
    color: white;
}

.log-console {
    font-family: 'Courier New', Courier, monospace;
    background: #0d0d0d;
    color: #00ff00;
    padding: 15px;
    border-radius: 8px;
    height: 400px;
    overflow-y: auto;
    border: 1px solid #333;
}

.metric-card {
    background: #1e1e1e;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    border: 1px solid #333;
}
.metric-value { font-size: 1.5rem; font-weight: bold; color: #4ade80; }
"""

def render_grid(obs):
    grid_html = '<div class="grid-container">'
    
    # 10x10 grid, build from bottom-up or top-down?
    # Coordinates in env: [0,0] is bottom-left.
    # HTML grid: top-left is first.
    # So Y=9 is top, Y=0 is bottom.
    
    robots = obs.robots
    obstacles = obs.environment.obstacles
    spills = obs.environment.spills
    stations = obs.environment.charging_stations
    
    # Map robots to positions
    robot_pos_map = {tuple(r.pos): (rid, r) for rid, r in robots.items()}
    obstacle_set = set(tuple(o) for o in obstacles)
    spill_set = set(tuple(s) for s in spills)
    station_set = set(tuple(st) for st in stations)

    for y in range(9, -1, -1):
        for x in range(10):
            cell_classes = ["cell"]
            content = ""
            
            pos = (x, y)
            if pos in obstacle_set:
                cell_classes.append("shelf")
                content = "📦"
            elif pos in station_set:
                cell_classes.append("station")
                content = "⚡"
            elif pos in spill_set:
                cell_classes.append("spill")
            
            if pos in robot_pos_map:
                rid, r = robot_pos_map[pos]
                icon = "🤖" if not r.picked else "🦾"
                content = f'<div class="robot robot-{rid}">{icon}<span class="robot-label">{rid}</span></div>'
            
            grid_html += f'<div class="{" ".join(cell_classes)}">{content}</div>'
    
    grid_html += '</div>'
    return grid_html

def run_simulation(task_name):
    config = load_task_config(task_name)
    env = WarehouseOpenEnv(config)
    obs = env.reset()
    
    step_idx = 0
    done = False
    logs = []
    
    # Initial render
    yield render_grid(obs), "\n".join(logs), f"{obs.task_info.completed_tasks}/{config['target_tasks']}", "0.00", "0.00"

    while not done and step_idx < 100:
        step_idx += 1
        
        # Get LLM action
        action_dict, reasoning = get_llm_action(obs.model_dump())
        
        # Add to logs
        log_entry = f"--- Step {step_idx} ---\nREASONING: {reasoning}\nACTIONS: {json.dumps(action_dict)}\n"
        logs.insert(0, log_entry)
        
        # Step env
        action_input = Action(actions=action_dict)
        obs, reward, done, info = env.step(action_input)
        
        grade = info.get("grade", env.env.get_grade())
        
        yield (
            render_grid(obs), 
            "\n".join(logs), 
            f"{obs.task_info.completed_tasks}/{config['target_tasks']}", 
            f"{reward.total:.2f}",
            f"{grade:.2f}"
        )
        
        time.sleep(0.5) # Speed for visualization

# --- UI Layout ---
with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏢 Warehouse Fleet Manager: Swarm Intelligence")
    gr.Markdown("Watch the LLM coordinate multiple robots to complete tasks in real-time.")
    
    with gr.Row():
        with gr.Column(scale=2):
            grid_viewer = gr.HTML(label="Warehouse Grid")
            
        with gr.Column(scale=1):
            task_select = gr.Dropdown(
                choices=["easy", "medium", "hard"], 
                value="easy", 
                label="Select Task"
            )
            play_btn = gr.Button("▶️ Run Simulation", variant="primary")
            
            with gr.Row():
                tasks_done = gr.Textbox(label="Tasks", value="0/0", interactive=False)
                step_reward = gr.Textbox(label="Last Reward", value="0.00", interactive=False)
                total_grade = gr.Textbox(label="Current Grade", value="0.00", interactive=False)
            
            log_viewer = gr.Textbox(
                label="LLM Intelligence Log", 
                lines=15, 
                max_lines=20, 
                interactive=False, 
                autoscroll=True
            )

    play_btn.click(
        run_simulation, 
        inputs=[task_select], 
        outputs=[grid_viewer, log_viewer, tasks_done, step_reward, total_grade]
    )

    # Initial state
    demo.load(lambda: [render_grid(WarehouseOpenEnv(load_task_config("easy")).reset()), "", "0/0", "0.00", "0.00"], 
              None, 
              [grid_viewer, log_viewer, tasks_done, step_reward, total_grade])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
