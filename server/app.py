from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
from typing import Optional

from warehouse_env import WarehouseOpenEnv
from models import Action

app = FastAPI()

ENV = None
TASKS_PATH = "tasks"

def load_task(task_name):
    path = os.path.join(TASKS_PATH, f"{task_name}.json")
    with open(path, "r") as f:
        return json.load(f)

@app.get("/")
def root():
    return {"status": "Elite Warehouse Env Running", "version": "2.0.0"}

@app.get("/status")
def status():
    return {"status": "Elite Warehouse Env Running"}

@app.post("/reset")
def reset(task_id: Optional[str] = "easy_navigation"):
    global ENV
    # Handle both filename and logical name
    task_map = {
        "easy_navigation": "easy",
        "medium_coordination": "medium",
        "hard_swarm": "hard"
    }
    file_key = task_map.get(task_id, "easy")
    
    try:
        config = load_task(file_key)
    except FileNotFoundError:
        config = load_task("easy")
        
    ENV = WarehouseOpenEnv(config)
    obs = ENV.reset()
    return obs.model_dump()

@app.post("/step")
def step(action: Action):
    global ENV
    if ENV is None:
        return {"error": "Environment not initialized. Call /reset first."}
    
    obs, reward, done, info = ENV.step(action)
    
    return {
        "observation": obs.model_dump(),
        "reward": reward.total,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    global ENV
    if ENV is None:
        return {"error": "No active environment"}
    return ENV.state().model_dump()

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()