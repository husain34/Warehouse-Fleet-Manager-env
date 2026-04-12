from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
from typing import Optional

from warehouse_env import WarehouseOpenEnv, Action

app = FastAPI()

ENV = None
TASKS_PATH = "tasks"

def load_task(task_name):
    path = os.path.join(TASKS_PATH, f"{task_name}.json")
    with open(path, "r") as f:
        return json.load(f)

@app.get("/")
def root():
    return {"status": "Warehouse Env Running"}

@app.get("/status")
def status():
    return {"status": "Warehouse Env Running"}

@app.post("/reset")
def reset(task_name: Optional[str] = "easy"):
    global ENV
    try:
        config = load_task(task_name)
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
    
    if done:
        info["rubric_breakdown"] = obs.rubric_scores
        
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

import uvicorn

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()