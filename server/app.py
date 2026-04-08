from fastapi import FastAPI
from pydantic import BaseModel
import json
import os

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


from typing import Optional

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
    return ENV.state().model_dump()

import uvicorn

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()