from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum

class ActionType(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    PICK = "PICK"
    DROP = "DROP"
    WAIT = "WAIT"
    CHARGE = "CHARGE" # New Elite Action

class RobotProfileType(str, Enum):
    SWIFT = "Swift"  # Faster, low capacity, high energy usage
    HAULER = "Hauler" # Slower, high capacity, low energy usage

class RobotProfile(BaseModel):
    name: RobotProfileType
    max_velocity: float
    acceleration: float
    battery_drain_rate: float
    load_penalty: float # multiplier for drain when carrying items

class RobotState(BaseModel):
    pos: List[float] = Field(..., description="[x, y] coordinates")
    velocity: List[float] = Field(default=[0.0, 0.0], description="[vx, vy] velocity")
    goal: List[int]
    picked: bool
    current_load: float = 0.0
    battery: float = 100.0
    status: str
    last_action: Optional[str] = None
    profile: RobotProfileType

class EnvironmentState(BaseModel):
    obstacles: List[List[int]]
    spills: List[List[int]]
    charging_stations: List[List[int]]
    racks: List[List[int]] = Field(default=[], description="Static warehouse racks")

class DynamicState(BaseModel):
    occupied_cells: List[List[int]]
    congestion: Dict[str, float]
    predicted_path_collisions: List[str] = []

class TaskInfo(BaseModel):
    remaining_tasks: int
    completed_tasks: int
    system_throughput: float = 0.0

class StepInfo(BaseModel):
    step_count: int
    max_steps: int

class Action(BaseModel):
    actions: Dict[str, str]

    @field_validator("actions")
    @classmethod
    def validate_actions(cls, v):
        valid_acts = {a.value for a in ActionType}
        for rid, act in v.items():
            if act not in valid_acts:
                raise ValueError(f"Invalid action '{act}' for {rid}")
        return v

class Observation(BaseModel):
    grid_size: List[int]
    robots: Dict[str, RobotState]
    environment: EnvironmentState
    dynamic: DynamicState
    task_info: TaskInfo
    step_info: StepInfo
    last_action_error: Optional[str]
    narrative: str
    rubric_scores: Dict[str, float]

class Reward(BaseModel):
    total: float
