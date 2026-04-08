import random
from typing import Dict, List, Optional
from pydantic import BaseModel, field_validator

# =========================
# 🔹 Pydantic Schemas
# =========================

VALID_ACTIONS = {"UP", "DOWN", "LEFT", "RIGHT", "PICK", "DROP", "WAIT"}

class Action(BaseModel):
    actions: Dict[str, str]

    @field_validator("actions")
    @classmethod
    def validate_actions(cls, v):
        for rid, act in v.items():
            if act not in VALID_ACTIONS:
                raise ValueError(f"Invalid action '{act}' for {rid}")
        return v


class Reward(BaseModel):
    total: float


class RobotState(BaseModel):
    pos: List[int]
    goal: List[int]
    picked: bool
    battery: int
    status: str
    last_action: Optional[str] = None


class EnvironmentState(BaseModel):
    obstacles: List[List[int]]
    spills: List[List[int]]
    charging_stations: List[List[int]]


class DynamicState(BaseModel):
    occupied_cells: List[List[int]]
    congestion: Dict[str, float]


class TaskInfo(BaseModel):
    remaining_tasks: int
    completed_tasks: int


class StepInfo(BaseModel):
    step_count: int
    max_steps: int


class Observation(BaseModel):
    grid_size: List[int]
    robots: Dict[str, RobotState]
    environment: EnvironmentState
    dynamic: DynamicState
    task_info: TaskInfo
    step_info: StepInfo
    last_action_error: Optional[str]


# =========================
# 🔹 Core Logic (UNCHANGED)
# =========================

class WarehouseEnv:
    def __init__(self, config):
        self.size = 10
        self.step_count = 0

        self.config = config

        self.max_steps = config["max_steps"]
        self.num_robots = config["num_robots"]
        self.spill_prob = config["spill_prob"]

        self.charging_stations = config["charging_stations"]
        self.shelves = config["shelves"]

        self.target_tasks = config["target_tasks"]
        self.completed_tasks = 0

        self.robots = {}
        self.obstacles = []
        self.spills = []

        self._initialize_elements()
        self._spawn_robots()

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_empty_cell(self):
        all_cells = [[x, y] for x in range(self.size) for y in range(self.size)]
        
        occupied = set(tuple(o) for o in (
            self.obstacles +
            self.charging_stations +
            self.shelves +
            [s["pos"] for s in self.spills] +
            [r["pos"] for r in self.robots.values()]
        ))

        free_cells = [cell for cell in all_cells if tuple(cell) not in occupied]

        if not free_cells:
            raise Exception("No free cells available!")

        return random.choice(free_cells)

    def _get_valid_adjacent(self, target_pos):
        x, y = target_pos
        candidates = [[x, y+1], [x, y-1], [x+1, y], [x-1, y]]

        occupied = (
            self.obstacles +
            self.shelves +
            self.charging_stations +
            [s["pos"] for s in self.spills] +
            [r["pos"] for r in self.robots.values()]
        )

        valid = [
            c for c in candidates
            if 0 <= c[0] < self.size and 0 <= c[1] < self.size and c not in occupied
        ]

        return random.choice(valid) if valid else self._get_empty_cell()

    def _assign_task(self, robot):
        pickup_shelf = random.choice(self.shelves)
        dropoff_shelf = random.choice([s for s in self.shelves if s != pickup_shelf])
        
        robot["pickup_target"] = self._get_valid_adjacent(pickup_shelf)
        robot["dropoff_target"] = self._get_valid_adjacent(dropoff_shelf)
        robot["goal"] = robot["pickup_target"]
        
        # NEW: track efficiency
        robot["task_start_step"] = self.step_count
        robot["delivery_steps"] = 0

    def _initialize_elements(self):
        self.obstacles = []  # No random obstacles

    def _spawn_robots(self):
        for i in range(1, self.num_robots + 1):
            self.robots[f"r{i}"] = {
                "pos": self._get_empty_cell(),
                "picked": 0,
                "battery": 100,
                "total_reward": 0,
                "charge_reward_eligible": True,
                "delivery_steps": 0,
                "last_action": None,
                "task_start_step": 0
            }
            self._assign_task(self.robots[f"r{i}"])

    def _update_spills(self):
        active_spills = []
        for s in self.spills:
            s["timer"] -= 1
            if s["timer"] > 0:
                active_spills.append(s)
        self.spills = active_spills

        goal_positions = []
        for r in self.robots.values():
            goal_positions.append(r["pickup_target"])
            goal_positions.append(r["dropoff_target"])

        if random.random() < self.spill_prob:
            while True:
                pos = self._get_empty_cell()
                if pos not in goal_positions:
                    self.spills.append({
                        "pos": pos,
                        "timer": random.randint(5, 6)
                    })
                    break

    def _calculate_congestion(self):
        congestion_data = []
        robot_positions = [r["pos"] for r in self.robots.values()]
        for pos in robot_positions:
            nearby = sum(
                1 for other in robot_positions
                if other != pos and abs(pos[0]-other[0]) + abs(pos[1]-other[1]) <= 2
            )
            value = min(1.0, nearby * 0.3)
            congestion_data.append([pos[0], pos[1], value])
        return congestion_data

    def get_observation(self):
        return {
            "robots": self.robots,
            "obstacles": self.obstacles,
            "spills": [s["pos"] for s in self.spills],
            "charging_stations": self.charging_stations,
            "occupied": [r["pos"] for r in self.robots.values()],
            "congestion": self._calculate_congestion(),
            "step_count": self.step_count,
            "max_steps": self.max_steps
        }

    def apply_step(self, actions):
        self.step_count += 1
        self._update_spills()
        step_rewards = {rid: 0.0 for rid in self.robots}

        # ✅ FIXED: Pre-populate intended_moves (WAIT bug fix)
        intended_moves = {
            rid: list(r["pos"]) for rid, r in self.robots.items()
        }
        
        step_errors = []

        for rid, act in actions.items():
            if rid not in self.robots or self.robots[rid]["battery"] <= 0:
                continue
            
            pos = list(self.robots[rid]["pos"])
            target = list(pos)
            if act == "UP": target[1] += 1
            elif act == "DOWN": target[1] -= 1
            elif act == "LEFT": target[0] -= 1
            elif act == "RIGHT": target[0] += 1
            
            if target[0] < 0 or target[0] >= self.size or target[1] < 0 or target[1] >= self.size:
                step_errors.append(f"{rid} blocked by boundary")
                target = list(pos)
                
            intended_moves[rid] = target

        for rid, act in actions.items():
            if rid not in self.robots or self.robots[rid]["battery"] <= 0:
                continue
            
            robot = self.robots[rid]
            robot["last_action"] = act
            old_pos = list(robot["pos"])
            reward = -0.05

            if robot["battery"] < 40:
                robot["charge_reward_eligible"] = True

            # previous + new position comparison (for shaping)
            prev_pos = old_pos
            target_pos = intended_moves[rid]
            new_pos = target_pos

            goal = robot["goal"]

            prev_dist = self._manhattan(prev_pos, goal)
            new_dist = self._manhattan(new_pos, goal)

            # =========================
            # 🧠 DENSE SHAPING REWARD
            # =========================
            if act in ["UP", "DOWN", "LEFT", "RIGHT"]:
                if target_pos == old_pos:
                    reward -= 3.0   # boundary hit (clamped)
                elif target_pos in self.obstacles or target_pos in self.shelves:
                    reward -= 3.0   # obstacle / shelf hit
                    step_errors.append(f"{rid} hit obstacle/shelf")
                else:
                    crash = False

                    for other_rid, other_pos in intended_moves.items():
                        if other_rid != rid and target_pos == other_pos:
                            crash = True
                            step_errors.append(f"{rid} collided with {other_rid}")

                    if crash:
                        reward -= 4.0   # collision with other robot
                    else:
                        robot["pos"] = target_pos

                        # 🔥 MOVEMENT SHAPING
                        if new_dist < prev_dist:
                            reward += 0.1
                        else:
                            reward -= 0.05
            # =========================
            # PICK LOGIC
            # =========================
            elif act == "PICK":
                if (robot["pos"] == robot["pickup_target"]) and (robot["picked"] == 0):
                    robot["picked"] = 1
                    robot["goal"] = robot["dropoff_target"]
                    reward += 6.0   # reward for pickup

            # =========================
            # DROP LOGIC
            # =========================
            elif act == "DROP":
                if (robot["pos"] == robot["dropoff_target"]) and (robot["picked"] == 1):
                    robot["picked"] = 0
                    self.completed_tasks += 1

                    # =========================
                    # ⏱ SPEED BONUS (NEW)
                    # =========================
                    steps_taken = self.step_count - robot.get("task_start_step", self.step_count)
            
                    speed_bonus = max(0.0, 15.0 - steps_taken * 0.3)
                    delay_penalty = min(8.0, steps_taken * 0.25)

                    drop_reward = 12.0 + speed_bonus - delay_penalty
                    drop_reward = min(25.0, max(5.0, drop_reward))

                    reward += drop_reward

                    # assign new task
                    self._assign_task(robot)

                else:
                    reward -= 4.0    # invalid drop penalty

            # =========================
            # WAIT LOGIC
            # =========================
            elif act == "WAIT":
                reward -= 0.1

            # ⚠️ SPILL COLLISION PENALTY
            if robot["pos"] in [s["pos"] for s in self.spills]:
                reward -= 6.0

            # 🚦 CONGESTION penalty
            for _, _, val in self._calculate_congestion():
                reward -= val * 0.2

            robot["total_reward"] += reward
            step_rewards[rid] = float(round(reward, 2))

        return self.get_observation(), step_rewards, step_errors

    def get_grade(self):
        return min(1.0, float(self.completed_tasks) / float(self.target_tasks))


# =========================
# 🔹 Wrapper
# =========================

class WarehouseOpenEnv:
    def __init__(self, config):
        self.config = config
        self.env = WarehouseEnv(config)
        self.last_error = None

    def _convert_obs(self):
        core = self.env.get_observation()

        robots = {}
        for rid, r in self.env.robots.items():
            status = "active" if r["battery"] > 0 else "failed"
            robots[rid] = RobotState(
                pos=r["pos"],
                goal=r["goal"],
                picked=bool(r["picked"]),
                battery=r["battery"],
                status=status,
                last_action=r.get("last_action")
            )

        congestion_dict = {
            str([x, y]): val for x, y, val in core["congestion"]
        }

        return Observation(
            grid_size=[self.env.size, self.env.size],
            robots=robots,
            environment=EnvironmentState(
                obstacles=core["obstacles"] + self.env.shelves,
                spills=core["spills"],
                charging_stations=core["charging_stations"]
            ),
            dynamic=DynamicState(
                occupied_cells=core["occupied"],
                congestion=congestion_dict
            ),
            task_info=TaskInfo(
                remaining_tasks=self.env.target_tasks - self.env.completed_tasks,
                completed_tasks=self.env.completed_tasks
            ),
            step_info=StepInfo(
                step_count=core["step_count"],
                max_steps=core["max_steps"]
            ),
            last_action_error=self.last_error
        )

    def reset(self):
        self.env = WarehouseEnv(self.config)
        self.last_error = None
        return self._convert_obs()

    def step(self, action: Action):
        try:
            actions_dict = {
                rid: action.actions.get(rid, "WAIT")
                for rid in self.env.robots
            }

            _, rewards, step_errors = self.env.apply_step(actions_dict)
            total_reward = float(sum(rewards.values()))
            reward_obj = Reward(total=total_reward)
            
            if step_errors:
                self.last_error = "; ".join(step_errors)
            else:
                self.last_error = None

        except ValueError as e:
            self.last_error = str(e)
            return self._convert_obs(), Reward(total=0.0), False, {}

        done = (
            self.env.step_count >= self.env.max_steps or
            self.env.completed_tasks >= self.env.target_tasks
        )

        info = {}
        if done:
            info["grade"] = self.env.get_grade()

        return self._convert_obs(), reward_obj, done, info

    def state(self):
        return self._convert_obs()

    def close(self):
        pass