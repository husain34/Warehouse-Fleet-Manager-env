import random
import math
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel

from models import (
    Action, ActionType, RobotProfileType, RobotProfile, 
    RobotState, EnvironmentState, DynamicState, TaskInfo, 
    StepInfo, Observation, Reward
)
from rubrics import EliteWarehouseRubric

# =========================
# 🔹 Elite Core Logic
# =========================

PROFILES = {
    RobotProfileType.SWIFT: RobotProfile(
        name=RobotProfileType.SWIFT,
        max_velocity=1.5,
        acceleration=0.5,
        battery_drain_rate=0.05,
        load_penalty=1.5
    ),
    RobotProfileType.HAULER: RobotProfile(
        name=RobotProfileType.HAULER,
        max_velocity=0.8,
        acceleration=0.2,
        battery_drain_rate=0.02,
        load_penalty=1.2
    )
}

class WarehouseEnv:
    def __init__(self, config):
        self.size = 10
        self.step_count = 0
        self.config = config

        self.max_steps = config.get("max_steps", 100)
        self.num_robots = config.get("num_robots", 2)
        self.spill_prob = config.get("spill_prob", 0.05)

        self.charging_stations = config.get("charging_stations", [[0, 0]])
        self.shelves = config.get("shelves", [])
        # Drop zones: explicit delivery targets (distinct from charging stations)
        self.drop_zones = config.get("drop_zones", [[0, 9], [9, 0]])
        
        # Industrial Layout: Racks
        self.racks = self._generate_industrial_racks()

        self.target_tasks = config.get("target_tasks", 5)
        self.completed_tasks = 0

        self.robots = {}
        self.spills = []
        self.spill_hits = 0
        self.collision_count = 0
        self.invalid_action_count = 0

        self._spawn_robots()

    def _generate_industrial_racks(self):
        """Generates standard industrial rack columns."""
        racks = []
        for x in [2, 5, 8]: # Rack columns
            for y in range(1, 9): # Leave space at top/bottom for corridors
                racks.append([x, y])
        return racks

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_empty_cell(self):
        all_cells = [[float(x), float(y)] for x in range(self.size) for y in range(self.size)]
        
        occupied = set(tuple(o) for o in (
            self.racks +
            self.charging_stations +
            self.shelves +
            [s["pos"] for s in self.spills] +
            [r["pos"] for r in self.robots.values()]
        ))

        free_cells = [cell for cell in all_cells if tuple(map(int, cell)) not in occupied]
        if not free_cells:
            return [0.0, 0.0] # Fallback
        return random.choice(free_cells)

    def _spawn_robots(self):
        for i in range(1, self.num_robots + 1):
            profile_type = RobotProfileType.SWIFT if i % 2 == 1 else RobotProfileType.HAULER
            rid = f"r{i}"
            self.robots[rid] = {
                "pos": self._get_empty_cell(),
                "velocity": [0.0, 0.0],
                "picked": 0,
                "battery": 100.0,
                "total_reward": 0,
                "profile": profile_type,
                "last_action": None,
                "task_start_step": self.step_count,
                "current_load": 0.0
            }
            self._assign_task(self.robots[rid])

    def _assign_task(self, robot):
        # Prevent "Goal Farming": ensure goal is not too close to current position
        max_retries = 10
        pickup_loc = random.choice(self.racks)
        for _ in range(max_retries):
            candidate = random.choice(self.racks)
            if self._manhattan(robot["pos"], candidate) > 4.0:
                pickup_loc = candidate
                break
        
        dropoff_loc = random.choice(self.charging_stations + self.drop_zones)
        
        robot["goal"] = pickup_loc
        robot["pickup_target"] = pickup_loc
        robot["dropoff_target"] = dropoff_loc
        robot["task_start_step"] = self.step_count

    def generate_narrative(self):
        narrative = f"Status: Elite Warehouse Ops. Step {self.step_count}/{self.max_steps}. System Throughput: {self.completed_tasks} units. "
        
        for rid, r in self.robots.items():
            profile = PROFILES[r["profile"]]
            v_mag = math.sqrt(r["velocity"][0]**2 + r["velocity"][1]**2)
            narrative += f"[{rid} ({r['profile'].value})]: At {r['pos']}, Vel {v_mag:.1f}, Batt {r['battery']:.0f}%. "
            
            if r["battery"] < 20:
                narrative += "CRITICAL: Return to charger immediately! "
            
            dist = self._manhattan(r["pos"], r["goal"])
            task = "Pickup" if not r["picked"] else "Dispatch"
            narrative += f"Goal: {task} at {r['goal']} ({dist:.1f}m away). "

        if self.spills:
            narrative += f"Alert: Spills detected at {', '.join([str(s['pos']) for s in self.spills])}. Slippery conditions!"
            
        return narrative

    def _update_spills(self):
        self.spills = [s for s in self.spills if s["timer"] > 1]
        for s in self.spills: s["timer"] -= 1

        if random.random() < self.spill_prob:
            pos = [random.randint(0, 9), random.randint(0, 9)]
            if pos not in self.racks:
                self.spills.append({"pos": pos, "timer": random.randint(5, 10)})

    def apply_step(self, actions):
        self.step_count += 1
        self._update_spills()
        step_rewards = {rid: 0.0 for rid in self.robots}
        step_errors = []

        for rid, act in actions.items():
            if rid not in self.robots: continue
            robot = self.robots[rid]
            profile = PROFILES[robot["profile"]]
            
            if robot["battery"] <= 0:
                step_errors.append(f"{rid} power failure")
                continue

            robot["last_action"] = act
            
            # 1. Physics: Velocity Update
            acc = profile.acceleration
            if act == ActionType.UP: robot["velocity"][1] += acc
            elif act == ActionType.DOWN: robot["velocity"][1] -= acc
            elif act == ActionType.LEFT: robot["velocity"][0] -= acc
            elif act == ActionType.RIGHT: robot["velocity"][0] += acc
            elif act == ActionType.WAIT:
                # Friction/Braking
                robot["velocity"][0] *= 0.5
                robot["velocity"][1] *= 0.5

            # Max Speed Constraint (Load impacts speed)
            max_v = profile.max_velocity / (1.0 + robot["current_load"] * 0.5)
            v_mag = math.sqrt(robot["velocity"][0]**2 + robot["velocity"][1]**2)
            if v_mag > max_v:
                scale = max_v / v_mag
                robot["velocity"][0] *= scale
                robot["velocity"][1] *= scale

            # 2. Position Update
            old_pos = list(robot["pos"])
            new_pos = [
                old_pos[0] + robot["velocity"][0],
                old_pos[1] + robot["velocity"][1]
            ]

            # 3. Collision & Boundary Checks (ANTI-EXPLOIT)
            hit = False
            # Boundary
            if new_pos[0] < 0 or new_pos[0] >= self.size or new_pos[1] < 0 or new_pos[1] >= self.size:
                hit = True
                step_errors.append(f"{rid} boundary collision")
            # Racks
            elif [round(new_pos[0]), round(new_pos[1])] in self.racks:
                hit = True
                step_errors.append(f"{rid} rack collision")
            
            if hit:
                # Anti-Exploit: Collision damage drains battery and stops velocity
                robot["velocity"] = [0.0, 0.0]
                robot["battery"] = max(0.0, robot["battery"] - 5.0) 
                self.collision_count += 1
            else:
                robot["pos"] = new_pos

            # 4. Energy Drain
            drain = profile.battery_drain_rate * (1.0 + v_mag * 0.5)
            if robot["picked"]: drain *= profile.load_penalty
            robot["battery"] = max(0.0, robot["battery"] - drain)

            # 5. Semantic Actions (PICK/DROP/CHARGE) (ANTI-EXPLOIT penalties)
            if act == ActionType.PICK:
                if self._manhattan(robot["pos"], robot["pickup_target"]) < 1.5 and not robot["picked"]:
                    robot["picked"] = 1
                    robot["current_load"] = 1.0
                    robot["goal"] = robot["dropoff_target"]
                else:
                    self.invalid_action_count += 1
            elif act == ActionType.DROP:
                if self._manhattan(robot["pos"], robot["dropoff_target"]) < 1.5 and robot["picked"]:
                    robot["picked"] = 0
                    robot["current_load"] = 0.0
                    self.completed_tasks += 1
                    self._assign_task(robot)
                else:
                    self.invalid_action_count += 1
            elif act == ActionType.CHARGE:
                at_station = any(self._manhattan(robot["pos"], s) < 1.5 for s in self.charging_stations)
                if at_station:
                    robot["battery"] = min(100.0, robot["battery"] + 10.0)
                    robot["velocity"] = [0.0, 0.0]
                else:
                    self.invalid_action_count += 1

            # Spill Penalty & SLIDE (ANTI-EXPLOIT velocity hard-clamp)
            if [round(robot["pos"][0]), round(robot["pos"][1])] in [s["pos"] for s in self.spills]:
                self.spill_hits += 1
                robot["velocity"][0] *= 1.2 
                robot["velocity"][1] *= 1.2
                
                # Hard clamp velocity to 1.5x max_v even when sliding
                v_mag_slide = math.sqrt(robot["velocity"][0]**2 + robot["velocity"][1]**2)
                if v_mag_slide > profile.max_velocity * 1.5:
                    scale = (profile.max_velocity * 1.5) / v_mag_slide
                    robot["velocity"][0] *= scale
                    robot["velocity"][1] *= scale

        return self.get_observation(), step_rewards, step_errors

    def get_observation(self):
        # Simplified core obs for the wrapper
        return {
            "robots": self.robots,
            "spills": [s["pos"] for s in self.spills],
            "step_count": self.step_count,
            "max_steps": self.max_steps
        }

    def get_grade(self):
        rubric = EliteWarehouseRubric()
        raw = rubric(None, self)
        # Squash raw score to strictly open (0, 1) — satisfies OpenEnv requirement.
        # Uses the same tanh sigmoid defined in rubrics._squash.
        from rubrics import _squash
        return float(round(_squash(float(raw)), 6))

    def get_rubric_breakdown(self):
        rubric = EliteWarehouseRubric()
        rubric(None, self)
        return {n: float(round(c.last_score, 2)) for n, c in rubric.named_children()}


class WarehouseOpenEnv:
    def __init__(self, config):
        self.config = config
        self.env = WarehouseEnv(config)
        self.last_error = None

    def _convert_obs(self):
        core = self.env.get_observation()
        
        robots = {}
        for rid, r in self.env.robots.items():
            robots[rid] = RobotState(
                pos=r["pos"],
                velocity=r["velocity"],
                goal=r["goal"],
                picked=bool(r["picked"]),
                current_load=r["current_load"],
                battery=int(r["battery"]),
                status="active" if r["battery"] > 0 else "failed",
                last_action=r.get("last_action"),
                profile=r["profile"]
            )

        return Observation(
            grid_size=[self.env.size, self.env.size],
            robots=robots,
            environment=EnvironmentState(
                obstacles=[],
                spills=core["spills"],
                charging_stations=self.env.charging_stations,
                racks=self.env.racks,
                drop_zones=self.env.drop_zones
            ),
            dynamic=DynamicState(
                occupied_cells=[[int(r["pos"][0]), int(r["pos"][1])] for r in self.env.robots.values()],
                congestion={} # Could implement if needed
            ),
            task_info=TaskInfo(
                remaining_tasks=self.env.target_tasks - self.env.completed_tasks,
                completed_tasks=self.env.completed_tasks,
                system_throughput=self.env.completed_tasks / max(1, self.env.step_count)
            ),
            step_info=StepInfo(
                step_count=core["step_count"],
                max_steps=core["max_steps"]
            ),
            last_action_error=self.last_error,
            narrative=self.env.generate_narrative(),
            rubric_scores=self.env.get_rubric_breakdown()
        )

    def reset(self):
        self.env = WarehouseEnv(self.config)
        self.last_error = None
        return self._convert_obs()

    def step(self, action: Action):
        try:
            _, _, step_errors = self.env.apply_step(action.actions)
            self.last_error = "; ".join(step_errors) if step_errors else None
        except Exception as e:
            self.last_error = str(e)

        done = (self.env.step_count >= self.env.max_steps or 
                self.env.completed_tasks >= self.env.target_tasks)
        
        info = {"grade": self.env.get_grade()} if done else {}
        return self._convert_obs(), Reward(total=0.0), done, info

    def state(self):
        return self._convert_obs()

    def close(self):
        pass