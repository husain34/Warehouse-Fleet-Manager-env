from typing import Any, Dict
from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import WeightedSum

class LogisticsEfficiencyRubric(Rubric):
    """Measures task throughput and path optimality."""
    def forward(self, action: Any, observation: Any) -> float:
        env = observation
        if env.target_tasks == 0: return 1.0
        
        completion_ratio = env.completed_tasks / env.target_tasks
        
        # Throughput bonus (tasks per 100 steps)
        throughput = (env.completed_tasks / max(1, env.step_count)) * 100
        throughput_score = min(1.0, throughput / 5.0) # Assume 5 tasks per 100 steps is elite
        
        return (completion_ratio * 0.7) + (throughput_score * 0.3)

class OperationalSafetyRubric(Rubric):
    """Measures collisions, near-misses, and boundary violations."""
    def forward(self, action: Any, observation: Any) -> float:
        env = observation
        if env.step_count == 0: return 1.0
        
        # Penalize collisions significantly
        collision_penalty = min(0.8, (env.collision_count * 0.2))
        
        # Penalize spill hits
        spill_penalty = min(0.2, (env.spill_hits * 0.05))
        
        # New: Penalize invalid actions (semantic spam)
        spam_penalty = min(0.2, (getattr(env, "invalid_action_count", 0) * 0.01))
        
        return max(0.0, 1.0 - (collision_penalty + spill_penalty + spam_penalty))

class SustainabilityRubric(Rubric):
    """Measures battery health and energy management."""
    def forward(self, action: Any, observation: Any) -> float:
        env = observation
        if not env.robots: return 1.0
        
        avg_battery = sum(r["battery"] for r in env.robots.values()) / len(env.robots)
        
        # Penalty for any robot being 'dead' or very low battery
        critical_battery_penalty = 0.0
        for r in env.robots.values():
            if r["battery"] < 5: critical_battery_penalty += 0.25
            elif r["battery"] < 20: critical_battery_penalty += 0.05
            
        battery_health = min(1.0, avg_battery / 100.0)
        
        return max(0.0, battery_health - min(0.5, critical_battery_penalty))

class SLAComplianceRubric(Rubric):
    """Measures if tasks were completed within reasonable timeframes."""
    def forward(self, action: Any, observation: Any) -> float:
        env = observation
        if env.completed_tasks == 0: return 0.5 # Neutral start
        
        # Calculate average steps per task
        avg_steps = env.step_count / env.completed_tasks
        # Assume < 20 steps is elite, > 40 is poor
        sla_score = 1.0 - min(1.0, max(0.0, (avg_steps - 20) / 20))
        
        return sla_score

class EliteWarehouseRubric(WeightedSum):
    def __init__(self):
        super().__init__(
            rubrics=[
                LogisticsEfficiencyRubric(), 
                OperationalSafetyRubric(), 
                SustainabilityRubric(),
                SLAComplianceRubric()
            ],
            weights=[0.4, 0.2, 0.2, 0.2]
        )
