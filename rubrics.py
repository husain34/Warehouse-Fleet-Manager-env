import math
from typing import Any, Dict
from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import WeightedSum


def _squash(raw: float) -> float:
    """
    Maps a raw score in [0, 1] to a strictly open (0, 1) interval using
    a tanh-based sigmoid transformation — no hardcoded floor/ceiling.

    Formula: squash(x) = (1 + tanh(k * (2x - 1))) / 2
      - k=3.5 gives ~ (0.0007, 0.9993) across the full [0,1] domain.
      - Symmetric around 0.5: squash(0.5) == 0.5 exactly.
      - Monotonically increasing: higher raw score → higher squashed score.
    """
    k = 3.5
    return (1.0 + math.tanh(k * (2.0 * raw - 1.0))) / 2.0


class LogisticsEfficiencyRubric(Rubric):
    """Measures task throughput and path optimality."""
    def forward(self, action: Any, observation: Any) -> float:
        env = observation
        if env.target_tasks == 0: return 0.5

        completion_ratio = env.completed_tasks / env.target_tasks

        # Throughput bonus (tasks per 100 steps)
        # Elite benchmark: 4 tasks per 100 steps (relaxed from 5)
        throughput = (env.completed_tasks / max(1, env.step_count)) * 100
        throughput_score = min(1.0, throughput / 4.0)

        return (completion_ratio * 0.7) + (throughput_score * 0.3)


class OperationalSafetyRubric(Rubric):
    """Measures collisions, near-misses, and boundary violations."""
    def forward(self, action: Any, observation: Any) -> float:
        env = observation
        if env.step_count == 0: return 0.9  # Near-perfect start

        # Penalize collisions — softened so a few bumps don't tank the score
        collision_penalty = min(0.6, (env.collision_count * 0.15))

        # Penalize spill hits
        spill_penalty = min(0.2, (env.spill_hits * 0.04))

        # Penalize invalid actions (semantic spam)
        spam_penalty = min(0.2, (getattr(env, "invalid_action_count", 0) * 0.008))

        return max(0.0, 1.0 - (collision_penalty + spill_penalty + spam_penalty))


class SustainabilityRubric(Rubric):
    """Measures battery health and energy management."""
    def forward(self, action: Any, observation: Any) -> float:
        env = observation
        if not env.robots: return 0.9

        # Support both dict robots (WarehouseEnv) and RobotState objects (WarehouseOpenEnv)
        def _battery(r):
            return r["battery"] if isinstance(r, dict) else r.battery

        avg_battery = sum(_battery(r) for r in env.robots.values()) / len(env.robots)

        # Penalty for critical/dead batteries
        critical_battery_penalty = 0.0
        for r in env.robots.values():
            b = _battery(r)
            if b < 5:   critical_battery_penalty += 0.20
            elif b < 20: critical_battery_penalty += 0.04

        battery_health = min(1.0, avg_battery / 100.0)

        return max(0.0, battery_health - min(0.5, critical_battery_penalty))


class SLAComplianceRubric(Rubric):
    """Measures if tasks were completed within reasonable timeframes."""
    def forward(self, action: Any, observation: Any) -> float:
        env = observation
        if env.completed_tasks == 0: return 0.1  # Small non-zero base — shows env ran

        # Average steps per task
        avg_steps = env.step_count / env.completed_tasks
        # Elite: < 30 steps/task, Poor: > 60 steps/task
        # Relaxed thresholds to account for momentum-based physics overhead
        sla_score = 1.0 - min(1.0, max(0.0, (avg_steps - 30) / 30))

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
