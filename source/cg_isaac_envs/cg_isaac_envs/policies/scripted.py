"""Privileged waypoint reference policy for plumbing and physics smoke tests."""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply

from ..tasks.desk_service.mdp.relations import GEOMETRY
from ..tasks.desk_service.task_catalog import TASK_BY_ID, RelationGoal


class ScriptedDeskPolicy:
    """Execute a canonical topological ordering of the requested relations.

    This controller reads simulator state and is intentionally privileged. It is a
    reproducible reference for environment/evaluator validation, not a learned or
    language-understanding baseline.
    """

    def __init__(self, env, position_gain: float = 8.0, tolerance: float = 0.018):
        self.env = env
        self.device = env.device
        self.position_gain = position_gain
        self.tolerance = tolerance
        self.operation_index = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.phase = torch.zeros_like(self.operation_index)
        self.dwell = torch.zeros_like(self.operation_index)
        self.plans: list[list[RelationGoal]] = [[] for _ in range(env.num_envs)]
        self._rebuild(torch.arange(env.num_envs, device=self.device))

    def _rebuild(self, env_ids: torch.Tensor) -> None:
        priority = {
            "inside": 0,
            "accessory": 1,
            "container": 2,
            "at": 3,
        }
        for env_id in env_ids.tolist():
            task_id = int(self.env.command_manager.get_term("task").task_ids[env_id])
            goals = [goal for goal in TASK_BY_ID[task_id].alternatives[0] if goal.required]

            def key(goal: RelationGoal):
                if goal.relation == "inside":
                    return priority["inside"]
                if goal.subject in {"coaster", "napkin", "spoon"}:
                    return priority["accessory"]
                if goal.relation == "at":
                    return priority["at"]
                return priority["container"]

            self.plans[env_id] = sorted(goals, key=key)
        self.operation_index[env_ids] = 0
        self.phase[env_ids] = 0
        self.dwell[env_ids] = 0

    def reset(self, env_ids: torch.Tensor, tasks=None) -> None:
        self._rebuild(env_ids)

    def _pose(self, name: str, env_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        asset = self.env.scene[name]
        return asset.data.root_pos_w[env_id], asset.data.root_quat_w[env_id]

    def _place_position(self, goal: RelationGoal, env_id: int) -> torch.Tensor:
        target_pos, target_quat = self._pose(goal.target, env_id)
        _, _, subject_bottom, _ = GEOMETRY[goal.subject]
        _, _, _, target_top = GEOMETRY[goal.target]
        if goal.relation == "inside":
            local = torch.tensor([0.0, 0.0, max(0.014, -subject_bottom + 0.010)], device=self.device)
        else:
            local = torch.tensor([0.0, 0.0, target_top - subject_bottom + 0.002], device=self.device)
        return target_pos + quat_apply(target_quat, local)

    def act(self, observations: dict, tasks=None) -> torch.Tensor:
        actions = torch.zeros(self.env.num_envs, 7, device=self.device)
        ee_sensor = self.env.scene["ee_frame"]
        ee_pos = ee_sensor.data.target_pos_w[:, 0, :]
        for env_id in range(self.env.num_envs):
            plan = self.plans[env_id]
            op_index = int(self.operation_index[env_id])
            if op_index >= len(plan):
                actions[env_id, 6] = 1.0
                continue
            goal = plan[op_index]
            source_pos, _ = self._pose(goal.subject, env_id)
            place_pos = self._place_position(goal, env_id)
            phase = int(self.phase[env_id])
            if phase == 0:
                target = source_pos + torch.tensor([0.0, 0.0, 0.14], device=self.device)
                gripper = 1.0
            elif phase == 1:
                target = source_pos + torch.tensor([0.0, 0.0, max(0.018, GEOMETRY[goal.subject][3] * 0.45)], device=self.device)
                gripper = 1.0
            elif phase == 2:
                target = ee_pos[env_id]
                gripper = -1.0
            elif phase == 3:
                target = source_pos + torch.tensor([0.0, 0.0, 0.17], device=self.device)
                gripper = -1.0
            elif phase == 4:
                target = place_pos + torch.tensor([0.0, 0.0, 0.16], device=self.device)
                gripper = -1.0
            elif phase == 5:
                target = place_pos + torch.tensor([0.0, 0.0, 0.010], device=self.device)
                gripper = -1.0
            elif phase == 6:
                target = ee_pos[env_id]
                gripper = 1.0
            else:
                target = place_pos + torch.tensor([0.0, 0.0, 0.16], device=self.device)
                gripper = 1.0

            delta = target - ee_pos[env_id]
            actions[env_id, :3] = torch.clamp(delta * self.position_gain, -1.0, 1.0)
            actions[env_id, 6] = gripper
            reached = torch.linalg.vector_norm(delta) < self.tolerance
            dwell_limit = 8 if phase in {2, 6} else 2
            self.dwell[env_id] = self.dwell[env_id] + 1 if reached or phase in {2, 6} else 0
            if int(self.dwell[env_id]) >= dwell_limit:
                self.dwell[env_id] = 0
                if phase >= 7:
                    self.phase[env_id] = 0
                    self.operation_index[env_id] += 1
                else:
                    self.phase[env_id] += 1
        return actions

