"""Privileged waypoint controller for environment and physics validation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.utils.math import quat_apply

from ..tasks.household.mdp.relations import GEOMETRY
from ..tasks.household.task_catalog import TASK_BY_ID


@dataclass(frozen=True)
class Operation:
    subject: str
    relation: str
    target: str


class ScriptedHouseholdPolicy:
    """Execute one canonical solution without constraining valid demonstrations."""

    def __init__(self, env, position_gain: float = 8.0, tolerance: float = 0.018):
        self.env = env
        self.device = env.device
        self.position_gain = position_gain
        self.tolerance = tolerance
        self.operation_index = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.grasp_anchor = torch.zeros(env.num_envs, 3, device=self.device)
        self.place_anchor = torch.zeros(env.num_envs, 3, device=self.device)
        self.source_z = torch.zeros(env.num_envs, device=self.device)
        self.phase = torch.zeros_like(self.operation_index)
        self.dwell = torch.zeros_like(self.operation_index)
        self.plans: list[list[Operation]] = [[] for _ in range(env.num_envs)]
        self._rebuild(torch.arange(env.num_envs, device=self.device))

    def _rebuild(self, env_ids):
        command = self.env.command_manager.get_term("task")
        for env_id in torch.as_tensor(env_ids, device=self.device).tolist():
            task = TASK_BY_ID[int(command.task_ids[env_id])]
            goals = [goal for goal in task.alternatives[0] if goal.required and goal.relation != "upright"]
            by_subject = {subject: [goal for goal in goals if goal.subject == subject] for subject in task.manipulation_subjects}
            subjects = list(task.manipulation_subjects)
            self.plans[env_id] = [
                Operation(subject, by_subject[subject][0].relation, by_subject[subject][0].target) for subject in subjects
            ]
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.operation_index[env_ids] = 0
        self.phase[env_ids] = 0
        self.dwell[env_ids] = 0

    def reset(self, env_ids, tasks=None):
        self._rebuild(env_ids)

    def _pose(self, name, env_id):
        asset = self.env.scene[name]
        return asset.data.root_pos_w[env_id], asset.data.root_quat_w[env_id]

    def _grasp_position(self, subject, env_id):
        pos, quat = self._pose(subject, env_id)
        if subject == "bowl":
            local = torch.tensor([0.0, 0.045, 0.025], device=self.device)
            return pos + quat_apply(quat, local)
        return pos + torch.tensor([0.0, 0.0, 0.020], device=self.device)

    def _place_position(self, operation, env_id, operation_index):
        target_pos, target_quat = self._pose(operation.target, env_id)
        if operation.relation == "inside":
            offsets = ((-0.030, 0.0), (0.030, 0.0), (0.0, 0.040))
            dx, dy = offsets[operation_index % len(offsets)]
            z = 0.072 if operation.target == "bowl" else 0.105
            local = torch.tensor([dx, dy, z], device=self.device)
        else:
            if operation.subject == "bowl":
                local = torch.tensor([0.0, 0.052, 0.040], device=self.device)
            else:
                local = torch.tensor([0.065, 0.0, GEOMETRY[operation.subject][2] * 0.5], device=self.device)
        return target_pos + quat_apply(target_quat, local)

    def act(self, observations, tasks=None):
        actions = torch.zeros(self.env.num_envs, 7, device=self.device)
        ee_pos = self.env.scene["ee_frame"].data.target_pos_w[:, 0]
        for env_id in range(self.env.num_envs):
            op_index = int(self.operation_index[env_id])
            if op_index >= len(self.plans[env_id]):
                actions[env_id, 6] = 1.0
                continue
            operation = self.plans[env_id][op_index]
            grasp = self._grasp_position(operation.subject, env_id)
            place = self._place_position(operation, env_id, op_index)
            phase = int(self.phase[env_id])
            if phase == 0:
                self.grasp_anchor[env_id] = grasp
                self.place_anchor[env_id] = place
                self.source_z[env_id] = self.env.scene[operation.subject].data.root_pos_w[env_id, 2]
            grasp = self.grasp_anchor[env_id]
            place = self.place_anchor[env_id]
            if phase == 0:
                target, gripper = grasp + torch.tensor([0.0, 0.0, 0.14], device=self.device), 1.0
            elif phase == 1:
                target, gripper = grasp, 1.0
            elif phase == 2:
                target, gripper = ee_pos[env_id], -1.0
            elif phase == 3:
                target, gripper = grasp + torch.tensor([0.0, 0.0, 0.16], device=self.device), -1.0
            elif phase == 4:
                target, gripper = place + torch.tensor([0.0, 0.0, 0.15], device=self.device), -1.0
            elif phase == 5:
                target, gripper = place, -1.0
            elif phase == 6:
                target, gripper = ee_pos[env_id], 1.0
            else:
                target, gripper = place + torch.tensor([0.0, 0.0, 0.15], device=self.device), 1.0
            delta = target - ee_pos[env_id]
            actions[env_id, :3] = torch.clamp(delta * self.position_gain, -1.0, 1.0)
            actions[env_id, 6] = gripper
            reached = torch.linalg.vector_norm(delta) < self.tolerance
            dwell_limit = 10 if phase in {2, 6} else 3
            self.dwell[env_id] = self.dwell[env_id] + 1 if reached or phase in {2, 6} else 0
            if int(self.dwell[env_id]) >= dwell_limit:
                self.dwell[env_id] = 0
                if phase == 7:
                    self.phase[env_id] = 0
                    self.operation_index[env_id] += 1
                else:
                    self.phase[env_id] += 1
                if phase == 3:
                    object_z = self.env.scene[operation.subject].data.root_pos_w[env_id, 2]
                    if object_z < self.source_z[env_id] + 0.035:
                        self.phase[env_id] = 0
                        continue
        return actions


ScriptedDeskPolicy = ScriptedHouseholdPolicy
