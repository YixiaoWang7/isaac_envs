"""Vectorized geometric relations and task-stage evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

from ..task_catalog import OBJECT_NAMES, STAGE_NAMES, TASK_BY_ID

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# (half x, half y, root-to-bottom, root-to-top)
GEOMETRY = {
    "mug": (0.036, 0.036, 0.0, 0.096),
    "bowl": (0.052, 0.052, 0.0, 0.058),
    "takeaway_box": (0.070, 0.055, 0.0, 0.069),
    "tray": (0.170, 0.120, 0.0, 0.045),
    "tea": (0.016, 0.016, -0.016, 0.016),
    "sugar": (0.014, 0.014, -0.014, 0.014),
    "red_candy": (0.015, 0.015, -0.015, 0.015),
    "blue_candy": (0.015, 0.015, -0.015, 0.015),
    "cookie": (0.025, 0.025, -0.008, 0.008),
    "spoon": (0.075, 0.012, -0.006, 0.006),
    "coaster": (0.055, 0.055, -0.003, 0.003),
    "napkin": (0.060, 0.060, -0.002, 0.002),
    "left": (0.130, 0.180, -0.004, 0.004),
    "right": (0.130, 0.180, -0.004, 0.004),
}


def _asset_pose(env: "ManagerBasedRLEnv", name: str) -> tuple[torch.Tensor, torch.Tensor]:
    asset = env.scene[name]
    if name in {"left", "right"}:
        return asset.data.root_pos_w, asset.data.root_quat_w
    return asset.data.root_pos_w, asset.data.root_quat_w


def relative_position(env: "ManagerBasedRLEnv", subject: str, target: str) -> torch.Tensor:
    """Subject position expressed in the target frame."""

    subject_pos, _ = _asset_pose(env, subject)
    target_pos, target_quat = _asset_pose(env, target)
    return math_utils.quat_apply_inverse(target_quat, subject_pos - target_pos)


def inside(env: "ManagerBasedRLEnv", subject: str, target: str) -> torch.Tensor:
    rel = relative_position(env, subject, target)
    target_x, target_y, _, target_top = GEOMETRY[target]
    subject_x, subject_y, _, _ = GEOMETRY[subject]
    margin_x = max(0.005, target_x - min(subject_x, target_x * 0.5) - 0.003)
    margin_y = max(0.005, target_y - min(subject_y, target_y * 0.5) - 0.003)
    return (
        (rel[:, 0].abs() < margin_x)
        & (rel[:, 1].abs() < margin_y)
        & (rel[:, 2] > 0.002)
        & (rel[:, 2] < target_top + 0.025)
    )


def on(env: "ManagerBasedRLEnv", subject: str, target: str) -> torch.Tensor:
    rel = relative_position(env, subject, target)
    target_x, target_y, _, target_top = GEOMETRY[target]
    subject_x, subject_y, subject_bottom, _ = GEOMETRY[subject]
    horizontal = (rel[:, 0].abs() < max(0.01, target_x - min(subject_x, target_x * 0.45))) & (
        rel[:, 1].abs() < max(0.01, target_y - min(subject_y, target_y * 0.45))
    )
    vertical = ((rel[:, 2] + subject_bottom) - target_top).abs() < 0.035
    return horizontal & vertical


def at(env: "ManagerBasedRLEnv", subject: str, target: str) -> torch.Tensor:
    rel = relative_position(env, subject, target)
    target_x, target_y, _, _ = GEOMETRY[target]
    return (rel[:, 0].abs() < target_x) & (rel[:, 1].abs() < target_y)


def relation_value(env: "ManagerBasedRLEnv", subject: str, relation: str, target: str) -> torch.Tensor:
    if relation == "inside":
        return inside(env, subject, target)
    if relation == "on":
        return on(env, subject, target)
    if relation == "at":
        return at(env, subject, target)
    raise KeyError(relation)


def task_term(env: "ManagerBasedRLEnv"):
    return env.command_manager.get_term("task")


def alternative_results(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """Return best alternative success and fraction for each environment."""

    command = task_term(env)
    success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fraction = torch.zeros(env.num_envs, device=env.device)
    for task_id, task in TASK_BY_ID.items():
        env_mask = command.task_ids == task_id
        if not torch.any(env_mask):
            continue
        best_fraction = torch.zeros(env.num_envs, device=env.device)
        best_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for alternative in task.alternatives:
            values = []
            for goal in alternative:
                value = relation_value(env, goal.subject, goal.relation, goal.target)
                values.append(value if goal.required else ~value)
            stacked = torch.stack(values, dim=1)
            alt_fraction = stacked.float().mean(dim=1)
            best_fraction = torch.maximum(best_fraction, alt_fraction)
            best_success |= stacked.all(dim=1)
        success[env_mask] = best_success[env_mask]
        fraction[env_mask] = best_fraction[env_mask]
    return success, fraction


def stage_results(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """Return current stage success and applicability, shape ``(N, 4)``."""

    command = task_term(env)
    result = torch.zeros(env.num_envs, len(STAGE_NAMES), dtype=torch.bool, device=env.device)
    applicable = torch.zeros_like(result)
    accessory_subjects = {"coaster", "napkin", "spoon"}
    container_subjects = {"mug", "bowl"}
    for task_id, task in TASK_BY_ID.items():
        env_mask = command.task_ids == task_id
        if not torch.any(env_mask):
            continue
        task_stage_best = torch.zeros(env.num_envs, len(STAGE_NAMES), dtype=torch.bool, device=env.device)
        task_applicable = torch.zeros_like(task_stage_best)
        for alternative in task.alternatives:
            buckets: list[list[torch.Tensor]] = [[] for _ in STAGE_NAMES]
            for goal in alternative:
                value = relation_value(env, goal.subject, goal.relation, goal.target)
                value = value if goal.required else ~value
                if goal.relation == "inside":
                    stage_idx = 0
                elif goal.subject in accessory_subjects:
                    stage_idx = 1
                elif goal.relation == "on" and goal.subject in container_subjects:
                    stage_idx = 2
                elif goal.relation == "at":
                    stage_idx = 3
                else:
                    stage_idx = 2
                buckets[stage_idx].append(value)
            for stage_idx, values in enumerate(buckets):
                if values:
                    task_applicable[:, stage_idx] = True
                    task_stage_best[:, stage_idx] |= torch.stack(values, dim=1).all(dim=1)
        result[env_mask] = task_stage_best[env_mask]
        applicable[env_mask] = task_applicable[env_mask]
    return result, applicable


def required_object_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Objects appearing in at least one goal alternative."""

    command = task_term(env)
    mask = torch.zeros(env.num_envs, len(OBJECT_NAMES), dtype=torch.bool, device=env.device)
    object_index = {name: idx for idx, name in enumerate(OBJECT_NAMES)}
    for task_id, task in TASK_BY_ID.items():
        env_mask = command.task_ids == task_id
        names = {goal.subject for alt in task.alternatives for goal in alt if goal.required}
        for name in names:
            if name in object_index:
                mask[env_mask, object_index[name]] = True
    return mask

