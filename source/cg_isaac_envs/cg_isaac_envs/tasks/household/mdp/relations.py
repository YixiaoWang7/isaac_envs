"""Vectorized final-state relations for household tasks."""

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils

from ..task_catalog import OBJECT_NAMES, STAGE_NAMES, TASK_BY_ID

GEOMETRY = {
    "bowl": (0.061, 0.061, 0.042),
    "plate": (0.071, 0.071, 0.015),
    "serving_pan": (0.080, 0.133, 0.085),
    "apple": (0.031, 0.031, 0.058),
    "banana": (0.039, 0.063, 0.026),
    "snack_package": (0.038, 0.035, 0.022),
    "left_place": (0.085, 0.085, 0.012),
    "right_place": (0.085, 0.085, 0.012),
    "packing_place": (0.120, 0.090, 0.025),
}


def task_term(env):
    return env.command_manager.get_term("task")


def relative_position(env, subject: str, target: str) -> torch.Tensor:
    subject_asset = env.scene[subject]
    target_asset = env.scene[target]
    return math_utils.quat_apply_inverse(
        target_asset.data.root_quat_w, subject_asset.data.root_pos_w - target_asset.data.root_pos_w
    )


def inside(env, subject: str, target: str) -> torch.Tensor:
    rel = relative_position(env, subject, target)
    tx, ty, tz = GEOMETRY[target]
    sx, sy, _ = GEOMETRY[subject]
    x_margin = max(0.018, tx - min(sx * 0.45, tx * 0.35))
    y_margin = max(0.018, ty - min(sy * 0.45, ty * 0.35))
    return (
        (rel[:, 0].abs() < x_margin)
        & (rel[:, 1].abs() < y_margin)
        & (rel[:, 2] > -0.035)
        & (rel[:, 2] < tz + 0.12)
    )


def at(env, subject: str, target: str) -> torch.Tensor:
    rel = relative_position(env, subject, target)
    tx, ty, tz = GEOMETRY[target]
    return (rel[:, 0].abs() < tx) & (rel[:, 1].abs() < ty) & (rel[:, 2].abs() < tz + 0.12)


def on(env, subject: str, target: str) -> torch.Tensor:
    return at(env, subject, target)

def beside(env, subject: str, target: str) -> torch.Tensor:
    rel = relative_position(env, subject, target)
    distance = torch.linalg.vector_norm(rel[:, :2], dim=1)
    return (distance > 0.045) & (distance < 0.16) & (rel[:, 2].abs() < 0.12)


def upright(env, subject: str) -> torch.Tensor:
    quat = env.scene[subject].data.root_quat_w
    local_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1)
    world_z = math_utils.quat_apply(quat, local_z)
    return world_z[:, 2] > 0.94


def relation_value(env, subject: str, relation: str, target: str) -> torch.Tensor:
    if relation == "inside":
        return inside(env, subject, target)
    if relation == "at":
        return at(env, subject, target)
    if relation == "on":
        return on(env, subject, target)
    if relation == "beside":
        return beside(env, subject, target)
    if relation == "upright":
        return upright(env, subject)
    raise KeyError(relation)


def alternative_results(env) -> tuple[torch.Tensor, torch.Tensor]:
    command = task_term(env)
    success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fraction = torch.zeros(env.num_envs, device=env.device)
    for task_id, task in TASK_BY_ID.items():
        mask = command.task_ids == task_id
        if not torch.any(mask):
            continue
        values = []
        for goal in task.alternatives[0]:
            value = relation_value(env, goal.subject, goal.relation, goal.target)
            values.append(value if goal.required else ~value)
        stacked = torch.stack(values, dim=1)
        success[mask] = stacked.all(dim=1)[mask]
        fraction[mask] = stacked.float().mean(dim=1)[mask]
    return success, fraction


def stage_results(env) -> tuple[torch.Tensor, torch.Tensor]:
    command = task_term(env)
    result = torch.zeros(env.num_envs, len(STAGE_NAMES), dtype=torch.bool, device=env.device)
    applicable = torch.ones_like(result)
    for task_id, task in TASK_BY_ID.items():
        mask = command.task_ids == task_id
        if not torch.any(mask):
            continue
        goals = task.alternatives[0]
        for index, subject in enumerate(task.manipulation_subjects):
            subject_values = []
            for goal in goals:
                if goal.subject == subject:
                    value = relation_value(env, goal.subject, goal.relation, goal.target)
                    subject_values.append(value if goal.required else ~value)
            result[mask, index] = torch.stack(subject_values, dim=1).all(dim=1)[mask]
    return result, applicable


def required_object_mask(env) -> torch.Tensor:
    command = task_term(env)
    mask = torch.zeros(env.num_envs, len(OBJECT_NAMES), dtype=torch.bool, device=env.device)
    index = {name: i for i, name in enumerate(OBJECT_NAMES)}
    for task_id, task in TASK_BY_ID.items():
        env_mask = command.task_ids == task_id
        for name in task.manipulation_subjects:
            mask[env_mask, index[name]] = True
    return mask
