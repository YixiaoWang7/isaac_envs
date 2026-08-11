"""Vectorized final-state relations for household tasks."""

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils

from ..task_catalog import OBJECT_NAMES, STAGE_NAMES, TASK_BY_ID

GEOMETRY = {
    "red_candy": (0.018, 0.018, 0.018),
    "blue_candy": (0.018, 0.018, 0.018),
    "green_candy": (0.018, 0.018, 0.018),
    "mug_a": (0.042, 0.042, 0.080),
    "mug_b": (0.042, 0.042, 0.080),
    "mug_c": (0.042, 0.042, 0.080),
    "hot_serving_place": (0.085, 0.085, 0.012),
    "cold_serving_place": (0.085, 0.085, 0.012),
    "storage_place": (0.120, 0.090, 0.025),
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


def handle_lift(env, subject: str) -> torch.Tensor:
    """Latch a mug lift performed while the fingers occupy its handle region."""
    command = task_term(env)
    mug = env.scene[subject]
    robot = env.scene["robot"]
    finger_ids, _ = robot.find_bodies(["panda_leftfinger", "panda_rightfinger"])
    finger_midpoint = robot.data.body_pos_w[:, finger_ids].mean(dim=1)
    relative_fingers = math_utils.quat_apply_inverse(
        mug.data.root_quat_w, finger_midpoint - mug.data.root_pos_w
    )
    radial_distance = torch.linalg.vector_norm(relative_fingers[:, :2], dim=1)
    mug_height = GEOMETRY[subject][2]
    fingers_at_handle = (
        (radial_distance > 0.050)
        & (radial_distance < 0.115)
        & (relative_fingers[:, 2] > 0.005)
        & (relative_fingers[:, 2] < mug_height + 0.025)
    )
    lifted = mug.data.root_pos_w[:, 2] > 0.080
    command.handle_lifted |= fingers_at_handle & lifted
    return command.handle_lifted


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
    if relation == "handle_lift":
        return handle_lift(env, subject)
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
