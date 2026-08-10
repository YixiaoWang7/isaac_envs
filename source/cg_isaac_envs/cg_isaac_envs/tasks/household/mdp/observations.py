"""Policy and analysis observations."""

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from ..task_catalog import OBJECT_NAMES
from .relations import alternative_results, stage_results, task_term


def object_poses(env) -> torch.Tensor:
    values = []
    for name in OBJECT_NAMES:
        asset = env.scene[name]
        values.append(torch.cat((asset.data.root_pos_w - env.scene.env_origins, asset.data.root_quat_w), dim=-1))
    return torch.cat(values, dim=-1)


def object_velocities(env) -> torch.Tensor:
    return torch.cat([env.scene[name].data.root_vel_w for name in OBJECT_NAMES], dim=-1)


def ee_pose(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    sensor = env.scene[sensor_cfg.name]
    return torch.cat(
        (sensor.data.target_pos_w[:, 0] - env.scene.env_origins, sensor.data.target_quat_w[:, 0]), dim=-1
    )


def object_relative_gripper_poses(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    sensor = env.scene[sensor_cfg.name]
    ee_pos = sensor.data.target_pos_w[:, 0]
    ee_quat = sensor.data.target_quat_w[:, 0]
    values = []
    for name in OBJECT_NAMES:
        asset = env.scene[name]
        rel_pos, rel_quat = math_utils.subtract_frame_transforms(
            asset.data.root_pos_w, asset.data.root_quat_w, ee_pos, ee_quat
        )
        values.append(torch.cat((rel_pos, rel_quat), dim=-1))
    return torch.cat(values, dim=-1)


def task_encoding(env):
    return task_term(env).command


def prompt_tokens(env):
    return task_term(env).prompt_tokens.float()


def task_id(env):
    return task_term(env).task_ids[:, None].float()


def relation_progress(env):
    return alternative_results(env)[1][:, None]


def stages(env):
    return stage_results(env)[0].float()


def front_rgb(env):
    return env.scene["front_camera"].data.output["rgb"][..., :3].float()
