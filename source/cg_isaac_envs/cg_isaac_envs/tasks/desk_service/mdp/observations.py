"""Observation functions."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from ..task_catalog import OBJECT_NAMES
from .relations import alternative_results, stage_results, task_term


def object_poses(env) -> torch.Tensor:
    values = []
    for name in OBJECT_NAMES:
        asset = env.scene[name]
        position = asset.data.root_pos_w - env.scene.env_origins
        values.append(torch.cat((position, asset.data.root_quat_w), dim=-1))
    return torch.cat(values, dim=-1)


def object_velocities(env) -> torch.Tensor:
    return torch.cat([env.scene[name].data.root_vel_w for name in OBJECT_NAMES], dim=-1)


def ee_pose(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    sensor = env.scene[sensor_cfg.name]
    position = sensor.data.target_pos_w[:, 0, :] - env.scene.env_origins
    orientation = sensor.data.target_quat_w[:, 0, :]
    return torch.cat((position, orientation), dim=-1)


def task_encoding(env) -> torch.Tensor:
    return task_term(env).command


def prompt_tokens(env) -> torch.Tensor:
    return task_term(env).prompt_tokens.float()


def task_id(env) -> torch.Tensor:
    return task_term(env).task_ids[:, None].float()


def relation_progress(env) -> torch.Tensor:
    _, fraction = alternative_results(env)
    return fraction[:, None]


def stages(env) -> torch.Tensor:
    stage, _ = stage_results(env)
    return stage.float()


def stage_applicability(env) -> torch.Tensor:
    _, applicable = stage_results(env)
    return applicable.float()
