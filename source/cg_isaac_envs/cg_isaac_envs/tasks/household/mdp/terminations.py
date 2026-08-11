"""Stable success and object-drop termination terms."""

from __future__ import annotations

import torch
from isaaclab.managers import ManagerTermBase

from ..task_catalog import OBJECT_NAMES
from .relations import alternative_results, required_object_mask


def required_objects_stable(env, max_speed: float = 0.08) -> torch.Tensor:
    speeds = torch.stack(
        [torch.linalg.vector_norm(env.scene[name].data.root_lin_vel_w, dim=1) for name in OBJECT_NAMES], dim=1
    )
    relevant_speeds = torch.where(required_object_mask(env), speeds, torch.zeros_like(speeds))
    return relevant_speeds.amax(dim=1) < max_speed


class StableTaskSuccess(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(self, env, consecutive_steps: int = 10, max_speed: float = 0.08):
        success, _ = alternative_results(env)
        stable = required_objects_stable(env, max_speed)
        self.counter = torch.where(success & stable, self.counter + 1, torch.zeros_like(self.counter))
        return self.counter >= consecutive_steps

    def reset(self, env_ids=None):
        self.counter[env_ids] = 0


def required_object_dropped(env, minimum_height: float = -0.08):
    required = required_object_mask(env)
    heights = torch.stack([env.scene[name].data.root_pos_w[:, 2] for name in OBJECT_NAMES], dim=1)
    return ((heights < minimum_height) & required).any(dim=1)
