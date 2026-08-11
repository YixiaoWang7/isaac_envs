"""Stable success and object-drop termination terms."""

from __future__ import annotations

import torch
from isaaclab.managers import ManagerTermBase

from .relations import alternative_results, required_object_mask


class StableTaskSuccess(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(self, env, consecutive_steps: int = 30, minimum_ee_height: float = 0.20):
        success, _ = alternative_results(env)
        ee_height = env.scene["ee_frame"].data.target_pos_w[:, 0, 2] - env.scene.env_origins[:, 2]
        all_criteria = success & (ee_height >= minimum_ee_height)
        self.counter = torch.where(all_criteria, self.counter + 1, torch.zeros_like(self.counter))
        return self.counter >= consecutive_steps

    def reset(self, env_ids=None):
        self.counter[env_ids] = 0


def required_object_dropped(env, minimum_height: float = -0.08):
    required = required_object_mask(env)
    heights = torch.stack([env.scene[name].data.root_pos_w[:, 2] for name in OBJECT_NAMES], dim=1)
    return ((heights < minimum_height) & required).any(dim=1)
