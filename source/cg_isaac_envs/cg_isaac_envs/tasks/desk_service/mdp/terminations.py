"""Success and safety termination terms."""

from __future__ import annotations

import torch

from isaaclab.managers import ManagerTermBase

from ..task_catalog import OBJECT_NAMES
from .relations import alternative_results, required_object_mask


class StableTaskSuccess(ManagerTermBase):
    """Terminate after the complete relation graph is stable."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(self, env, consecutive_steps: int = 10, max_speed: float = 0.08) -> torch.Tensor:
        success, _ = alternative_results(env)
        speeds = []
        for name in OBJECT_NAMES:
            speeds.append(torch.linalg.vector_norm(env.scene[name].data.root_lin_vel_w, dim=1))
        stable = torch.stack(speeds, dim=1).amax(dim=1) < max_speed
        self.counter = torch.where(success & stable, self.counter + 1, torch.zeros_like(self.counter))
        return self.counter >= consecutive_steps

    def reset(self, env_ids=None):
        self.counter[env_ids] = 0


def required_object_dropped(env, minimum_height: float = 0.58) -> torch.Tensor:
    required = required_object_mask(env)
    heights = torch.stack([env.scene[name].data.root_pos_w[:, 2] for name in OBJECT_NAMES], dim=1)
    return ((heights < minimum_height) & required).any(dim=1)

