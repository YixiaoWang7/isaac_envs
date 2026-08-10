"""Per-environment prompt command for household tasks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .task_catalog import CATALOG, TASK_BY_ID, split_task_ids, tokenize_prompt


class HouseholdTaskCommand(CommandTerm):
    cfg: "HouseholdTaskCommandCfg"

    def __init__(self, cfg: "HouseholdTaskCommandCfg", env):
        self.task_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.prompt_variant_ids = torch.zeros_like(self.task_ids)
        self._encoding = torch.zeros(env.num_envs, 16, device=env.device)
        self.prompt_tokens = torch.zeros(env.num_envs, cfg.max_prompt_tokens, dtype=torch.long, device=env.device)
        self._episode_number = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._fixed_task_ids = torch.full_like(self.task_ids, -1)
        self._pool = torch.tensor(split_task_ids(cfg.task_split), dtype=torch.long, device=env.device)
        super().__init__(cfg, env)

    @property
    def command(self):
        return self._encoding

    @property
    def prompts(self) -> list[str]:
        return [TASK_BY_ID[int(t)].prompts[int(v)] for t, v in zip(self.task_ids, self.prompt_variant_ids)]

    def set_task_ids(self, env_ids: Sequence[int] | torch.Tensor, task_ids: Sequence[int] | torch.Tensor):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        task_ids = torch.as_tensor(task_ids, dtype=torch.long, device=self.device)
        if torch.any((task_ids < 0) | (task_ids >= len(CATALOG))):
            raise ValueError(f"Task IDs must be in [0, {len(CATALOG) - 1}]")
        self._fixed_task_ids[env_ids] = task_ids
        self.task_ids[env_ids] = task_ids
        self._encode(env_ids)

    def clear_fixed_task_ids(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._fixed_task_ids[env_ids] = -1

    def _resample_command(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        indices = (env_ids + self._episode_number[env_ids] * self.num_envs) % len(self._pool)
        sampled = self._pool[indices]
        fixed = self._fixed_task_ids[env_ids] >= 0
        sampled[fixed] = self._fixed_task_ids[env_ids][fixed]
        self.task_ids[env_ids] = sampled
        self.prompt_variant_ids[env_ids] = self._episode_number[env_ids] % 2
        self._episode_number[env_ids] += 1
        self._encode(env_ids)

    def _encode(self, env_ids):
        self._encoding[env_ids] = 0.0
        for env_id in env_ids.tolist():
            task = TASK_BY_ID[int(self.task_ids[env_id])]
            self._encoding[env_id, 0] = task.task_id / max(1, len(CATALOG) - 1)
            for stage, choice in enumerate(task.code):
                self._encoding[env_id, 1 + stage * 3 + choice - 1] = 1.0
            self._encoding[env_id, 10] = float(task.family == "wash")
            self._encoding[env_id, 11:14] = torch.tensor(task.code, device=self.device) / 3.0
            self._encoding[env_id, 14] = 2.0 / 3.0
            self._encoding[env_id, 15] = 1.0
            prompt = task.prompts[int(self.prompt_variant_ids[env_id])]
            self.prompt_tokens[env_id] = torch.tensor(
                tokenize_prompt(prompt, self.prompt_tokens.shape[1]), dtype=torch.long, device=self.device
            )

    def _update_metrics(self):
        return

    def _update_command(self):
        return

    def _set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError


@configclass
class HouseholdTaskCommandCfg(CommandTermCfg):
    class_type: type[CommandTerm] = HouseholdTaskCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    task_split: str = "all"
    max_prompt_tokens: int = 32
