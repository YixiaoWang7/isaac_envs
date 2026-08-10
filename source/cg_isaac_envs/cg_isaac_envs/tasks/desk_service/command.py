"""Prompt-conditioned command term."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .task_catalog import CATALOG, TASK_BY_ID, tokenize_prompt, split_task_ids


class DeskTaskCommand(CommandTerm):
    """Samples semantic tasks independently for every parallel environment."""

    cfg: "DeskTaskCommandCfg"

    def __init__(self, cfg: "DeskTaskCommandCfg", env):
        self.task_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.prompt_variant_ids = torch.zeros_like(self.task_ids)
        self._encoding = torch.zeros(env.num_envs, 16, device=env.device)
        self.prompt_tokens = torch.zeros(env.num_envs, cfg.max_prompt_tokens, dtype=torch.long, device=env.device)
        self._episode_number = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._fixed_task_ids = torch.full_like(self.task_ids, -1)
        self._pool = torch.tensor(split_task_ids(cfg.task_split), dtype=torch.long, device=env.device)
        super().__init__(cfg, env)

    @property
    def command(self) -> torch.Tensor:
        return self._encoding

    @property
    def prompts(self) -> list[str]:
        return [TASK_BY_ID[int(tid)].prompts[int(variant)] for tid, variant in zip(self.task_ids, self.prompt_variant_ids)]

    def set_task_ids(self, env_ids: Sequence[int] | torch.Tensor, task_ids: Sequence[int] | torch.Tensor) -> None:
        """Pin tasks for selected environments, primarily for teleoperation."""

        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        task_ids_t = torch.as_tensor(task_ids, dtype=torch.long, device=self.device)
        if torch.any((task_ids_t < 0) | (task_ids_t >= len(CATALOG))):
            raise ValueError(f"Task IDs must be in [0, {len(CATALOG) - 1}]")
        self._fixed_task_ids[env_ids_t] = task_ids_t
        self.task_ids[env_ids_t] = task_ids_t
        self._encode(env_ids_t)

    def clear_fixed_task_ids(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._fixed_task_ids[env_ids_t] = -1

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids_t.numel() == 0:
            return
        pool_indices = (env_ids_t + self._episode_number[env_ids_t] * self.num_envs) % len(self._pool)
        sampled = self._pool[pool_indices]
        fixed_mask = self._fixed_task_ids[env_ids_t] >= 0
        sampled[fixed_mask] = self._fixed_task_ids[env_ids_t][fixed_mask]
        self.task_ids[env_ids_t] = sampled
        self.prompt_variant_ids[env_ids_t] = self._episode_number[env_ids_t] % 2
        self._episode_number[env_ids_t] += 1
        self._encode(env_ids_t)

    def _encode(self, env_ids: torch.Tensor) -> None:
        family_index = {"tea": 0, "candy": 1, "combo": 2, "takeaway": 3}
        self._encoding[env_ids] = 0.0
        for env_id in env_ids.tolist():
            task = TASK_BY_ID[int(self.task_ids[env_id])]
            factors = task.factor_dict
            self._encoding[env_id, 0] = task.task_id / max(1, len(CATALOG) - 1)
            self._encoding[env_id, 1 + family_index[task.family]] = 1.0
            self._encoding[env_id, 5 if factors["destination"] == "left" else 6] = 1.0
            self._encoding[env_id, 7] = float(factors.get("sweetener") == "sugar")
            self._encoding[env_id, 8] = float(factors.get("spoon") == "yes")
            self._encoding[env_id, 9] = float(factors.get("napkin") == "yes")
            self._encoding[env_id, 10] = float(factors.get("cookie") == "yes")
            candy = factors.get("candies", factors.get("candy", ""))
            self._encoding[env_id, 11] = float(candy in {"red", "both"})
            self._encoding[env_id, 12] = float(candy in {"blue", "both"})
            self._encoding[env_id, 13] = float(task.family == "candy")
            self._encoding[env_id, 14] = len(task.alternatives)
            self._encoding[env_id, 15] = len(task.alternatives[0]) / 12.0
            prompt = task.prompts[int(self.prompt_variant_ids[env_id])]
            self.prompt_tokens[env_id] = torch.tensor(
                tokenize_prompt(prompt, self.cfg.max_prompt_tokens), dtype=torch.long, device=self.device
            )

    def _update_metrics(self) -> None:
        return

    def _update_command(self) -> None:
        return

    def _set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError


@configclass
class DeskTaskCommandCfg(CommandTermCfg):
    """Configuration for :class:`DeskTaskCommand`."""

    class_type: type[CommandTerm] = DeskTaskCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    task_split: str = "all"
    max_prompt_tokens: int = 32

