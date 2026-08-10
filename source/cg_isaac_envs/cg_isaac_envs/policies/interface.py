"""Small policy interface used by the parallel evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class TaskBatch:
    task_ids: torch.Tensor
    prompt_variant_ids: torch.Tensor
    encodings: torch.Tensor
    prompt_tokens: torch.Tensor
    prompts: list[str]


class Policy(Protocol):
    def reset(self, env_ids: torch.Tensor, tasks: TaskBatch) -> None: ...

    def act(self, observations: dict, tasks: TaskBatch) -> torch.Tensor: ...

