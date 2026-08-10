"""Task-context demonstration recorder."""

from __future__ import annotations

import torch

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from .relations import stage_results, task_term


class TaskContextRecorder(RecorderTerm):
    def record_post_reset(self, env_ids):
        command = task_term(self._env)
        return "task", {
            "task_id": command.task_ids[:, None],
            "prompt_variant_id": command.prompt_variant_ids[:, None],
            "encoding": command.command,
            "prompt_tokens": command.prompt_tokens,
        }

    def record_post_step(self):
        stage, applicable = stage_results(self._env)
        return "task_progress", {"stages": stage, "applicable": applicable}


@configclass
class TaskContextRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = TaskContextRecorder

