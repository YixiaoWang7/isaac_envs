"""Task and grasp-mode context for demonstration datasets."""

from __future__ import annotations

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from .observations import object_relative_gripper_poses
from .relations import stage_results, task_term


class HouseholdContextRecorder(RecorderTerm):
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
        return "household_progress", {
            "stages": stage,
            "applicable": applicable,
            "object_relative_gripper_poses": object_relative_gripper_poses(self._env),
        }


@configclass
class HouseholdContextRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = HouseholdContextRecorder
