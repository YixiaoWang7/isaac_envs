"""Gym registrations for the realistic household benchmark."""

import gymnasium as gym

from .task_catalog import *  # noqa: F401, F403

_REGISTRATIONS = {
    "Isaac-CG-Household-Franka-IK-Rel-v0": "HouseholdEnvCfg",
    "Isaac-CG-Household-Franka-IK-Rel-Vision-v0": "HouseholdVisionEnvCfg",
    "Isaac-CG-DeskService-Franka-IK-Rel-v0": "HouseholdEnvCfg",
    "Isaac-CG-DeskService-Franka-IK-Rel-Vision-v0": "HouseholdVisionEnvCfg",
}

for env_id, cfg_name in _REGISTRATIONS.items():
    if env_id not in gym.registry:
        gym.register(
            id=env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:{cfg_name}"},
            disable_env_checker=True,
        )
