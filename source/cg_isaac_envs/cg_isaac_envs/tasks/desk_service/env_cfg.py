"""Compatibility exports for IsaacLab's unmodified official Franka lift task.

No scene entities or configuration values are overridden here.  The robot,
Seattle Lab table, cube, lighting, reset events, observations, and actions all
come directly from IsaacLab.
"""

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import (
    FrankaCubeLiftEnvCfg,
    FrankaCubeLiftEnvCfg_PLAY,
)

__all__ = ["FrankaCubeLiftEnvCfg", "FrankaCubeLiftEnvCfg_PLAY"]
