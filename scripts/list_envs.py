"""List CG environments. Launch Isaac Sim before importing the extension."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import cg_isaac_envs  # noqa: E402, F401

for spec in gym.registry.values():
    if spec.id.startswith("Isaac-CG-"):
        print(f"{spec.id}: {spec.kwargs['env_cfg_entry_point']}")

simulation_app.close()

