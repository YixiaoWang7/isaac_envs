"""Simulator smoke test: registration, reset, and 100 zero-action steps."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import cg_isaac_envs  # noqa: E402, F401

TASK = "Isaac-CG-Household-Franka-IK-Rel-v0"
cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
env = gym.make(TASK, cfg=cfg).unwrapped
observations, _ = env.reset(seed=0)
assert env.action_manager.total_action_dim == 7
assert observations["policy"]["task_encoding"].shape == (args.num_envs, 16)
for _ in range(100):
    action = torch.zeros(args.num_envs, 7, device=env.device)
    action[:, 6] = 1.0
    env.step(action)
print("smoke_env: PASS")
env.close()
simulation_app.close()

