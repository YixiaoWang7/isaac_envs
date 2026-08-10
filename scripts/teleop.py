"""Teleoperate the desk-service Franka with keyboard or SpaceMouse."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-CG-Household-Franka-IK-Rel-v0")
parser.add_argument("--task-id", type=int)
parser.add_argument("--teleop-device", choices=("spacemouse", "keyboard"), default="spacemouse")
parser.add_argument("--sensitivity", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import cg_isaac_envs  # noqa: E402, F401


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.terminations.time_out = None
    env = gym.make(args.task, cfg=cfg).unwrapped
    reset_requested = False

    def request_reset():
        nonlocal reset_requested
        reset_requested = True

    if args.teleop_device == "spacemouse":
        device = Se3SpaceMouse(
            Se3SpaceMouseCfg(
                pos_sensitivity=0.05 * args.sensitivity,
                rot_sensitivity=0.05 * args.sensitivity,
                gripper_term=True,
                sim_device=env.device,
            )
        )
    else:
        device = Se3Keyboard(
            Se3KeyboardCfg(
                pos_sensitivity=0.05 * args.sensitivity,
                rot_sensitivity=0.05 * args.sensitivity,
                gripper_term=True,
                sim_device=env.device,
            )
        )
    device.add_callback("R", request_reset)
    env.reset()
    if args.task_id is not None:
        env.command_manager.get_term("task").set_task_ids([0], [args.task_id])
        env.reset()
    device.reset()
    print(device)
    print("Prompt:", env.command_manager.get_term("task").prompts[0])
    with torch.inference_mode():
        while simulation_app.is_running():
            env.step(device.advance()[None, :])
            if reset_requested:
                env.reset()
                device.reset()
                reset_requested = False
                print("Environment reset.")
                print("Prompt:", env.command_manager.get_term("task").prompts[0])
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
