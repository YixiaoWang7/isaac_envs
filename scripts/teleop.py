"""Teleoperate the desk-service Franka with keyboard or SpaceMouse."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-CG-Household-Franka-IK-Rel-Vision-v0")
parser.add_argument("--task-id", type=int)
parser.add_argument("--teleop-device", choices=("spacemouse", "keyboard"), default="spacemouse")
parser.add_argument("--spacemouse-backend", choices=("hid", "input"), default="hid")
parser.add_argument("--spacemouse-device", help="Event node for the legacy input backend, e.g. /dev/input/event4")
parser.add_argument("--sensitivity", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from omni.kit.viewport.utility import get_active_viewport  # noqa: E402
from omni.kit.viewport.window import ViewportWindow  # noqa: E402

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import cg_isaac_envs  # noqa: E402, F401
from cg_isaac_envs.devices import LinuxSpaceMouse, RawHidSpaceMouse  # noqa: E402


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.terminations.time_out = None
    env = gym.make(args.task, cfg=cfg).unwrapped
    reset_requested = False

    def request_reset():
        nonlocal reset_requested
        reset_requested = True

    if args.teleop_device == "spacemouse":
        if args.spacemouse_backend == "input":
            if not args.spacemouse_device:
                raise ValueError("--spacemouse-device is required with --spacemouse-backend input")
            device = LinuxSpaceMouse(
                args.spacemouse_device,
                sim_device=env.device,
                pos_sensitivity=0.08 * args.sensitivity,
                rot_sensitivity=0.05 * args.sensitivity,
            )
        else:
            device = RawHidSpaceMouse(
                sim_device=env.device,
                pos_sensitivity=0.08 * args.sensitivity,
                rot_sensitivity=0.05 * args.sensitivity,
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
    # Keep the overview in the normal viewport and open wrist and side feeds
    # in dedicated windows so all three remain visible during teleoperation.
    viewport = get_active_viewport()
    overview_camera = "/World/envs/env_0/FrontCamera"
    wrist_camera = "/World/envs/env_0/Robot/panda_hand/WristCamera"
    side_camera = "/World/envs/env_0/SideCamera"
    viewport.set_active_camera(overview_camera)
    wrist_viewport_window = ViewportWindow("Wrist Camera", width=640, height=480)
    wrist_viewport_window.viewport_api.set_active_camera(wrist_camera)
    side_viewport_window = ViewportWindow("Side Camera", width=640, height=480)
    side_viewport_window.viewport_api.set_active_camera(side_camera)

    # The gripper optical/tool axis points down at roll=pi, pitch=0. Yaw is
    # the only orientation degree of freedom exposed during teleoperation.
    target_yaw = torch.zeros(1, device=env.device)
    orientation_gain = 2.0
    print(device)
    print("Prompt:", env.command_manager.get_term("task").prompts[0])
    print("Orientation: straight down (roll/pitch locked); twist controls yaw")
    print("Cameras: overview, wrist, and robot-right side views are open simultaneously")
    with torch.inference_mode():
        while simulation_app.is_running():
            action = device.advance()
            target_yaw += action[5:6] * 0.5
            # Ignore raw roll/pitch and replace all angular commands with an
            # orientation error that continuously restores the down-facing pose.
            current_quat = env.scene["ee_frame"].data.target_quat_w[:, 0]
            desired_quat = math_utils.quat_from_euler_xyz(
                torch.full_like(target_yaw, torch.pi), torch.zeros_like(target_yaw), target_yaw
            )
            _, rotation_error = math_utils.compute_pose_error(
                torch.zeros((1, 3), device=env.device), current_quat,
                torch.zeros((1, 3), device=env.device), desired_quat,
                rot_error_type="axis_angle",
            )
            action[3:6] = torch.clamp(rotation_error[0] * orientation_gain, -1.0, 1.0)
            env.step(action[None, :])
            if reset_requested:
                env.reset()
                device.reset()
                target_yaw.zero_()
                reset_requested = False
                print("Environment reset.")
                print("Prompt:", env.command_manager.get_term("task").prompts[0])
    wrist_viewport_window.destroy()
    side_viewport_window.destroy()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
