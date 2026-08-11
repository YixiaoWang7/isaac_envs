"""Collect balanced, prompt-conditioned household demonstrations."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-CG-Household-Franka-IK-Rel-Vision-v0")
parser.add_argument("--task-id", type=int, help="Collect only one task instead of the full catalog")
parser.add_argument("--teleop-device", choices=("spacemouse", "keyboard"), default="spacemouse")
parser.add_argument("--spacemouse-backend", choices=("hid", "input"), default="hid")
parser.add_argument("--spacemouse-device", help="Linux input-event node, e.g. /dev/input/event4")
parser.add_argument("--sensitivity", type=float, default=1.0)
parser.add_argument("--dataset-dir", required=True, help="Directory containing dataset.hdf5 and videos/")
parser.add_argument("--demos-per-task", type=int, default=20)
parser.add_argument("--num-success-steps", type=int, default=10)
parser.add_argument("--seed", type=int, default=0, help="Reproducible household layout randomization seed")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import omni.ui as ui  # noqa: E402
import torch  # noqa: E402
from omni.kit.viewport.utility import get_active_viewport  # noqa: E402
from omni.kit.viewport.window import ViewportWindow  # noqa: E402

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import cg_isaac_envs  # noqa: E402, F401
from cg_isaac_envs.data_collection import (  # noqa: E402
    BalancedTaskScheduler,
    DemonstrationDataset,
    EpisodeMetadata,
)
from cg_isaac_envs.devices import LinuxSpaceMouse, RawHidSpaceMouse  # noqa: E402
from cg_isaac_envs.tasks.household.mdp.relations import alternative_results  # noqa: E402
from cg_isaac_envs.tasks.household.mdp.terminations import required_object_dropped  # noqa: E402
from cg_isaac_envs.tasks.household.task_catalog import CATALOG, OBJECT_NAMES, TASK_BY_ID  # noqa: E402


CAMERA_SCENE_NAMES = {"front": "front_camera", "wrist": "wrist_camera", "side": "side_camera"}


class PromptPanel:
    """Small operator panel updated from the simulation thread."""

    def __init__(self):
        self.window = ui.Window("Demonstration Collection", width=640, height=230)
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                self.task_label = ui.Label("Task", height=22)
                self.prompt_label = ui.Label("", word_wrap=True, height=86)
                self.progress_label = ui.Label("Progress", height=22)
                self.status_label = ui.Label("Status", height=22)
                ui.Label("Button 1 = gripper; button 2 = start when READY, discard/reset when RECORDING", height=24)

    def update(self, task_id, task_code, prompt, task_count, quota, accepted, target, status):
        self.task_label.text = f"Task {task_id:02d}  |  stages {task_code}"
        self.prompt_label.text = prompt
        self.progress_label.text = (
            f"Task progress: {task_count}/{quota}     Overall: {accepted}/{target} successful demonstrations"
        )
        self.status_label.text = f"Status: {status}"

    def destroy(self):
        self.window.destroy()


def make_device(env, reset_callback):
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
    device.add_callback("R", reset_callback)
    return device


def task_rows():
    return [
        {
            "task_id": task.task_id,
            "code": list(task.code),
            "prompts": list(task.prompts),
            "selected_object": task.selected_object,
            "mug": task.associated_object,
            "destination": task.destination,
        }
        for task in CATALOG
    ]


def camera_images(env) -> dict[str, np.ndarray]:
    images = {}
    for output_name, scene_name in CAMERA_SCENE_NAMES.items():
        tensor = env.scene[scene_name].data.output["rgb"][0, ..., :3]
        images[output_name] = tensor.clamp(0, 255).to(torch.uint8).cpu().numpy()
    return images


def main():
    if args.num_success_steps <= 0:
        raise ValueError("--num-success-steps must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    selected_task_ids = [args.task_id] if args.task_id is not None else list(range(len(CATALOG)))
    if any(task_id not in TASK_BY_ID for task_id in selected_task_ids):
        raise ValueError(f"Task ID must be in [0, {len(CATALOG) - 1}]")

    dataset = DemonstrationDataset(
        args.dataset_dir, task_rows(), fps=30, resolution=(256, 256), collection_seed=args.seed
    )
    dataset.set_requested_quota(args.demos_per_task)
    scheduler = BalancedTaskScheduler(selected_task_ids, dataset.task_counts(), args.demos_per_task)
    if scheduler.complete:
        print(f"Dataset quota is already complete: {scheduler.accepted}/{scheduler.target}")
        dataset.close()
        return

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.seed = args.seed
    cfg.terminations.success = None
    cfg.terminations.dropped = None
    cfg.terminations.time_out = None
    cfg.observations.policy.concatenate_terms = False
    for camera_name in CAMERA_SCENE_NAMES.values():
        camera_cfg = getattr(cfg.scene, camera_name)
        camera_cfg.width = 256
        camera_cfg.height = 256
    env = gym.make(args.task, cfg=cfg).unwrapped

    reset_requested = False

    def request_reset():
        nonlocal reset_requested
        reset_requested = True

    teleop = make_device(env, request_reset)
    panel = PromptPanel()

    viewport = get_active_viewport()
    viewport.set_active_camera("/World/envs/env_0/FrontCamera")
    wrist_window = ViewportWindow("Wrist Camera", width=640, height=480)
    wrist_window.viewport_api.set_active_camera("/World/envs/env_0/Robot/panda_hand/WristCamera")
    side_window = ViewportWindow("Side Camera", width=640, height=480)
    side_window.viewport_api.set_active_camera("/World/envs/env_0/SideCamera")

    arm_term = env.action_manager.get_term("arm_action")
    robot = env.scene["robot"]
    finger_joint_ids, _ = robot.find_joints(["panda_finger_joint.*"])
    task_command = env.command_manager.get_term("task")
    target_yaw = torch.zeros(1, device=env.device)
    orientation_gain = 2.0
    success_steps = 0
    current_task_id = None
    current_layout_index = None
    attempt = None
    recording = False

    def update_panel(status):
        task = TASK_BY_ID[current_task_id]
        panel.update(
            current_task_id,
            task.code,
            task_command.prompts[0],
            scheduler.counts[current_task_id],
            scheduler.quota,
            scheduler.accepted,
            scheduler.target,
            status,
        )

    def reset_current_task():
        nonlocal current_layout_index, attempt, success_steps, reset_requested, recording
        if attempt is not None:
            attempt.discard()
            attempt = None
        task_command.set_task_ids([0], [current_task_id])
        current_layout_index = dataset.allocate_layout_index()
        if not hasattr(env, "_household_reset_counts"):
            env._household_reset_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._household_reset_counts[0] = current_layout_index
        env.reset()
        teleop.reset()
        target_yaw.zero_()
        success_steps = 0
        reset_requested = False
        recording = False
        update_panel("READY — press button 2 to start")
        print(
            f"Task {current_task_id:02d} [{scheduler.counts[current_task_id]}/{scheduler.quota}]: "
            f"{task_command.prompts[0]}"
        )
        print("READY: inspect the task, then press SpaceMouse button 2 to start recording.")

    def prepare_next_task():
        nonlocal current_task_id
        current_task_id = scheduler.next_task()
        if current_task_id is None:
            return False
        reset_current_task()
        return True

    def start_recording():
        nonlocal attempt, success_steps, reset_requested, recording
        teleop.reset()
        target_yaw.zero_()
        success_steps = 0
        reset_requested = False
        task = TASK_BY_ID[current_task_id]
        attempt = dataset.new_attempt(
            EpisodeMetadata(
                task_id=current_task_id,
                task_code=task.code,
                prompt=task_command.prompts[0],
                prompt_variant_id=int(task_command.prompt_variant_ids[0]),
                layout_index=current_layout_index,
            )
        )
        recording = True
        update_panel("RECORDING")
        print("RECORDING: press SpaceMouse button 2 to discard and reset this task.")

    print(teleop)
    print("Dataset:", dataset.hdf5_path)
    try:
        if not prepare_next_task():
            return
        with torch.inference_mode():
            while simulation_app.is_running() and not scheduler.complete:
                if not recording:
                    if reset_requested:
                        start_recording()
                        continue
                    # Keep the GUI, cameras, and scene responsive without
                    # recording or accepting SpaceMouse motion in READY mode.
                    ready_action = torch.zeros((1, 7), device=env.device)
                    ready_action[:, 6] = 1.0
                    env.step(ready_action)
                    if bool(required_object_dropped(env)[0]):
                        print("Initial scene became invalid; resetting before recording.")
                        reset_current_task()
                    continue

                if reset_requested:
                    print(f"Discarded task {current_task_id:02d} attempt (operator reset).")
                    reset_current_task()
                    continue

                # Observation at time t, before applying action t.
                ee_pos_b, ee_quat_b = arm_term._compute_frame_pose()
                ee_pose = torch.cat((ee_pos_b[0], ee_quat_b[0])).cpu().numpy()
                gripper_width = float(robot.data.joint_pos[0, finger_joint_ids].sum().item())
                images = camera_images(env)

                action = teleop.advance()
                target_yaw += action[5:6] * 0.5
                current_quat = env.scene["ee_frame"].data.target_quat_w[:, 0]
                desired_quat = math_utils.quat_from_euler_xyz(
                    torch.full_like(target_yaw, torch.pi), torch.zeros_like(target_yaw), target_yaw
                )
                _, rotation_error = math_utils.compute_pose_error(
                    torch.zeros((1, 3), device=env.device),
                    current_quat,
                    torch.zeros((1, 3), device=env.device),
                    desired_quat,
                    rot_error_type="axis_angle",
                )
                action[3:6] = torch.clamp(rotation_error[0] * orientation_gain, -1.0, 1.0)

                env.step(action[None, :])
                ee_delta_pose = arm_term.processed_actions[0].detach().cpu().numpy()
                gripper_target = int(action[6].item() > 0.0)
                task_success = bool(alternative_results(env)[0][0])
                speeds = torch.stack(
                    [torch.linalg.vector_norm(env.scene[name].data.root_lin_vel_w[0]) for name in OBJECT_NAMES]
                )
                stable_now = task_success and bool(speeds.amax() < 0.08)
                success_steps = success_steps + 1 if stable_now else 0
                stable_success = success_steps >= args.num_success_steps

                attempt.append(
                    images=images,
                    ee_pose=ee_pose,
                    gripper_width=gripper_width,
                    ee_delta_pose=ee_delta_pose,
                    gripper_target=gripper_target,
                    task_success=task_success,
                    stable_success=stable_success,
                )

                if reset_requested:
                    print(f"Discarded task {current_task_id:02d} attempt (operator reset).")
                    reset_current_task()
                elif bool(required_object_dropped(env)[0]):
                    print(f"Discarded task {current_task_id:02d} attempt (required object dropped).")
                    reset_current_task()
                elif stable_success:
                    demo_name = attempt.commit()
                    scheduler.mark_success(current_task_id)
                    print(
                        f"Saved {demo_name}: task {current_task_id:02d} "
                        f"[{scheduler.counts[current_task_id]}/{scheduler.quota}]"
                    )
                    update_panel("SUCCESS — saved")
                    attempt = None
                    recording = False
                    if not prepare_next_task():
                        break
        print(f"Collection complete: {scheduler.accepted}/{scheduler.target} successful demonstrations")
    finally:
        if attempt is not None:
            attempt.discard()
        if hasattr(teleop, "close"):
            teleop.close()
        wrist_window.destroy()
        side_window.destroy()
        panel.destroy()
        env.close()
        dataset.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
