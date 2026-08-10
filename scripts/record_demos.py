"""Record successful prompt-conditioned demonstrations to HDF5."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-CG-Household-Franka-IK-Rel-Vision-v0")
parser.add_argument("--task-id", type=int)
parser.add_argument("--task-split", default="all")
parser.add_argument("--teleop-device", choices=("spacemouse", "keyboard"), default="spacemouse")
parser.add_argument("--dataset-file", default="datasets/desk_service.hdf5")
parser.add_argument("--num-demos", type=int, default=0)
parser.add_argument("--num-success-steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg  # noqa: E402
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import cg_isaac_envs  # noqa: E402, F401
from cg_isaac_envs.tasks.household.mdp.recorders import HouseholdContextRecorderCfg  # noqa: E402
from cg_isaac_envs.tasks.household.mdp.relations import alternative_results  # noqa: E402


@configclass
class DeskRecorderCfg(ActionStateRecorderManagerCfg):
    task_context = HouseholdContextRecorderCfg()


def make_device(env, reset_callback):
    if args.teleop_device == "spacemouse":
        device = Se3SpaceMouse(
            Se3SpaceMouseCfg(pos_sensitivity=0.05, rot_sensitivity=0.05, gripper_term=True, sim_device=env.device)
        )
    else:
        device = Se3Keyboard(
            Se3KeyboardCfg(pos_sensitivity=0.05, rot_sensitivity=0.05, gripper_term=True, sim_device=env.device)
        )
    device.add_callback("R", reset_callback)
    return device


def main():
    dataset_path = Path(args.dataset_file).resolve()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.commands.task.task_split = args.task_split
    cfg.terminations.success = None
    cfg.terminations.time_out = None
    cfg.observations.policy.concatenate_terms = False
    cfg.recorders = DeskRecorderCfg()
    cfg.recorders.dataset_export_dir_path = str(dataset_path.parent)
    cfg.recorders.dataset_filename = dataset_path.stem
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env = gym.make(args.task, cfg=cfg).unwrapped
    reset_requested = False

    def request_reset():
        nonlocal reset_requested
        reset_requested = True

    teleop = make_device(env, request_reset)
    env.reset()
    if args.task_id is not None:
        env.command_manager.get_term("task").set_task_ids([0], [args.task_id])
        env.reset()
    teleop.reset()
    success_steps = 0
    print(teleop)
    print("Prompt:", env.command_manager.get_term("task").prompts[0])
    with torch.inference_mode():
        while simulation_app.is_running():
            env.step(teleop.advance()[None, :])
            if bool(alternative_results(env)[0][0]):
                success_steps += 1
            else:
                success_steps = 0
            if success_steps >= args.num_success_steps:
                env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                env.recorder_manager.set_success_to_episodes([0], torch.ones((1, 1), dtype=torch.bool, device=env.device))
                env.recorder_manager.export_episodes([0])
                print("Recorded successful demo", env.recorder_manager.exported_successful_episode_count)
                reset_requested = True
            if args.num_demos and env.recorder_manager.exported_successful_episode_count >= args.num_demos:
                break
            if reset_requested:
                if success_steps < args.num_success_steps:
                    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                env.reset()
                teleop.reset()
                success_steps = 0
                reset_requested = False
                print("Prompt:", env.command_manager.get_term("task").prompts[0])
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

