"""Record the unmodified official IsaacLab Franka lift-cube setup.

Only the viewport pose is changed for framing.  The robot, table, cube,
lighting, actions, observations, and reset logic come from the official task.
"""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seconds", type=float, default=8.0)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output-dir", default="outputs/official_setup")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


TASK_ID = "Isaac-Lift-Cube-Franka-IK-Rel-v0"


def main() -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=1)
    # The scene itself remains exactly official.  This camera is the Kit
    # viewport used for recording, not an additional sensor in the scene.
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (1.65, -1.55, 1.20)
    cfg.viewer.lookat = (0.40, 0.0, 0.34)
    cfg.viewer.resolution = (1280, 720)

    video_kwargs = {
        "video_folder": str(output_dir),
        "step_trigger": lambda step: step == 0,
        "video_length": max(1, round(args.seconds * args.fps)),
        "name_prefix": "official_franka_lift_cube",
        "disable_logger": True,
        "fps": args.fps,
    }
    env = gym.make(TASK_ID, cfg=cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    try:
        env.reset(seed=args.seed)
        action = torch.zeros(
            (env.unwrapped.num_envs, env.unwrapped.single_action_space.shape[0]),
            device=env.unwrapped.device,
        )
        for _ in range(video_kwargs["video_length"]):
            if not simulation_app.is_running():
                break
            env.step(action)
    finally:
        env.close()

    videos = sorted(output_dir.glob("official_franka_lift_cube*.mp4"), key=lambda path: path.stat().st_mtime)
    if not videos:
        raise RuntimeError(f"No video was written to {output_dir}")
    print(f"[INFO] Video: {videos[-1].resolve()}", flush=True)
    return videos[-1]


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
