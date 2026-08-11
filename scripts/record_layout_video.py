"""Record front-camera videos of the desk-service environment.

Produces both a single-environment view and a tiled parallel-environment view.
The privileged scripted policy is meant to exercise the scene, not to provide a
learned-policy benchmark.
"""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-CG-Household-Franka-IK-Rel-Vision-v0")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--task-ids", default="0,10,20,26", help="Comma-separated 27-task catalog IDs")
parser.add_argument("--output-dir", default="outputs/layout_videos")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import cg_isaac_envs  # noqa: E402, F401
from cg_isaac_envs.policies import ScriptedHouseholdPolicy  # noqa: E402


def _ffmpeg(path: Path, width: int, height: int):
    return subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(path),
        ],
        stdin=subprocess.PIPE,
    )


def _mosaic(frames: torch.Tensor) -> torch.Tensor:
    """Tile NHWC uint8 frames into a nearly square canvas."""
    count, height, width, channels = frames.shape
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    canvas = torch.zeros((rows * height, columns * width, channels), dtype=torch.uint8, device=frames.device)
    for index in range(count):
        row, column = divmod(index, columns)
        canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = frames[index]
    return canvas


def main():
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.scene.front_camera.width = args.width
    cfg.scene.front_camera.height = args.height
    env = gym.make(args.task, cfg=cfg).unwrapped
    observations, _ = env.reset(seed=args.seed)
    camera = env.scene["front_camera"]
    env_origins = env.scene.env_origins
    eye_local = torch.tensor([1.85, 0.0, 1.50], device=env.device).repeat(args.num_envs, 1)
    target_local = torch.tensor([0.45, 0.0, 0.18], device=env.device).repeat(args.num_envs, 1)
    camera.set_world_poses_from_view(env_origins + eye_local, env_origins + target_local)
    env.sim.render()
    requested_ids = [int(value) for value in args.task_ids.split(",") if value]
    assigned_ids = [requested_ids[index % len(requested_ids)] for index in range(args.num_envs)]
    command = env.command_manager.get_term("task")
    command.set_task_ids(torch.arange(args.num_envs, device=env.device), assigned_ids)
    policy = ScriptedHouseholdPolicy(env)
    camera_positions = env.scene["front_camera"].data.pos_w.detach().cpu().clone()
    camera_local_positions = camera_positions - env.scene.env_origins.detach().cpu()

    columns = math.ceil(math.sqrt(args.num_envs))
    rows = math.ceil(args.num_envs / columns)
    single_path = output_dir / "household_single.mp4"
    parallel_path = output_dir / f"household_parallel_{args.num_envs}.mp4"
    single_writer = _ffmpeg(single_path, args.width, args.height)
    parallel_writer = _ffmpeg(parallel_path, columns * args.width, rows * args.height)

    control_hz = 1.0 / (cfg.sim.dt * cfg.decimation)
    control_steps = max(1, round(args.seconds * control_hz))
    capture_every = max(1, round(control_hz / args.fps))
    frames_written = 0
    try:
        with torch.inference_mode():
            for step in range(control_steps):
                if not simulation_app.is_running():
                    break
                actions = policy.act(observations)
                observations, _, terminated, truncated, _ = env.step(actions)
                done_ids = (terminated | truncated).nonzero().flatten()
                if done_ids.numel():
                    policy.reset(done_ids)
                if step % capture_every:
                    continue
                frames = observations["policy"]["front_rgb"][..., :3].clamp(0, 255).to(torch.uint8)
                single_writer.stdin.write(frames[0].contiguous().cpu().numpy().tobytes())
                parallel_writer.stdin.write(_mosaic(frames).contiguous().cpu().numpy().tobytes())
                frames_written += 1
    finally:
        for writer in (single_writer, parallel_writer):
            if writer.stdin:
                writer.stdin.close()
            return_code = writer.wait()
            if return_code:
                raise RuntimeError(f"ffmpeg exited with status {return_code}")
        env.close()

    metadata = {
        "front_camera_prim": "{ENV_REGEX_NS}/FrontCamera",
        "front_camera_world_positions": camera_positions.tolist(),
        "front_camera_local_positions": camera_local_positions.tolist(),
        "task": args.task,
        "task_ids": assigned_ids,
        "visible_objects": ["YCB bowl", "YCB plate", "YCB serving pan", "YCB apple", "YCB banana", "YCB snack package", "three place targets"],
        "num_envs": args.num_envs,
        "seed": args.seed,
        "frames": frames_written,
        "fps": args.fps,
        "policy": "privileged scripted waypoint controller",
        "videos": [single_path.name, parallel_path.name],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
