"""Physically pick and lift every movable household asset in parallel."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-CG-Household-Franka-IK-Rel-v0")
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--lift-threshold", type=float, default=0.035)
parser.add_argument("--output", default="outputs/grasp_qualification.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json  # noqa: E402
from pathlib import Path  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import cg_isaac_envs  # noqa: E402, F401
from cg_isaac_envs.policies.household_scripted import Operation, ScriptedHouseholdPolicy  # noqa: E402
from cg_isaac_envs.tasks.household.task_catalog import OBJECT_NAMES  # noqa: E402


def main():
    objects = list(OBJECT_NAMES)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=len(objects))
    cfg.episode_length_s = max(cfg.episode_length_s, args.seconds + 2.0)
    env = gym.make(args.task, cfg=cfg).unwrapped
    observations, _ = env.reset(seed=0)
    policy = ScriptedHouseholdPolicy(env)
    for env_id, subject in enumerate(objects):
        policy.plans[env_id] = [Operation(subject, "on", "left_place")]
    policy.operation_index.zero_()
    policy.phase.zero_()
    policy.dwell.zero_()
    start_z = torch.stack([env.scene[name].data.root_pos_w[index, 2] for index, name in enumerate(objects)])
    peak_z = start_z.clone()
    max_phase = policy.phase.clone()
    control_hz = 1.0 / (cfg.sim.dt * cfg.decimation)
    with torch.inference_mode():
        for _ in range(round(args.seconds * control_hz)):
            actions = policy.act(observations)
            observations, _, _, _, _ = env.step(actions)
            max_phase = torch.maximum(max_phase, policy.phase)
            for env_id, subject in enumerate(objects):
                peak_z[env_id] = torch.maximum(peak_z[env_id], env.scene[subject].data.root_pos_w[env_id, 2])
            if torch.all(peak_z - start_z >= args.lift_threshold):
                break
    lifts = (peak_z - start_z).detach().cpu()
    results = {
        name: {"passed": bool(lift >= args.lift_threshold), "lift_m": round(float(lift), 5), "max_phase": int(max_phase[index])}
        for index, (name, lift) in enumerate(zip(objects, lifts))
    }
    report = {
        "criterion": f"object root rises at least {args.lift_threshold:.3f} m after physical finger closure",
        "all_passed": all(item["passed"] for item in results.values()),
        "objects": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
