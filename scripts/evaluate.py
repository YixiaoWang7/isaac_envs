"""Parallel prompt-conditioned policy evaluation."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-CG-Household-Franka-IK-Rel-v0")
parser.add_argument("--task-split", default="all", choices=("all", "orthogonal_train", "id_eval", "ood_recombination"))
parser.add_argument("--policy", default="scripted", help="scripted, zero, random, or module:factory")
parser.add_argument("--checkpoint")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--episodes-per-task", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output-dir", default="outputs/evaluation")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import csv  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import cg_isaac_envs  # noqa: E402, F401
from cg_isaac_envs.policies import ScriptedHouseholdPolicy, TaskBatch  # noqa: E402
from cg_isaac_envs.tasks.household.mdp.relations import stage_results  # noqa: E402
from cg_isaac_envs.tasks.household.task_catalog import STAGE_NAMES, TASK_BY_ID, split_task_ids  # noqa: E402


class ZeroPolicy:
    def __init__(self, env):
        self.env = env

    def reset(self, env_ids, tasks):
        return

    def act(self, observations, tasks):
        actions = torch.zeros(self.env.num_envs, 7, device=self.env.device)
        actions[:, 6] = 1.0
        return actions


class RandomPolicy(ZeroPolicy):
    def act(self, observations, tasks):
        actions = torch.rand(self.env.num_envs, 7, device=self.env.device) * 2.0 - 1.0
        actions[:, :6] *= 0.2
        return actions


def task_batch(env) -> TaskBatch:
    command = env.command_manager.get_term("task")
    return TaskBatch(
        task_ids=command.task_ids.clone(),
        prompt_variant_ids=command.prompt_variant_ids.clone(),
        encodings=command.command.clone(),
        prompt_tokens=command.prompt_tokens.clone(),
        prompts=command.prompts,
    )


def load_policy(env):
    if args.policy == "scripted":
        return ScriptedHouseholdPolicy(env)
    if args.policy == "zero":
        return ZeroPolicy(env)
    if args.policy == "random":
        return RandomPolicy(env)
    if ":" not in args.policy:
        raise ValueError("External policy must use module:factory syntax")
    module_name, factory_name = args.policy.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    return factory(env=env, checkpoint=args.checkpoint, device=env.device)


def summarize(rows: list[dict]) -> dict:
    def group(keys):
        buckets = defaultdict(list)
        for row in rows:
            buckets[tuple(row[key] for key in keys)].append(row)
        result = {}
        for values, items in buckets.items():
            label = "/".join(str(value) for value in values)
            result[label] = {
                "episodes": len(items),
                "success_rate": sum(int(item["success"]) for item in items) / len(items),
                "mean_steps": sum(int(item["steps"]) for item in items) / len(items),
            }
        return result

    failures = Counter(row["failure"] for row in rows if not row["success"])
    return {
        "episodes": len(rows),
        "success_rate": sum(int(row["success"]) for row in rows) / max(1, len(rows)),
        "by_task": group(["task_id"]),
        "by_family": group(["family"]),
        "by_side": group(["side"]),
        "failures": dict(failures),
    }


def main():
    torch.manual_seed(args.seed)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.commands.task.task_split = args.task_split
    env = gym.make(args.task, cfg=cfg).unwrapped
    observations, _ = env.reset(seed=args.seed)
    policy = load_policy(env)
    tasks = task_batch(env)
    policy.reset(torch.arange(env.num_envs, device=env.device), tasks)
    task_pool = set(split_task_ids(args.task_split))
    counts = Counter()
    episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    rows: list[dict] = []

    with torch.inference_mode():
        while simulation_app.is_running() and any(counts[task_id] < args.episodes_per_task for task_id in task_pool):
            previous_tasks = task_batch(env)
            previous_stages, previous_applicable = stage_results(env)
            actions = policy.act(observations, previous_tasks)
            observations, _, terminated, truncated, _ = env.step(actions)
            episode_steps += 1
            done = terminated | truncated
            if torch.any(done):
                success_term = env.termination_manager.get_term("success")
                dropped_term = env.termination_manager.get_term("dropped")
                for env_id in done.nonzero().flatten().tolist():
                    task_id = int(previous_tasks.task_ids[env_id])
                    if task_id not in task_pool or counts[task_id] >= args.episodes_per_task:
                        continue
                    success = bool(success_term[env_id])
                    dropped = bool(dropped_term[env_id])
                    task = TASK_BY_ID[task_id]
                    factors = task.factor_dict
                    stages = previous_stages[env_id].clone()
                    applicable = previous_applicable[env_id]
                    if success:
                        stages[applicable] = True
                    incomplete = [
                        STAGE_NAMES[index]
                        for index in range(len(STAGE_NAMES))
                        if bool(applicable[index]) and not bool(stages[index])
                    ]
                    failure = "" if success else ("dropped_required_object" if dropped else f"timeout_before_{incomplete[0]}" if incomplete else "invalid_final_state")
                    row = {
                        "seed": args.seed,
                        "env_id": env_id,
                        "task_id": task_id,
                        "family": task.family,
                        "item": factors["associated_object"],
                        "side": factors["side"],
                        "prompt": previous_tasks.prompts[env_id],
                        "success": success,
                        "failure": failure,
                        "steps": int(episode_steps[env_id]),
                    }
                    for index, stage_name in enumerate(STAGE_NAMES):
                        row[f"stage_{stage_name}"] = bool(stages[index]) if bool(applicable[index]) else "n/a"
                    rows.append(row)
                    counts[task_id] += 1
                done_ids = done.nonzero().flatten()
                episode_steps[done_ids] = 0
                policy.reset(done_ids, task_batch(env))
            tasks = task_batch(env)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "episodes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(rows)
    summary["config"] = vars(args)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "name", "episodes", "success_rate", "mean_steps"])
        for group in ("by_task", "by_family", "by_side"):
            for name, values in summary[group].items():
                writer.writerow([group, name, values["episodes"], values["success_rate"], values["mean_steps"]])
    print(json.dumps({"episodes": summary["episodes"], "success_rate": summary["success_rate"]}, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

