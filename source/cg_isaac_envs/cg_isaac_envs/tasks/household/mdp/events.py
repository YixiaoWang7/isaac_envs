"""Safe reset placement for all realistic household objects."""

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

CONTAINER_NAMES = ("mug_a", "mug_b", "mug_c")
FOOD_NAMES = ("red_candy", "blue_candy", "green_candy")
DESTINATION_NAMES = ("hot_serving_place", "cold_serving_place", "storage_place")

# Shuffle semantic objects between collision-safe slots independently in every env.
CONTAINER_SLOTS = ((0.44, -0.15), (0.44, 0.0), (0.44, 0.15))
FOOD_SLOTS = ((0.30, -0.13), (0.30, 0.0), (0.30, 0.13))
DESTINATION_SLOTS = ((0.61, -0.22), (0.61, 0.0), (0.61, 0.22))
SPAWN_HEIGHT = {
    "mug_a": 0.003,
    "mug_b": 0.003,
    "mug_c": 0.003,
    "red_candy": 0.011,
    "blue_candy": 0.011,
    "green_candy": 0.011,
    "hot_serving_place": 0.0005,
    "cold_serving_place": 0.0005,
    "storage_place": 0.0005,
}


def reset_household_scene(env, env_ids: torch.Tensor, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    if env_ids is None or len(env_ids) == 0:
        return
    robot = env.scene[robot_cfg.name]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)

    if not hasattr(env, "_household_reset_counts"):
        env._household_reset_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    base_seed = int(env.cfg.seed if env.cfg.seed is not None else 0)

    for env_id in env_ids.tolist():
        generator = torch.Generator(device=env.device)
        reset_index = int(env._household_reset_counts[env_id])
        generator.manual_seed(base_seed + env_id * 7919 + reset_index * 104729)
        ids = torch.tensor([env_id], dtype=torch.long, device=env.device)
        groups = (
            (CONTAINER_NAMES, CONTAINER_SLOTS),
            (FOOD_NAMES, FOOD_SLOTS),
            (DESTINATION_NAMES, DESTINATION_SLOTS),
        )
        for names, slots in groups:
            permutation = torch.randperm(len(slots), generator=generator, device=env.device)
            for object_index, name in enumerate(names):
                x, y = slots[int(permutation[object_index])]
                # Randomize locations, but keep object headings repeatable.  Stable
                # mug handles and candy wrappers make teleoperation much clearer.
                jitter_scale = 0.020 if names != DESTINATION_NAMES else 0.0
                jitter = (torch.rand(2, generator=generator, device=env.device) - 0.5) * jitter_scale
                # Handles face 90 degrees further counterclockwise than before.
                yaw = torch.tensor([3.14159265 if name in CONTAINER_NAMES else 0.0], device=env.device)
                # The procedural mugs and candy cubes are authored Z-up.
                roll = torch.zeros(1, device=env.device)

                position = torch.stack((
                    torch.as_tensor(x, device=env.device) + jitter[0],
                    torch.as_tensor(y, device=env.device) + jitter[1],
                    torch.as_tensor(SPAWN_HEIGHT[name], device=env.device),
                ))
                position += env.scene.env_origins[env_id]
                quat = math_utils.quat_from_euler_xyz(
                    roll, torch.zeros(1, device=env.device), yaw
                )[0]
                env.scene[name].write_root_pose_to_sim(torch.cat((position, quat))[None, :], env_ids=ids)
                env.scene[name].write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=ids)
        env._household_reset_counts[env_id] += 1
