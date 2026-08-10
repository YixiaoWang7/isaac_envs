"""Safe reset placement for all realistic household objects."""

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

CONTAINER_NAMES = ("bowl", "plate", "serving_pan")
FOOD_NAMES = ("apple", "banana", "snack_package")

# Shuffle semantic objects between collision-safe slots independently in every env.
CONTAINER_SLOTS = ((0.36, -0.20), (0.36, 0.0), (0.36, 0.20))
FOOD_SLOTS = ((0.54, -0.18), (0.54, 0.0), (0.54, 0.18))
SPAWN_HEIGHT = {
    "bowl": 0.030,
    "plate": 0.020,
    "serving_pan": 0.045,
    "apple": 0.040,
    "banana": 0.025,
    "snack_package": 0.030,
}


def reset_household_scene(env, env_ids: torch.Tensor, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    if env_ids is None or len(env_ids) == 0:
        return
    robot = env.scene[robot_cfg.name]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)

    for local_index, env_id in enumerate(env_ids.tolist()):
        generator = torch.Generator(device=env.device)
        generator.manual_seed(int(env.common_step_counter + env_id * 7919 + local_index * 104729))
        ids = torch.tensor([env_id], dtype=torch.long, device=env.device)
        groups = ((CONTAINER_NAMES, CONTAINER_SLOTS), (FOOD_NAMES, FOOD_SLOTS))
        for names, slots in groups:
            permutation = torch.randperm(len(slots), generator=generator, device=env.device)
            for object_index, name in enumerate(names):
                x, y = slots[int(permutation[object_index])]
                jitter = (torch.rand(2, generator=generator, device=env.device) - 0.5) * 0.036

                # The pan handle stays broadly sideways; all other objects get full yaw variation.
                if name == "serving_pan":
                    yaw = 1.5708 + (torch.rand(1, generator=generator, device=env.device) - 0.5) * 0.30
                else:
                    yaw = (torch.rand(1, generator=generator, device=env.device) - 0.5) * 6.2832

                position = torch.stack((
                    torch.as_tensor(x, device=env.device) + jitter[0],
                    torch.as_tensor(y, device=env.device) + jitter[1],
                    torch.as_tensor(SPAWN_HEIGHT[name], device=env.device),
                ))
                position += env.scene.env_origins[env_id]
                quat = math_utils.quat_from_euler_xyz(
                    torch.zeros(1, device=env.device), torch.zeros(1, device=env.device), yaw
                )[0]
                env.scene[name].write_root_pose_to_sim(torch.cat((position, quat))[None, :], env_ids=ids)
                env.scene[name].write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=ids)
