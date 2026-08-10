"""Reset randomization for the persistent desk scene."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from ..task_catalog import OBJECT_NAMES


def reset_desk_scene(
    env,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the robot and place every object into a distinct randomized slot."""

    if env_ids is None or len(env_ids) == 0:
        return
    robot = env.scene[robot_cfg.name]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)

    # Stable workspace anchors. Large serving objects occupy the back row; small
    # objects are permuted over eight front/middle slots.
    fixed_xy = {
        "mug": (0.42, -0.22),
        "bowl": (0.42, 0.22),
        "tray": (0.72, 0.0),
        "red_candy": (0.55, -0.10),
        "blue_candy": (0.55, 0.10),
        "tea": (0.68, -0.18),
        "sugar": (0.68, 0.18),
    }
    visible_names = set(fixed_xy)
    root_z = {"mug": 0.106, "bowl": 0.087, "tray": 0.013, "red_candy": 0.051, "blue_candy": 0.095, "tea": 0.106, "sugar": 0.051}
    slots = torch.tensor(
        [
            [0.36, -0.08],
            [0.46, -0.08],
            [0.56, -0.08],
            [0.66, -0.08],
            [0.36, 0.06],
            [0.46, 0.06],
            [0.56, 0.06],
            [0.66, 0.06],
        ],
        device=env.device,
    )
    small_names = [name for name in OBJECT_NAMES if name not in fixed_xy]
    small_z = {
        "tea": 0.771,
        "sugar": 0.769,
        "red_candy": 0.770,
        "blue_candy": 0.770,
        "cookie": 0.763,
        "spoon": 0.761,
        "coaster": 0.758,
        "napkin": 0.757,
    }
    for local_index, env_id in enumerate(env_ids.tolist()):
        generator = torch.Generator(device=env.device)
        generator.manual_seed(int(env.common_step_counter + env_id * 7919 + local_index * 104729))
        permutation = torch.randperm(len(slots), generator=generator, device=env.device)
        jitter = (torch.rand((len(slots), 2), generator=generator, device=env.device) - 0.5) * 0.018
        yaw = (torch.rand(len(OBJECT_NAMES), generator=generator, device=env.device) - 0.5) * 0.8
        for object_index, name in enumerate(OBJECT_NAMES):
            asset = env.scene[name]
            if name in visible_names:
                xy = torch.tensor(fixed_xy[name], device=env.device)
                z = root_z[name]
            else:
                # Park non-starter objects below the ground while retaining their
                # scene handles for compatibility with the full task catalog.
                xy = torch.tensor([0.0, 0.0], device=env.device)
                z = -2.0
            position = torch.tensor([xy[0], xy[1], z], device=env.device) + env.scene.env_origins[env_id]
            orientation = math_utils.quat_from_euler_xyz(
                torch.zeros(1, device=env.device),
                torch.zeros(1, device=env.device),
                yaw[object_index : object_index + 1],
            )[0]
            pose = torch.cat((position, orientation))[None, :]
            ids = torch.tensor([env_id], device=env.device)
            asset.write_root_pose_to_sim(pose, env_ids=ids)
            asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=ids)

