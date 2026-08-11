"""Franka household environment built on the official Seattle-table setup."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from . import mdp
from .command import HouseholdTaskCommandCfg

PROCEDURAL_DIR = Path(__file__).resolve().parents[5] / "assets" / "procedural"


def graspable_material() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="max",
        restitution_combine_mode="min",
        static_friction=1.80,
        dynamic_friction=1.50,
        restitution=0.0,
    )


def dynamic_usd(path: str, scale=(1.0, 1.0, 1.0), color=None, mass=0.025) -> sim_utils.UsdFileCfg:
    return sim_utils.UsdFileCfg(
        usd_path=path,
        scale=scale,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=24,
            solver_velocity_iteration_count=4,
            max_depenetration_velocity=0.5,
            linear_damping=0.08,
            angular_damping=0.12,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        visual_material=(sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.65) if color else None),
    )


def candy_cube(color) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=(0.020, 0.020, 0.020),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=8,
            max_depenetration_velocity=0.08,
            linear_damping=0.15,
            angular_damping=0.20,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
        physics_material=graspable_material(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.65),
    )


@configclass
class HouseholdSceneCfg(InteractiveSceneCfg):
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # The stock high-PD hand is very stiff (200 N effort limit), which can
    # eject small objects.  A compliant, force-limited hand closes around both
    # the 20 mm cubes and thin mug handles without generating a large impulse.
    robot.actuators["panda_hand"].effort_limit_sim = 15.0
    robot.actuators["panda_hand"].stiffness = 350.0
    robot.actuators["panda_hand"].damping = 45.0
    # Same reachable home position as the stock Franka pose, with the tool
    # optical axis straight down and zero world yaw.
    robot.init_state.joint_pos = {
        "panda_joint1": -0.0089,
        "panda_joint2": -0.5296,
        "panda_joint3": 0.0210,
        "panda_joint4": -2.9649,
        "panda_joint5": 0.0163,
        "panda_joint6": 2.4354,
        "panda_joint7": 0.7822,
        "panda_finger_joint.*": 0.04,
    }

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.78, 0.80, 0.84), intensity=1200.0),
    )

    red_candy = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RedCandy", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, -0.13, 0.011)),
        spawn=candy_cube((0.85, 0.04, 0.04)),
    )
    blue_candy = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlueCandy", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.0, 0.011)),
        spawn=candy_cube((0.04, 0.18, 0.90)),
    )
    green_candy = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GreenCandy", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.13, 0.011)),
        spawn=candy_cube((0.04, 0.72, 0.18)),
    )
    mug_a = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MugA", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.44, -0.15, 0.003)),
        spawn=dynamic_usd(str(PROCEDURAL_DIR / "manipulation_mug.usd"), color=(0.92, 0.92, 0.94), mass=0.025),
    )
    mug_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MugB", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.44, 0.0, 0.003)),
        spawn=dynamic_usd(str(PROCEDURAL_DIR / "manipulation_mug.usd"), color=(0.10, 0.55, 0.90), mass=0.025),
    )
    mug_c = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MugC", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.44, 0.15, 0.003)),
        spawn=dynamic_usd(str(PROCEDURAL_DIR / "manipulation_mug.usd"), color=(0.90, 0.12, 0.08), mass=0.025),
    )

    hot_serving_place = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HotServingPlace", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.60, -0.22, 0.0005)),
        # All destination footprints are approximately 0.018 m^2.
        # These are flush visual markers; the table supports placed objects.
        spawn=sim_utils.CylinderCfg(radius=0.076, height=0.001,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.12, 0.04), emissive_color=(0.25, 0.02, 0.0), roughness=0.9),
            collision_props=None, rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
    )
    cold_serving_place = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ColdServingPlace", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.60, 0.22, 0.0005)),
        spawn=sim_utils.CuboidCfg(size=(0.134, 0.134, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.03, 0.30, 0.92), roughness=0.9),
            collision_props=None, rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
    )
    storage_place = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/StoragePlace", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.64, 0.0, 0.0005)),
        spawn=sim_utils.CuboidCfg(
            size=(0.18, 0.10, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.70, 0.20), roughness=0.72
            ),
            collision_props=None,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    marker_cfg = None
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            )
        ],
    )


@configclass
class ActionsCfg:
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class CommandsCfg:
    task = HouseholdTaskCommandCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_pose = ObsTerm(func=mdp.ee_pose)
        object_poses = ObsTerm(func=mdp.object_poses)
        object_velocities = ObsTerm(func=mdp.object_velocities)
        task_encoding = ObsTerm(func=mdp.task_encoding)
        prompt_tokens = ObsTerm(func=mdp.prompt_tokens)
        relation_progress = ObsTerm(func=mdp.relation_progress)
        stages = ObsTerm(func=mdp.stages)
        grasp_relative_poses = ObsTerm(func=mdp.object_relative_gripper_poses)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class VisionObservationsCfg(ObservationsCfg):
    @configclass
    class VisionPolicyCfg(ObservationsCfg.PolicyCfg):
        front_rgb = ObsTerm(func=mdp.front_rgb)
        wrist_rgb = ObsTerm(func=mdp.wrist_rgb)

    policy: VisionPolicyCfg = VisionPolicyCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_household_scene, mode="reset")


@configclass
class RewardsCfg:
    relation_progress = RewTerm(func=mdp.relation_progress_reward, weight=2.0)
    success = RewTerm(func=mdp.success_bonus_reward, weight=20.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)


@configclass
class TerminationsCfg:
    success = DoneTerm(func=mdp.StableTaskSuccess, params={"consecutive_steps": 10, "max_speed": 0.08})
    dropped = DoneTerm(func=mdp.required_object_dropped, params={"minimum_height": -0.08})
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class HouseholdEnvCfg(ManagerBasedRLEnvCfg):
    scene: HouseholdSceneCfg = HouseholdSceneCfg(num_envs=1024, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 45.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        # Apply the high-friction, non-bouncy material globally so it also
        # covers imported USD mugs and the Franka fingertips.
        self.sim.physics_material = graspable_material()
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (1.15, 0.0, 0.95)
        self.viewer.lookat = (0.48, 0.0, 0.03)
        self.viewer.resolution = (1280, 720)


@configclass
class HouseholdVisionEnvCfg(HouseholdEnvCfg):
    observations: VisionObservationsCfg = VisionObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.front_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/FrontCamera",
            update_period=0.0,
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=1.15, horizontal_aperture=24.0, clipping_range=(0.05, 3.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.15, 0.0, 0.95),
                rot=(0.218886, -0.672376, -0.672376, 0.218886),
                convention="ros",
            ),
        )
        self.scene.wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=0.45, horizontal_aperture=24.0, clipping_range=(0.025, 2.0)
            ),
            # Mount the camera on the opposite side of the gripper body.  Its optical axis is
            # nearly parallel to the fingers, with only a small inward tilt so
            # the gripper remains visible without looking steeply at its tip.
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.0),
                rot=(0.703666, -0.069677, -0.069677, 0.703666),
                convention="ros",
            ),
        )
        self.scene.side_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/SideCamera",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=1.10, horizontal_aperture=24.0, clipping_range=(0.05, 3.0)
            ),
            # Robot-right view (negative world Y), aimed across and slightly
            # downward at the tabletop manipulation area.
            offset=CameraCfg.OffsetCfg(
                pos=(0.50, -0.95, 0.62),
                rot=(0.489, -0.872, 0.0, 0.0),
                convention="ros",
            ),
        )
        self.num_rerenders_on_reset = 2


@configclass
class HouseholdEnvCfg_PLAY(HouseholdEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class HouseholdVisionEnvCfg_PLAY(HouseholdVisionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
