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

ASSET_DIR = Path(__file__).resolve().parents[5] / "assets" / "real" / "ycb"


def dynamic_usd(path: str, scale=(1.0, 1.0, 1.0)) -> sim_utils.UsdFileCfg:
    return sim_utils.UsdFileCfg(
        usd_path=path,
        scale=scale,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
            max_depenetration_velocity=2.0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
    )


def fixed_bin(path: str, scale) -> sim_utils.UsdFileCfg:
    return sim_utils.UsdFileCfg(
        usd_path=path,
        scale=scale,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003),
    )


@configclass
class HouseholdSceneCfg(InteractiveSceneCfg):
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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

    bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bowl", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.38, -0.18, 0.030)),
        spawn=dynamic_usd(str(ASSET_DIR / "024_bowl" / "bowl.usd"), (0.75, 0.75, 0.75)),
    )
    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.38, 0.0, 0.020)),
        spawn=dynamic_usd(str(ASSET_DIR / "029_plate" / "plate.usd"), (0.55, 0.55, 0.55)),
    )
    serving_pan = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ServingPan", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.38, 0.19, 0.035)),
        spawn=dynamic_usd(str(ASSET_DIR / "027_skillet" / "serving_pan.usd"), (0.60, 0.60, 0.60)),
    )
    apple = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Apple", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.53, -0.16, 0.040)),
        spawn=dynamic_usd(str(ASSET_DIR / "013_apple" / "apple.usd"), (0.80, 0.80, 0.80)),
    )
    banana = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Banana", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.53, 0.0, 0.025)),
        spawn=dynamic_usd(str(ASSET_DIR / "011_banana" / "banana.usd"), (0.70, 0.70, 0.70)),
    )
    snack_package = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SnackPackage", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.53, 0.16, 0.030)),
        spawn=dynamic_usd(str(ASSET_DIR / "008_pudding_box" / "candy_package.usd"), (0.55, 0.55, 0.55)),
    )

    left_place = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LeftPlace", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.69, -0.25, 0.006)),
        spawn=sim_utils.CylinderCfg(radius=0.085, height=0.009,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.28, 0.12), roughness=0.9),
            collision_props=None, rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
    )
    right_place = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RightPlace", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.69, 0.25, 0.006)),
        spawn=sim_utils.CylinderCfg(radius=0.085, height=0.009,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.36, 0.18, 0.08), roughness=0.9),
            collision_props=None, rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
    )
    packing_place = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PackingPlace", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.72, 0.0, 0.018)),
        spawn=sim_utils.CuboidCfg(
            size=(0.24, 0.18, 0.014),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.36, 0.16, 0.055), roughness=0.72
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
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
        self.decimation = 2
        self.episode_length_s = 45.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (1.85, 0.0, 1.50)
        self.viewer.lookat = (0.45, 0.0, 0.18)
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
                focal_length=20.0, focus_distance=1.2, horizontal_aperture=24.0, clipping_range=(0.05, 3.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.45, 0.0, 0.82),
                rot=(0.35355, -0.61237, -0.61237, 0.35355),
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
