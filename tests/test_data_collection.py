import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from cg_isaac_envs.data_collection import (
    BalancedTaskScheduler,
    DemonstrationDataset,
    EpisodeMetadata,
)


TASK_ROWS = [
    {"task_id": task_id, "code": [1, 1, task_id + 1], "prompts": [f"task {task_id}"]}
    for task_id in range(3)
]


def test_balanced_scheduler_skips_complete_tasks():
    scheduler = BalancedTaskScheduler([0, 1, 2], {0: 2, 1: 1}, quota=2)
    assert scheduler.next_task() == 1
    scheduler.mark_success(1)
    assert scheduler.next_task() == 2
    scheduler.mark_success(2)
    assert scheduler.next_task() == 2
    scheduler.mark_success(2)
    assert scheduler.complete
    assert scheduler.accepted == scheduler.target == 6


def test_scheduler_finishes_one_task_before_advancing():
    scheduler = BalancedTaskScheduler([0, 1, 2], {}, quota=2)
    assert scheduler.next_task() == 0
    scheduler.mark_success(0)
    assert scheduler.next_task() == 0
    scheduler.mark_success(0)
    assert scheduler.next_task() == 1


def sample(
    attempt,
    success_held=False,
    gripper=1,
    cube_inside_mug=None,
    mug_on_station=None,
    end_effector_high=None,
):
    images = {
        camera: np.full((256, 256, 3), channel * 40, dtype=np.uint8)
        for channel, camera in enumerate(("front", "wrist", "side"), start=1)
    }
    attempt.append(
        images=images,
        ee_pose=np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]),
        gripper_width=0.08 if gripper else 0.02,
        ee_delta_pose=np.array([0.01, 0.0, -0.005, 0.0, 0.0, 0.035]),
        gripper_target=gripper,
        cube_inside_mug=success_held if cube_inside_mug is None else cube_inside_mug,
        mug_on_station=success_held if mug_on_station is None else mug_on_station,
        end_effector_high=success_held if end_effector_high is None else end_effector_high,
        success_held=success_held,
    )


def test_success_commit_writes_synchronized_hdf5_and_mp4(tmp_path: Path):
    dataset = DemonstrationDataset(tmp_path, TASK_ROWS)
    attempt = dataset.new_attempt(EpisodeMetadata(1, (1, 1, 2), "task one", 0))
    sample(attempt, success_held=False, gripper=1)
    sample(attempt, success_held=True, gripper=1)
    assert attempt.commit() == "demo_000000"
    dataset.close()

    with h5py.File(tmp_path / "dataset.hdf5", "r") as stream:
        episode = stream["data/demo_000000"]
        assert episode["obs/ee_pose"].shape == (2, 7)
        assert episode["obs/gripper_width"].shape == (2, 1)
        assert episode["actions/ee_delta_pose"].shape == (2, 6)
        assert episode["actions/gripper_target"][:, 0].tolist() == [1, 1]
        assert episode["signals/cube_inside_mug_after_action"][:, 0].tolist() == [False, True]
        assert episode["signals/mug_on_station_after_action"][:, 0].tolist() == [False, True]
        assert episode["signals/end_effector_high_after_action"][:, 0].tolist() == [False, True]
        assert episode["signals/success_held_after_action"][:, 0].tolist() == [False, True]
        assert episode.attrs["task_id"] == 1
        assert stream["data"].attrs["total"] == 2

    for camera in ("front", "wrist", "side"):
        path = tmp_path / "videos" / f"demo_000000_{camera}.mp4"
        capture = cv2.VideoCapture(str(path))
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 256
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 256
        assert round(capture.get(cv2.CAP_PROP_FPS)) == 30
        capture.release()

    resumed = DemonstrationDataset(tmp_path, TASK_ROWS)
    assert resumed.task_counts() == {1: 1}
    assert resumed.next_demo_id == 1
    resumed.close()


def test_discard_leaves_no_committed_artifacts(tmp_path: Path):
    dataset = DemonstrationDataset(tmp_path, TASK_ROWS)
    attempt = dataset.new_attempt(EpisodeMetadata(0, (1, 1, 1), "task zero", 0))
    sample(attempt)
    attempt.discard()
    assert list(dataset.data) == []
    assert list((tmp_path / "videos").iterdir()) == []
    assert list((tmp_path / ".partial").iterdir()) == []
    dataset.close()


def test_commit_requires_cube_inside_mug(tmp_path: Path):
    dataset = DemonstrationDataset(tmp_path, TASK_ROWS)
    attempt = dataset.new_attempt(EpisodeMetadata(0, (1, 1, 1), "task zero", 0))
    sample(attempt, success_held=True, cube_inside_mug=False)
    with pytest.raises(ValueError, match="three success criteria"):
        attempt.commit()
    attempt.discard()
    dataset.close()


def test_commit_requires_mug_on_station(tmp_path: Path):
    dataset = DemonstrationDataset(tmp_path, TASK_ROWS)
    attempt = dataset.new_attempt(EpisodeMetadata(0, (1, 1, 1), "task zero", 0))
    sample(attempt, success_held=True, mug_on_station=False)
    with pytest.raises(ValueError, match="three success criteria"):
        attempt.commit()
    attempt.discard()
    dataset.close()


def test_commit_requires_end_effector_high(tmp_path: Path):
    dataset = DemonstrationDataset(tmp_path, TASK_ROWS)
    attempt = dataset.new_attempt(EpisodeMetadata(0, (1, 1, 1), "task zero", 0))
    sample(attempt, success_held=True, end_effector_high=False)
    with pytest.raises(ValueError, match="three success criteria"):
        attempt.commit()
    attempt.discard()
    dataset.close()


def test_commit_requires_success_hold_duration(tmp_path: Path):
    dataset = DemonstrationDataset(tmp_path, TASK_ROWS)
    attempt = dataset.new_attempt(EpisodeMetadata(0, (1, 1, 1), "task zero", 0))
    sample(
        attempt,
        success_held=False,
        cube_inside_mug=True,
        mug_on_station=True,
        end_effector_high=True,
    )
    with pytest.raises(ValueError, match="required duration"):
        attempt.commit()
    attempt.discard()
    dataset.close()


def test_resume_rejects_incompatible_camera_rate(tmp_path: Path):
    dataset = DemonstrationDataset(tmp_path, TASK_ROWS, fps=30)
    dataset.close()
    with pytest.raises(ValueError, match="incompatible"):
        DemonstrationDataset(tmp_path, TASK_ROWS, fps=15)


def test_resume_rejects_a_different_collection_seed(tmp_path: Path):
    dataset = DemonstrationDataset(tmp_path, TASK_ROWS, collection_seed=1001)
    dataset.close()
    with pytest.raises(ValueError, match="incompatible"):
        DemonstrationDataset(tmp_path, TASK_ROWS, collection_seed=2002)


def test_resume_rejects_a_different_success_hold_time(tmp_path: Path):
    dataset = DemonstrationDataset(tmp_path, TASK_ROWS, success_hold_seconds=1.0)
    dataset.close()
    with pytest.raises(ValueError, match="incompatible"):
        DemonstrationDataset(tmp_path, TASK_ROWS, success_hold_seconds=0.5)


def test_empty_dataset_accepts_a_different_end_effector_height(tmp_path: Path):
    dataset = DemonstrationDataset(tmp_path, TASK_ROWS, ee_height_above_mug_m=0.15)
    dataset.close()
    migrated = DemonstrationDataset(tmp_path, TASK_ROWS, ee_height_above_mug_m=0.20)
    metadata = json.loads(migrated.data.attrs["collection_schema"])
    assert metadata["ee_height_above_mug_m"] == 0.20
    migrated.close()


def test_committed_dataset_rejects_a_different_end_effector_height(tmp_path: Path):
    dataset = DemonstrationDataset(tmp_path, TASK_ROWS, ee_height_above_mug_m=0.15)
    attempt = dataset.new_attempt(EpisodeMetadata(0, (1, 1, 1), "task zero", 0))
    sample(attempt, success_held=True)
    attempt.commit()
    dataset.close()
    with pytest.raises(ValueError, match="incompatible"):
        DemonstrationDataset(tmp_path, TASK_ROWS, ee_height_above_mug_m=0.20)


def test_empty_dataset_migrates_success_definition(tmp_path: Path):
    dataset = DemonstrationDataset(tmp_path, TASK_ROWS)
    dataset.close()
    with h5py.File(tmp_path / "dataset.hdf5", "a") as stream:
        metadata = json.loads(stream["data"].attrs["collection_schema"])
        metadata["success_criteria"] = [
            "selected_cube_inside_target_mug",
            "target_mug_on_target_station",
            "end_effector_at_or_above_success_height",
        ]
        del metadata["ee_height_above_mug_m"]
        metadata["success_ee_height_m"] = 0.20
        stream["data"].attrs["collection_schema"] = json.dumps(metadata, sort_keys=True)

    migrated = DemonstrationDataset(tmp_path, TASK_ROWS)
    migrated_metadata = json.loads(migrated.data.attrs["collection_schema"])
    assert migrated_metadata["success_criteria"] == [
        "selected_cube_inside_target_mug",
        "target_mug_on_station",
        "end_effector_high_enough",
    ]
    assert migrated_metadata["ee_height_above_mug_m"] == 0.15
    assert "success_ee_height_m" not in migrated_metadata
    migrated.close()


def test_resume_rejects_a_different_task_subset(tmp_path: Path):
    dataset = DemonstrationDataset(tmp_path, TASK_ROWS, collection_task_ids=[0, 1])
    dataset.close()
    with pytest.raises(ValueError, match="incompatible"):
        DemonstrationDataset(tmp_path, TASK_ROWS, collection_task_ids=[0, 2])


def test_gripper_target_is_strict_binary(tmp_path: Path):
    dataset = DemonstrationDataset(tmp_path, TASK_ROWS)
    attempt = dataset.new_attempt(EpisodeMetadata(0, (1, 1, 1), "task zero", 0))
    images = {camera: np.zeros((256, 256, 3), dtype=np.uint8) for camera in ("front", "wrist", "side")}
    with pytest.raises(ValueError, match="0 or 1"):
        attempt.append(images, np.zeros(7), 0.08, np.zeros(6), -1, False, False, False, False)
    attempt.discard()
    dataset.close()


def test_layout_index_persists_across_resume(tmp_path: Path):
    dataset = DemonstrationDataset(tmp_path, TASK_ROWS, collection_seed=1001)
    assert dataset.allocate_layout_index() == 0
    assert dataset.allocate_layout_index() == 1
    dataset.close()
    resumed = DemonstrationDataset(tmp_path, TASK_ROWS, collection_seed=1001)
    assert resumed.allocate_layout_index() == 2
    resumed.close()
