"""Transactional storage and balanced scheduling for teleoperated demonstrations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil

import cv2
import h5py
import numpy as np


SCHEMA_VERSION = 5
CAMERA_NAMES = ("front", "wrist", "side")
ACTION_ORDER = ("dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper_target")


def catalog_fingerprint(task_rows: list[dict]) -> str:
    """Return a stable identity for the task semantics stored in a dataset."""
    payload = json.dumps(task_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class BalancedTaskScheduler:
    """Complete each task's successful-demo quota in catalog order."""

    def __init__(self, task_ids: list[int], counts: dict[int, int], quota: int):
        if quota <= 0:
            raise ValueError("demos_per_task must be positive")
        if not task_ids:
            raise ValueError("At least one task ID is required")
        self.task_ids = tuple(task_ids)
        self.counts = {task_id: int(counts.get(task_id, 0)) for task_id in self.task_ids}
        self.quota = quota

    @property
    def complete(self) -> bool:
        return all(self.counts[task_id] >= self.quota for task_id in self.task_ids)

    @property
    def accepted(self) -> int:
        return sum(min(self.counts[task_id], self.quota) for task_id in self.task_ids)

    @property
    def target(self) -> int:
        return len(self.task_ids) * self.quota

    def next_task(self) -> int | None:
        for task_id in self.task_ids:
            if self.counts[task_id] < self.quota:
                return task_id
        return None

    def mark_success(self, task_id: int) -> None:
        if task_id not in self.counts:
            raise KeyError(task_id)
        self.counts[task_id] += 1


@dataclass(frozen=True)
class EpisodeMetadata:
    task_id: int
    task_code: tuple[int, int, int]
    prompt: str
    prompt_variant_id: int
    layout_index: int = 0


class DemonstrationDataset:
    """Append-only HDF5 dataset with matching per-camera MP4 files."""

    def __init__(
        self,
        dataset_dir: str | Path,
        task_rows: list[dict],
        fps: int = 30,
        resolution: tuple[int, int] = (256, 256),
        collection_seed: int = 0,
        success_hold_seconds: float = 1.0,
        gripper_release_width_m: float = 0.06,
        collection_task_ids: list[int] | tuple[int, ...] | None = None,
    ):
        self.root = Path(dataset_dir).expanduser().resolve()
        self.video_dir = self.root / "videos"
        self.partial_dir = self.root / ".partial"
        self.hdf5_path = self.root / "dataset.hdf5"
        self.fps = int(fps)
        self.resolution = tuple(int(value) for value in resolution)
        self.task_rows = task_rows
        self.collection_seed = int(collection_seed)
        self.success_hold_seconds = float(success_hold_seconds)
        self.gripper_release_width_m = float(gripper_release_width_m)
        self.collection_task_ids = tuple(
            int(task_id) for task_id in (
                collection_task_ids
                if collection_task_ids is not None
                else (row["task_id"] for row in task_rows)
            )
        )
        self.fingerprint = catalog_fingerprint(task_rows)
        self.root.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(exist_ok=True)
        self.partial_dir.mkdir(exist_ok=True)
        self._discard_stale_partials()
        self.stream = h5py.File(self.hdf5_path, "a")
        self.data = self.stream.require_group("data")
        try:
            self._initialize_or_validate()
            self._validate_committed_episodes()
        except Exception:
            self.stream.close()
            raise
        self.next_demo_id = self._find_next_demo_id()

    def _metadata(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_catalog_sha256": self.fingerprint,
            "task_count": len(self.task_rows),
            "collection_task_ids": list(self.collection_task_ids),
            "collection_seed": self.collection_seed,
            "fps": self.fps,
            "resolution": list(self.resolution),
            "camera_names": list(CAMERA_NAMES),
            "video_codec": "mp4v",
            "pose_frame": "robot_base",
            "quaternion_order": "wxyz",
            "rotation_delta": "axis_angle_radians",
            "translation_delta": "robot_base_meters",
            "gripper_target": {"closed": 0, "open": 1},
            "success_hold_seconds": self.success_hold_seconds,
            "gripper_release_width_m": self.gripper_release_width_m,
            "success_criteria": [
                "task_relations",
                "gripper_released",
                "required_objects_stable",
            ],
            "action_order": list(ACTION_ORDER),
        }

    def _initialize_or_validate(self) -> None:
        expected = json.dumps(self._metadata(), sort_keys=True)
        existing = self.data.attrs.get("collection_schema")
        if existing is None:
            self.data.attrs["collection_schema"] = expected
            self.data.attrs["total"] = 0
            self.data.attrs["next_layout_index"] = 0
            self.stream.flush()
        elif existing != expected:
            # A launch may create the HDF5 file before the first recording. It is
            # safe to update that empty file when the collector schema changes;
            # committed demonstrations are never migrated implicitly.
            has_demos = any(name.startswith("demo_") for name in self.data)
            try:
                existing_version = int(json.loads(existing)["schema_version"])
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                existing_version = SCHEMA_VERSION
            if (
                not has_demos
                and int(self.data.attrs.get("total", 0)) == 0
                and existing_version != SCHEMA_VERSION
            ):
                self.data.attrs["collection_schema"] = expected
                self.stream.flush()
            else:
                raise ValueError(
                    f"Dataset at {self.hdf5_path} is incompatible with this collector. "
                    f"Existing schema: {existing}; requested schema: {expected}"
                )

    def _discard_stale_partials(self) -> None:
        for path in self.partial_dir.glob("attempt_*"):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    def _validate_committed_episodes(self) -> None:
        for name, episode in self.data.items():
            if not name.startswith("demo_"):
                raise ValueError(f"Unexpected group in dataset: {name}")
            if not bool(episode.attrs.get("success", False)):
                raise ValueError(f"Committed episode {name} is not marked successful")
            for camera in CAMERA_NAMES:
                relative_path = episode.attrs.get(f"{camera}_video")
                if not relative_path or not (self.root / relative_path).is_file():
                    raise ValueError(f"Episode {name} is missing its {camera} video")

    def _find_next_demo_id(self) -> int:
        ids = [int(name.removeprefix("demo_")) for name in self.data if name.startswith("demo_")]
        return max(ids, default=-1) + 1

    def task_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for episode in self.data.values():
            task_id = int(episode.attrs["task_id"])
            counts[task_id] = counts.get(task_id, 0) + 1
        return counts

    def set_requested_quota(self, demos_per_task: int) -> None:
        """Record the latest collection target without making it a compatibility constraint."""
        self.data.attrs["requested_demos_per_task"] = int(demos_per_task)
        self.stream.flush()

    def allocate_layout_index(self) -> int:
        """Reserve a unique reset index, including for attempts later discarded."""
        index = int(self.data.attrs.get("next_layout_index", 0))
        self.data.attrs["next_layout_index"] = index + 1
        self.stream.flush()
        return index

    def new_attempt(self, metadata: EpisodeMetadata) -> "EpisodeAttempt":
        return EpisodeAttempt(self, metadata)

    def close(self) -> None:
        if self.stream:
            self.stream.flush()
            self.stream.close()
            self.stream = None


class EpisodeAttempt:
    """One temporary trajectory that is either committed whole or discarded."""

    def __init__(self, dataset: DemonstrationDataset, metadata: EpisodeMetadata):
        self.dataset = dataset
        self.metadata = metadata
        self.frames = 0
        self.closed = False
        self.buffers: dict[str, list[np.ndarray | float | bool]] = {
            "ee_pose": [],
            "gripper_width": [],
            "ee_delta_pose": [],
            "gripper_target": [],
            "task_success": [],
            "gripper_released": [],
            "required_objects_stable": [],
            "stable_success": [],
            "time": [],
        }
        width, height = dataset.resolution
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.partial_paths = {
            camera: dataset.partial_dir / f"attempt_{camera}.mp4" for camera in CAMERA_NAMES
        }
        self.writers = {
            camera: cv2.VideoWriter(str(path), fourcc, dataset.fps, (width, height))
            for camera, path in self.partial_paths.items()
        }
        failed = [camera for camera, writer in self.writers.items() if not writer.isOpened()]
        if failed:
            self.discard()
            raise RuntimeError(f"Could not open MP4 writers for: {', '.join(failed)}")

    def append(
        self,
        images: dict[str, np.ndarray],
        ee_pose: np.ndarray,
        gripper_width: float,
        ee_delta_pose: np.ndarray,
        gripper_target: int,
        task_success: bool,
        gripper_released: bool,
        required_objects_stable: bool,
        stable_success: bool,
    ) -> None:
        if self.closed:
            raise RuntimeError("Attempt is already closed")
        width, height = self.dataset.resolution
        for camera in CAMERA_NAMES:
            frame = np.asarray(images[camera])
            if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                raise ValueError(f"{camera} frame must be uint8 {(height, width, 3)}, got {frame.shape} {frame.dtype}")
            self.writers[camera].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if gripper_target not in (0, 1):
            raise ValueError("gripper_target must be the absolute binary state 0 or 1")
        self.buffers["ee_pose"].append(np.asarray(ee_pose, dtype=np.float32).reshape(7))
        self.buffers["gripper_width"].append(float(gripper_width))
        self.buffers["ee_delta_pose"].append(np.asarray(ee_delta_pose, dtype=np.float32).reshape(6))
        self.buffers["gripper_target"].append(gripper_target)
        self.buffers["task_success"].append(task_success)
        self.buffers["gripper_released"].append(gripper_released)
        self.buffers["required_objects_stable"].append(required_objects_stable)
        self.buffers["stable_success"].append(stable_success)
        self.buffers["time"].append(self.frames / self.dataset.fps)
        self.frames += 1

    def _close_videos(self) -> None:
        for writer in self.writers.values():
            writer.release()
        self.closed = True

    def _validate_videos(self) -> None:
        width, height = self.dataset.resolution
        for camera, path in self.partial_paths.items():
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Video encoder produced no data: {path}")
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    raise RuntimeError(f"Could not reopen encoded {camera} video: {path}")
                properties = {
                    "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": round(capture.get(cv2.CAP_PROP_FPS)),
                }
                expected = {"frames": self.frames, "width": width, "height": height, "fps": self.dataset.fps}
                if properties != expected:
                    raise RuntimeError(f"Invalid {camera} video properties: {properties}; expected {expected}")
            finally:
                capture.release()

    def discard(self) -> None:
        if not self.closed:
            self._close_videos()
        for path in self.partial_paths.values():
            path.unlink(missing_ok=True)

    def commit(self) -> str:
        if self.frames == 0:
            raise ValueError("Cannot commit an empty demonstration")
        if not bool(self.buffers["stable_success"][-1]):
            raise ValueError("Cannot commit an episode without terminal stable success")
        if not (
            bool(self.buffers["task_success"][-1])
            and bool(self.buffers["gripper_released"][-1])
            and bool(self.buffers["required_objects_stable"][-1])
        ):
            raise ValueError("Cannot commit before the task is complete, released, and stable")
        self._close_videos()
        try:
            self._validate_videos()
        except Exception:
            self.discard()
            raise

        demo_id = self.dataset.next_demo_id
        name = f"demo_{demo_id:06d}"
        final_paths = {
            camera: self.dataset.video_dir / f"{name}_{camera}.mp4" for camera in CAMERA_NAMES
        }
        moved: list[Path] = []
        previous_total = int(self.dataset.data.attrs.get("total", 0))
        try:
            for camera in CAMERA_NAMES:
                os.replace(self.partial_paths[camera], final_paths[camera])
                moved.append(final_paths[camera])
            group = self.dataset.data.create_group(name)
            group.attrs.update(
                {
                    "success": True,
                    "num_samples": self.frames,
                    "task_id": self.metadata.task_id,
                    "task_code": self.metadata.task_code,
                    "prompt": self.metadata.prompt,
                    "prompt_variant_id": self.metadata.prompt_variant_id,
                    "layout_index": self.metadata.layout_index,
                }
            )
            for camera, path in final_paths.items():
                group.attrs[f"{camera}_video"] = str(path.relative_to(self.dataset.root))
            obs = group.create_group("obs")
            obs.create_dataset("ee_pose", data=np.stack(self.buffers["ee_pose"]), compression="gzip")
            obs.create_dataset(
                "gripper_width", data=np.asarray(self.buffers["gripper_width"], dtype=np.float32)[:, None], compression="gzip"
            )
            actions = group.create_group("actions")
            actions.create_dataset("ee_delta_pose", data=np.stack(self.buffers["ee_delta_pose"]), compression="gzip")
            actions.create_dataset(
                "gripper_target", data=np.asarray(self.buffers["gripper_target"], dtype=np.uint8)[:, None], compression="gzip"
            )
            signals = group.create_group("signals")
            signals.create_dataset(
                "task_success_after_action",
                data=np.asarray(self.buffers["task_success"], dtype=np.bool_)[:, None],
                compression="gzip",
            )
            signals.create_dataset(
                "gripper_released_after_action",
                data=np.asarray(self.buffers["gripper_released"], dtype=np.bool_)[:, None],
                compression="gzip",
            )
            signals.create_dataset(
                "required_objects_stable_after_action",
                data=np.asarray(self.buffers["required_objects_stable"], dtype=np.bool_)[:, None],
                compression="gzip",
            )
            signals.create_dataset(
                "stable_success_after_action",
                data=np.asarray(self.buffers["stable_success"], dtype=np.bool_)[:, None],
                compression="gzip",
            )
            group.create_dataset("time", data=np.asarray(self.buffers["time"], dtype=np.float32), compression="gzip")
            self.dataset.data.attrs["total"] = previous_total + self.frames
            self.dataset.stream.flush()
            self.dataset.next_demo_id += 1
            return name
        except Exception:
            if name in self.dataset.data:
                del self.dataset.data[name]
                self.dataset.data.attrs["total"] = previous_total
                self.dataset.stream.flush()
            for path in moved:
                path.unlink(missing_ok=True)
            raise
