"""SpaceMouse reader using the Linux input-event interface."""

from __future__ import annotations

import select
import struct
import threading
from collections.abc import Callable
from pathlib import Path

import torch


_EVENT = struct.Struct("llHHi")
_EV_SYN, _EV_KEY, _EV_REL, _EV_ABS = 0x00, 0x01, 0x02, 0x03
_SYN_REPORT = 0
_AXES = {0x00: "x", 0x01: "y", 0x02: "z", 0x03: "rx", 0x04: "ry", 0x05: "rz"}


class LinuxSpaceMouse:
    """Read six-axis commands and buttons from a Linux ``/dev/input/event*`` node."""

    def __init__(
        self,
        device_path: str,
        sim_device: str,
        pos_sensitivity: float = 0.05,
        rot_sensitivity: float = 0.05,
        scale: float = 350.0,
        deadzone: float = 0.05,
    ):
        self.device_path = Path(device_path)
        self.sim_device = sim_device
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.scale = scale
        self.deadzone = deadzone
        self._axes = dict.fromkeys(_AXES.values(), 0.0)
        self._close_gripper = False
        self._callbacks: dict[str, Callable[[], None]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # Open synchronously so missing nodes and permissions fail with a useful traceback.
        self._device = self.device_path.open("rb", buffering=0)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def __str__(self) -> str:
        return (
            "Linux SpaceMouse Controller for SE(3)\n"
            f"\tDevice: {self.device_path}\n"
            "\tFirst button: toggle gripper\n"
            "\tSecond button: reset environment"
        )

    def reset(self):
        with self._lock:
            self._axes = dict.fromkeys(_AXES.values(), 0.0)
            self._close_gripper = False

    def add_callback(self, key: str, callback: Callable[[], None]):
        self._callbacks[key] = callback

    def advance(self) -> torch.Tensor:
        with self._lock:
            axes = dict(self._axes)
            gripper = -1.0 if self._close_gripper else 1.0
        command = [
            -axes["y"] * self.pos_sensitivity,
            axes["x"] * self.pos_sensitivity,
            -axes["z"] * self.pos_sensitivity,
            axes["ry"] * self.rot_sensitivity,
            axes["rx"] * self.rot_sensitivity,
            -axes["rz"] * self.rot_sensitivity,
            gripper,
        ]
        return torch.tensor(command, dtype=torch.float32, device=self.sim_device)

    def close(self):
        self._stop.set()
        self._device.close()
        self._thread.join(timeout=1.0)

    def _normalized(self, value: int) -> float:
        normalized = max(-self.scale, min(self.scale, float(value))) / self.scale
        return 0.0 if abs(normalized) < self.deadzone else normalized

    def _read_loop(self):
        pending = dict.fromkeys(_AXES.values(), 0)
        changed = False
        while not self._stop.is_set():
            try:
                readable, _, _ = select.select([self._device], [], [], 0.1)
                if not readable:
                    continue
                data = self._device.read(_EVENT.size)
            except (OSError, ValueError):
                return
            if len(data) != _EVENT.size:
                return
            _, _, event_type, code, value = _EVENT.unpack(data)
            axis = _AXES.get(code) if event_type in (_EV_REL, _EV_ABS) else None
            if axis is not None:
                pending[axis] = value
                changed = True
            elif event_type == _EV_KEY and value:
                if code == 256:
                    with self._lock:
                        self._close_gripper = not self._close_gripper
                    callback = self._callbacks.get("L")
                    if callback:
                        callback()
                elif code == 257:
                    callback = self._callbacks.get("R")
                    if callback:
                        callback()
            elif event_type == _EV_SYN and code == _SYN_REPORT and changed:
                with self._lock:
                    self._axes = {axis_name: self._normalized(axis_value) for axis_name, axis_value in pending.items()}
                pending = dict.fromkeys(pending, 0)
                changed = False
