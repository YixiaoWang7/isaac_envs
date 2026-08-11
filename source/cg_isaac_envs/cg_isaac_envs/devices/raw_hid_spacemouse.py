"""SpaceMouse reader backed by pyspacemouse/raw HID."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pyspacemouse
import torch


class RawHidSpaceMouse:
    """Read wireless and Bluetooth SpaceMouse devices through raw HID."""

    def __init__(self, sim_device: str, pos_sensitivity: float = 0.05, rot_sensitivity: float = 0.05):
        self.sim_device = sim_device
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self._axes = dict.fromkeys(("x", "y", "z", "roll", "pitch", "yaw"), 0.0)
        self._close_gripper = False
        self._callbacks: dict[str, Callable[[], None]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._device = pyspacemouse.open()
        if self._device is None:
            raise OSError("pyspacemouse could not open a supported raw-HID interface")
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def __str__(self) -> str:
        return (
            "Raw-HID SpaceMouse Controller for SE(3)\n"
            "\tBackend: pyspacemouse\n"
            "\tFirst button: toggle gripper\n"
            "\tSecond button: reset environment"
        )

    def reset(self):
        with self._lock:
            self._axes = dict.fromkeys(self._axes, 0.0)
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
            axes["z"] * self.pos_sensitivity,
            axes["pitch"] * self.rot_sensitivity,
            axes["roll"] * self.rot_sensitivity,
            -axes["yaw"] * self.rot_sensitivity,
            gripper,
        ]
        return torch.tensor(command, dtype=torch.float32, device=self.sim_device)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._device.close()

    def _read_loop(self):
        previous_buttons: set[int] = set()
        while not self._stop.is_set():
            state = self._device.read()
            if state is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._axes.update(
                    x=float(state.x),
                    y=float(state.y),
                    z=float(state.z),
                    roll=float(state.roll),
                    pitch=float(state.pitch),
                    yaw=float(state.yaw),
                )
            button_values = getattr(state, "buttons", getattr(state, "button", []))
            pressed = {index for index, value in enumerate(button_values) if value}
            if 0 in pressed - previous_buttons:
                with self._lock:
                    self._close_gripper = not self._close_gripper
                callback = self._callbacks.get("L")
                if callback:
                    callback()
            if 1 in pressed - previous_buttons:
                callback = self._callbacks.get("R")
                if callback:
                    callback()
            previous_buttons = pressed
            time.sleep(0.01)
