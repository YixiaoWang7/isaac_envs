"""Runtime setup that must happen before Isaac Sim is launched."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configure_private_tmpdir(runtime_root: str | Path | None = None) -> Path:
    """Select a writable per-user temp directory unless the caller supplied one."""
    configured = os.environ.get("TMPDIR")
    if configured:
        tempfile.tempdir = configured
        return Path(configured)

    root = Path(runtime_root) if runtime_root is not None else Path("/tmp")
    private_tmpdir = root / f"cg_isaaclab_{os.getuid()}"
    private_tmpdir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(private_tmpdir)
    tempfile.tempdir = str(private_tmpdir)
    return private_tmpdir
