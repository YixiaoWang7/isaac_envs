import os
import runpy
import tempfile
from pathlib import Path


RUNTIME_SETUP = Path(__file__).resolve().parents[1] / "scripts" / "runtime_setup.py"


def runtime_setup_function():
    return runpy.run_path(RUNTIME_SETUP)["configure_private_tmpdir"]


def test_configure_private_tmpdir_creates_per_user_directory(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", None)

    configured = runtime_setup_function()(tmp_path)

    assert configured == tmp_path / f"cg_isaaclab_{os.getuid()}"
    assert configured.is_dir()
    assert os.environ["TMPDIR"] == str(configured)
    assert tempfile.gettempdir() == str(configured)


def test_configure_private_tmpdir_preserves_explicit_value(tmp_path: Path, monkeypatch):
    explicit = tmp_path / "caller-selected"
    monkeypatch.setenv("TMPDIR", str(explicit))
    monkeypatch.setattr(tempfile, "tempdir", None)

    configured = runtime_setup_function()()

    assert configured == explicit
    assert not explicit.exists()
    assert tempfile.gettempdir() == str(explicit)
