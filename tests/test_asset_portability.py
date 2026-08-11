"""Regression checks for assets copied or cloned onto another machine."""

from pathlib import Path
import zipfile


YCB_DIR = Path(__file__).resolve().parents[1] / "assets" / "real" / "ycb"
KENNEY_DIR = Path(__file__).resolve().parents[1] / "assets" / "real" / "kenney_food"
HF_MUG_DIR = Path(__file__).resolve().parents[1] / "assets" / "real" / "hf_mugs"
PROCEDURAL_DIR = Path(__file__).resolve().parents[1] / "assets" / "procedural"


def test_huggingface_mugs_are_vendored_usdz_archives():
    expected = {"red_mug.usdz"}
    paths = {path.name: path for path in HF_MUG_DIR.glob("*.usdz")}
    assert paths.keys() == expected
    for path in paths.values():
        assert path.stat().st_size > 100_000
        assert zipfile.is_zipfile(path)
    assert (HF_MUG_DIR / "simple_empty_mug.usd").stat().st_size > 100_000


def test_procedural_manipulation_mug_is_vendored():
    mug = PROCEDURAL_DIR / "manipulation_mug.usd"
    assert mug.is_file() and mug.stat().st_size > 2_000


def test_ycb_converter_inputs_and_textures_are_local():
    for config_path in sorted(YCB_DIR.glob("*/config.yaml")):
        lines = config_path.read_text().splitlines()
        asset_path = next(line.split(":", 1)[1].strip() for line in lines if line.startswith("asset_path:"))
        usd_dir = next(line.split(":", 1)[1].strip() for line in lines if line.startswith("usd_dir:"))
        assert not Path(asset_path).is_absolute()
        assert usd_dir == "."
        assert (config_path.parent / asset_path).is_file()

    for texture_path in YCB_DIR.glob("*/textures/texture_map.png"):
        assert texture_path.is_file() and texture_path.stat().st_size > 0


def test_kenney_converter_inputs_and_usd_are_portable():
    for config_path in sorted(KENNEY_DIR.glob("*/config.yaml")):
        lines = config_path.read_text().splitlines()
        asset_path = next(line.split(":", 1)[1].strip() for line in lines if line.startswith("asset_path:"))
        usd_dir = next(line.split(":", 1)[1].strip() for line in lines if line.startswith("usd_dir:"))
        usd_name = next(line.split(":", 1)[1].strip() for line in lines if line.startswith("usd_file_name:"))
        assert not Path(asset_path).is_absolute()
        assert usd_dir == "."
        assert (config_path.parent / asset_path).is_file()
        usd_path = config_path.parent / usd_name
        assert usd_path.is_file()
