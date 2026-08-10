"""Optionally localize NVIDIA visual USD layers; procedural physics assets need no download."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="assets/downloaded")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import omni.client  # noqa: E402
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR  # noqa: E402


def main():
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "mug.usd": f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd",
        "sorting_bowl_yellow.usd": (
            f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bowl_yellow.usd"
        ),
    }
    results = {}
    for filename, source in sources.items():
        destination = (output / filename).as_uri()
        result = omni.client.copy(source, destination, omni.client.CopyBehavior.OVERWRITE)
        results[filename] = {"source": source, "result": str(result)}
        print(filename, result)
    (output / "manifest.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

