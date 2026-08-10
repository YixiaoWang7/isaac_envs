"""Print demonstration metadata and episode shapes."""

import argparse
import json

import h5py

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
args = parser.parse_args()

with h5py.File(args.dataset, "r") as stream:
    data = stream["data"]
    print(json.dumps(json.loads(data.attrs["env_args"]), indent=2))
    print(f"episodes={len(data)}, samples={data.attrs['total']}")
    for name in list(data.keys())[:5]:
        episode = data[name]
        print(name, "success=", episode.attrs.get("success"), "samples=", episode.attrs.get("num_samples"))

