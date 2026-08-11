# Real-object asset licenses

The meshes in `ycb/` are derived from the Yale-CMU-Berkeley Object and Model Set.

- Objects include `005_tomato_soup_can`, `008_pudding_box`, `009_gelatin_box`,
  `017_orange`, `025_mug`, `029_plate`, and `031_spoon`.
- Upstream: https://www.ycbbenchmarks.com/object-models/
- Download mirror: https://huggingface.co/datasets/ai-habitat/ycb
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Citation: B. Calli et al., "The YCB Object and Model Set: Towards Common
  Benchmarks for Manipulation Research," 2015.

The converted USD files were generated locally with Isaac Lab's `convert_mesh.py`.
The downloaded GLB files are retained beside each USD for provenance and texture
references.

The active asset in `hf_mugs/` comes from the
[`barszot/3d-models-for-isaac-sim-dataset`](https://huggingface.co/datasets/barszot/3d-models-for-isaac-sim-dataset)
dataset (CC BY 4.0 dataset license), with the following upstream attribution:

- `red_mug.usdz`: hungry_beagle, CC Attribution,
  https://sketchfab.com/3d-models/classic-red-coffee-mug-c89fa8b5390145488a28c41ec54ccc8d

`simple_empty_mug.usd` is extracted from `red_mug.usdz` and modified only to
replace the cavity-sealing convex-hull collider with an SDF mesh collider.
