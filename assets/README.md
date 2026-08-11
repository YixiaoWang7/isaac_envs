# Assets

The environment is physics-complete using procedural Isaac Lab primitives. Optional NVIDIA mug and bowl USD visuals can
be localized with `scripts/prepare_assets.py`; downloaded files are intentionally ignored because their redistribution
is governed by the NVIDIA Isaac Sim Additional Software and Materials License.

The hollow mug, bowl, tray, and takeaway box use compound primitive colliders defined by the environment configuration,
so insertion tests do not depend on an accidentally closed convex-hull collider.

The active candy-mug task uses `procedural/manipulation_mug.usd`, a lightweight
compound rigid body designed for reliable manipulation. It has 3 mm open walls,
an 84 mm clear inner diameter, a 6 mm collidable bottom, and an oversized handle.
Its wall is assembled from 24 slightly overlapping vertical bucket-style planks
whose rendered and collision shapes match. The asset is generated reproducibly
by `scripts/build_manipulation_mug.py` and presented in three colors.
