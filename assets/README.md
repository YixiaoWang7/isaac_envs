# Assets

The environment is physics-complete using procedural Isaac Lab primitives. Optional NVIDIA mug and bowl USD visuals can
be localized with `scripts/prepare_assets.py`; downloaded files are intentionally ignored because their redistribution
is governed by the NVIDIA Isaac Sim Additional Software and Materials License.

The hollow mug, bowl, tray, and takeaway box use compound primitive colliders defined by the environment configuration,
so insertion tests do not depend on an accidentally closed convex-hull collider.

