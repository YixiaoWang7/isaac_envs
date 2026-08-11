"""Build the simple hollow mug used by the household task."""

from pathlib import Path
import math

from isaaclab.app import AppLauncher


app = AppLauncher(headless=True).app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "procedural" / "manipulation_mug.usd"
OUTER_RADIUS = 0.045
WALL_THICKNESS = 0.003
HEIGHT = 0.080
BOTTOM_THICKNESS = 0.006
WALL_SEGMENTS = 24


def add_cube(stage, path, position, scale, rotate_z=0.0):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    # Keep the non-uniform scale in the plank's local frame. Rotating after
    # scaling makes the long axis fan radially instead of forming bucket staves.
    cube.AddTranslateOp().Set(Gf.Vec3d(*position))
    if rotate_z:
        cube.AddRotateZOp().Set(rotate_z)
    cube.AddScaleOp().Set(Gf.Vec3d(*scale))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube.GetPrim()


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
stage = Usd.Stage.CreateNew(str(OUTPUT))
root = UsdGeom.Xform.Define(stage, "/ManipulationMug")
stage.SetDefaultPrim(root.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
mass.CreateMassAttr(0.025)

material = UsdShade.Material.Define(stage, "/ManipulationMug/Looks/MugMaterial")
shader = UsdShade.Shader.Define(stage, "/ManipulationMug/Looks/MugMaterial/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.8, 0.82))
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

# Bucket-like wall: broad upright planks are tangent to the circle and overlap
# slightly, producing a continuous wall with matching visual/collision shape.
wall_center_radius = OUTER_RADIUS - WALL_THICKNESS / 2.0
segment_length = 2.0 * wall_center_radius * math.tan(math.pi / WALL_SEGMENTS) * 1.08
for index in range(WALL_SEGMENTS):
    angle = 2.0 * math.pi * index / WALL_SEGMENTS
    x = wall_center_radius * math.cos(angle)
    y = wall_center_radius * math.sin(angle)
    prim = add_cube(
        stage,
        f"/ManipulationMug/WallPlank_{index:02d}",
        (x, y, HEIGHT / 2.0),
        (WALL_THICKNESS, segment_length, HEIGHT),
        math.degrees(angle),
    )
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

# The thick opaque floor is rendered and collidable, preventing cubes or
# station markers from appearing through the vessel.
bottom = UsdGeom.Cylinder.Define(stage, "/ManipulationMug/Bottom")
bottom.CreateAxisAttr("Z")
bottom.CreateRadiusAttr(OUTER_RADIUS)
bottom.CreateHeightAttr(BOTTOM_THICKNESS)
bottom.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, BOTTOM_THICKNESS / 2.0))
UsdPhysics.CollisionAPI.Apply(bottom.GetPrim())
UsdShade.MaterialBindingAPI.Apply(bottom.GetPrim()).Bind(material)

# Large rectangular handle on +X: 30 mm deep by 38 mm tall clear opening.
# Its 14 mm pinch depth gives both Franka fingers a broad, stable contact patch.
handle_parts = (
    ("HandleLower", (0.062, 0.0, 0.020), (0.034, 0.014, 0.007)),
    ("HandleUpper", (0.062, 0.0, 0.066), (0.034, 0.014, 0.007)),
    ("HandleOuter", (0.079, 0.0, 0.043), (0.007, 0.014, 0.053)),
)
for name, position, scale in handle_parts:
    prim = add_cube(stage, f"/ManipulationMug/{name}", position, scale)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

stage.GetRootLayer().Save()
print(f"Wrote {OUTPUT}")
app.close()
