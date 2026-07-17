"""Generate a stylized Blender scene from openMagMapper vector-field CSV exports.

Usage:
- Open Blender
- Scripting tab
- Load this file in the text editor
- Update CSV paths near the bottom if needed
- Run Script

The script intentionally prioritizes visuals over exact scientific fidelity:
- dark studio environment
- emissive arrow glyphs
- color by vector magnitude
- optional depth-of-field and slow camera orbit
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import bpy
    from mathutils import Euler, Matrix, Vector
except ImportError as exc:  # pragma: no cover - Blender provides bpy.
    raise SystemExit("This script must be run inside Blender.") from exc


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ROOT = Path(r"C:\Users\scedg10\OneDrive - Cardiff University\projects\openMagMapper")
DATA_DIR = ROOT / "dataKeep"
OUTPUT_CSV = DATA_DIR / "Exp_cam_outputData_USBwebcam_JLC1080_20260716_143634.csv"
FROZEN_CSV = DATA_DIR / "Exp_cam_frozenVectors_USBwebcam_JLC1080_20260716_143634.csv"

# Stylized scene controls.
MAX_ROWS = 250
ARROW_RADIUS = 0.0035
ARROW_HEAD_RADIUS = 0.010
ARROW_HEAD_LENGTH = 0.030
ARROW_BASE_LENGTH = 0.040
VECTOR_SCALE = 0.28
ORIGIN_DOT_RADIUS = 0.006
FIELD_POINT_SIZE = 0.004
GROUND_SIZE = 4.0
CAMERA_DISTANCE = 2.2
CAMERA_ELEVATION = 0.85
CAMERA_ORBIT_DEG = 18.0
USE_ANIMATION = False
FRAME_STEP = 2

# Color palette designed for strong visual contrast.
LOW_COLOR = (0.12, 0.42, 1.0, 1.0)   # blue
MID_COLOR = (0.10, 0.95, 0.95, 1.0)   # cyan
HIGH_COLOR = (1.0, 0.92, 0.18, 1.0)   # yellow


@dataclass
class VectorRow:
    frame_idx: int
    origin: Vector
    vec: Vector
    magnitude: float
    valid: bool


# -----------------------------------------------------------------------------
# Scene helpers
# -----------------------------------------------------------------------------


def clean_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in list(bpy.data.curves):
        if block.users == 0:
            bpy.data.curves.remove(block)


def set_render_preset() -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.film_transparent = False
    scene.view_settings.look = "Very High Contrast"

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bpy.context.scene.world = world
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for node in list(nodes):
        nodes.remove(node)

    bg = nodes.new(type="ShaderNodeBackground")
    bg.inputs[0].default_value = (0.015, 0.018, 0.025, 1.0)
    bg.inputs[1].default_value = 0.7
    out = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg.outputs[0], out.inputs[0])

    bpy.context.scene.eevee.use_gtao = True
    bpy.context.scene.eevee.gtao_factor = 1.2
    bpy.context.scene.eevee.use_bloom = True


def make_material(name: str, base_color: Tuple[float, float, float, float], emission_strength: float = 0.0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    principled = nodes.get("Principled BSDF")
    principled.inputs[0].default_value = base_color
    principled.inputs[7].default_value = 0.35
    principled.inputs[9].default_value = 0.9
    if emission_strength > 0.0:
        emission = nodes.new(type="ShaderNodeEmission")
        emission.inputs[0].default_value = base_color
        emission.inputs[1].default_value = emission_strength
        mix = nodes.new(type="ShaderNodeAddShader")
        out = nodes.get("Material Output")
        links.new(principled.outputs[0], mix.inputs[0])
        links.new(emission.outputs[0], mix.inputs[1])
        links.new(mix.outputs[0], out.inputs[0])
    return mat


def color_lerp(a: Sequence[float], b: Sequence[float], t: float) -> Tuple[float, float, float, float]:
    t = max(0.0, min(1.0, float(t)))
    return tuple(a[i] * (1.0 - t) + b[i] * t for i in range(4))


def magnitude_color(magnitude: float, min_mag: float, max_mag: float) -> Tuple[float, float, float, float]:
    if not math.isfinite(magnitude):
        return MID_COLOR
    if not math.isfinite(min_mag) or not math.isfinite(max_mag) or max_mag <= min_mag:
        return MID_COLOR
    t = (magnitude - min_mag) / (max_mag - min_mag)
    if t < 0.5:
        return color_lerp(LOW_COLOR, MID_COLOR, t * 2.0)
    return color_lerp(MID_COLOR, HIGH_COLOR, (t - 0.5) * 2.0)


def add_primitive_uv_sphere(name: str, radius: float, location: Vector, material: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def add_primitive_cylinder(name: str, radius: float, depth: float, location: Vector, rotation: Euler, material: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def add_arrow(name: str, origin: Vector, vec: Vector, color: Tuple[float, float, float, float], magnitude: float) -> bpy.types.Object:
    direction = vec.normalized() if vec.length > 1e-12 else Vector((0.0, 0.0, 1.0))
    vec_len = max(vec.length, 1e-6)
    shaft_len = max(vec_len * VECTOR_SCALE - ARROW_HEAD_LENGTH, ARROW_BASE_LENGTH)
    head_len = ARROW_HEAD_LENGTH
    shaft_end = origin + direction * shaft_len
    tip = origin + direction * (shaft_len + head_len)

    line_mat = make_material(f"{name}_mat", color, emission_strength=4.0)
    shaft_rot = direction.to_track_quat("Z", "Y").to_euler()
    head_rot = direction.to_track_quat("Z", "Y").to_euler()

    shaft = add_primitive_cylinder(f"{name}_shaft", ARROW_RADIUS, shaft_len, origin + direction * (shaft_len * 0.5), shaft_rot, line_mat)
    head = add_primitive_cylinder(f"{name}_head", ARROW_HEAD_RADIUS, head_len, shaft_end + direction * (head_len * 0.5), head_rot, line_mat)
    head.scale = (1.0, 1.0, 1.0)

    root = bpy.data.objects.new(f"{name}_root", None)
    bpy.context.collection.objects.link(root)
    shaft.parent = root
    head.parent = root
    root.location = origin
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.001

    # Add a small origin dot so the vector starts are easier to read in space.
    dot_mat = make_material(f"{name}_dot_mat", color, emission_strength=2.0)
    dot = add_primitive_uv_sphere(f"{name}_dot", ORIGIN_DOT_RADIUS, origin, dot_mat)
    dot.parent = root

    return root


def add_ground_plane() -> None:
    bpy.ops.mesh.primitive_plane_add(size=GROUND_SIZE, location=(0.0, 0.0, 0.0))
    plane = bpy.context.object
    plane.name = "Ground"
    mat = make_material("GroundMat", (0.03, 0.03, 0.04, 1.0), emission_strength=0.0)
    plane.data.materials.append(mat)
    plane.rotation_euler = (0.0, 0.0, 0.0)


def add_camera_and_lights() -> None:
    bpy.ops.object.camera_add(location=(CAMERA_DISTANCE, -CAMERA_DISTANCE * 0.15, CAMERA_ELEVATION))
    cam = bpy.context.object
    cam.rotation_euler = Euler((math.radians(68.0), 0.0, math.radians(84.0)), "XYZ")
    cam.data.lens = 55
    cam.data.dof.use_dof = True
    cam.data.dof.aperture_fstop = 3.5
    bpy.context.scene.camera = cam

    bpy.ops.object.light_add(type="AREA", location=(2.5, -2.0, 3.0))
    key = bpy.context.object
    key.data.energy = 2200
    key.data.shape = "RECTANGLE"
    key.data.size = 4.0
    key.data.size_y = 4.0
    key.rotation_euler = Euler((math.radians(55.0), 0.0, math.radians(40.0)), "XYZ")

    bpy.ops.object.light_add(type="AREA", location=(-2.0, 1.5, 2.0))
    fill = bpy.context.object
    fill.data.energy = 700
    fill.data.shape = "RECTANGLE"
    fill.data.size = 3.0
    fill.data.size_y = 3.0
    fill.rotation_euler = Euler((math.radians(78.0), 0.0, math.radians(-120.0)), "XYZ")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def read_output_vectors(csv_path: Path) -> List[VectorRow]:
    rows: List[VectorRow] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                frame_idx = int(float(row["frame_idx"]))
                origin = Vector((float(row["sensor_table_x_m"]), float(row["sensor_table_y_m"]), float(row["sensor_table_z_m"])))
                vec = Vector((float(row["mag_table_x_uT"]), float(row["mag_table_y_uT"]), float(row["mag_table_z_uT"])))
                magnitude = vec.length
                valid = all(math.isfinite(v) for v in (*origin, *vec))
                rows.append(VectorRow(frame_idx=frame_idx, origin=origin, vec=vec, magnitude=magnitude, valid=valid))
            except Exception:
                continue
    return rows


def read_frozen_vectors(csv_path: Path) -> List[VectorRow]:
    rows: List[VectorRow] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                frame_idx = int(float(row["frame_idx"]))
                origin = Vector((float(row["sensor_table_x_m"]), float(row["sensor_table_y_m"]), float(row["sensor_table_z_m"])))
                vec = Vector((float(row["mag_table_x_uT"]), float(row["mag_table_y_uT"]), float(row["mag_table_z_uT"])))
                magnitude = vec.length
                valid = all(math.isfinite(v) for v in (*origin, *vec))
                rows.append(VectorRow(frame_idx=frame_idx, origin=origin, vec=vec, magnitude=magnitude, valid=valid))
            except Exception:
                continue
    return rows


# -----------------------------------------------------------------------------
# Main build
# -----------------------------------------------------------------------------


def build_scene(output_csv: Path, frozen_csv: Optional[Path] = None) -> None:
    clean_scene()
    set_render_preset()
    add_ground_plane()
    add_camera_and_lights()

    live_rows = [row for row in read_output_vectors(output_csv) if row.valid]
    if MAX_ROWS > 0:
        live_rows = live_rows[:MAX_ROWS]

    frozen_rows: List[VectorRow] = []
    if frozen_csv is not None and frozen_csv.exists():
        frozen_rows = [row for row in read_frozen_vectors(frozen_csv) if row.valid]

    all_mags = [row.magnitude for row in live_rows + frozen_rows if math.isfinite(row.magnitude)]
    min_mag = min(all_mags) if all_mags else 0.0
    max_mag = max(all_mags) if all_mags else 1.0

    # Create a faint cloud of points to anchor the field visually.
    point_mat = make_material("FieldPointMat", (0.7, 0.7, 0.75, 1.0), emission_strength=1.2)
    for idx, row in enumerate(live_rows[::max(1, FRAME_STEP)]):
        add_primitive_uv_sphere(f"FieldPoint_{idx:04d}", FIELD_POINT_SIZE, row.origin, point_mat)

    # Live vectors - brighter and slightly transparent-looking via emissive materials.
    for idx, row in enumerate(live_rows):
        color = magnitude_color(row.magnitude, min_mag, max_mag)
        add_arrow(f"LiveVec_{idx:04d}", row.origin, row.vec, color, row.magnitude)

    # Frozen vectors - slightly more muted, but still vivid.
    for idx, row in enumerate(frozen_rows):
        color = magnitude_color(row.magnitude, min_mag, max_mag)
        muted = (color[0] * 0.8, color[1] * 0.8, color[2] * 0.8, 1.0)
        add_arrow(f"FrozenVec_{idx:04d}", row.origin, row.vec, muted, row.magnitude)

    # Optional camera orbit keyframes for a presentation-style turntable.
    if USE_ANIMATION:
        cam = bpy.context.scene.camera
        scene = bpy.context.scene
        scene.frame_start = 1
        scene.frame_end = 180
        for f in [1, 60, 120, 180]:
            angle = math.radians(CAMERA_ORBIT_DEG) * (f / 180.0)
            x = CAMERA_DISTANCE * math.cos(angle)
            y = CAMERA_DISTANCE * math.sin(angle)
            cam.location = (x, y, CAMERA_ELEVATION)
            cam.keyframe_insert(data_path="location", frame=f)
            cam.rotation_euler = Euler((math.radians(68.0), 0.0, angle + math.radians(90.0)), "XYZ")
            cam.keyframe_insert(data_path="rotation_euler", frame=f)

    bpy.context.view_layer.update()


if __name__ == "__main__":
    build_scene(OUTPUT_CSV, FROZEN_CSV)
    print(f"Built Blender scene from: {OUTPUT_CSV}")
    if FROZEN_CSV.exists():
        print(f"Included frozen vectors from: {FROZEN_CSV}")
