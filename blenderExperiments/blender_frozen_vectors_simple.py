"""Very simple frozen-vector loader for Blender.

How to use:
1. Open Blender.
2. Go to Scripting.
3. Open this file.
4. Set FROZEN_CSV to your CSV.
5. Run Script.
"""

from pathlib import Path
import csv
import math

import bpy
from mathutils import Vector


# Update this path to your frozen vectors file.
FROZEN_CSV = Path(
    r"C:\Users\scedg10\OneDrive - Cardiff University\projects\openMagMapper\dataKeep\Exp_cam_frozenVectors_USBwebcam_JLC1080_20260716_143634.csv"
)

# Match runExperiment.py scaling logic.
BASE_VIEW_SCALE_M_PER_UT = 15e-2 / 1000.0
VECTOR_SCALE_MULTIPLIER = 5.0
VECTOR_LENGTH_POWER = 0.5

# Simple arrow geometry.
SHAFT_RADIUS = 0.002
HEAD_RADIUS = 0.006
HEAD_LENGTH = 0.02

_MATERIAL_CACHE = {}


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def read_frozen_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ox = float(row["sensor_table_x_m"])
                oy = float(row["sensor_table_y_m"])
                oz = float(row["sensor_table_z_m"])
                vx = float(row["mag_table_x_uT"])
                vy = float(row["mag_table_y_uT"])
                vz = float(row["mag_table_z_uT"])
            except Exception:
                continue

            if not all(math.isfinite(v) for v in [ox, oy, oz, vx, vy, vz]):
                continue

            rows.append((Vector((ox, oy, oz)), Vector((vx, vy, vz))))
    return rows


def scale_magnetic_vector(raw_vec_ut: Vector):
    mag_ut = raw_vec_ut.length
    if not math.isfinite(mag_ut) or mag_ut <= 0.0:
        return Vector((0.0, 0.0, 0.0)), 0.0

    unit_vec = raw_vec_ut / mag_ut
    scaled_len_m = BASE_VIEW_SCALE_M_PER_UT * VECTOR_SCALE_MULTIPLIER * (mag_ut ** VECTOR_LENGTH_POWER)
    return unit_vec * scaled_len_m, mag_ut


def _lerp(a: float, b: float, t: float) -> float:
    return a * (1.0 - t) + b * t


def _magnitude_to_color(mag: float, min_mag: float, max_mag: float):
    if not math.isfinite(mag):
        return (0.7, 0.7, 0.7, 1.0)
    if not math.isfinite(min_mag) or not math.isfinite(max_mag) or max_mag <= min_mag:
        return (0.1, 0.9, 0.9, 1.0)

    t = max(0.0, min(1.0, (mag - min_mag) / (max_mag - min_mag)))

    # Blue -> Cyan -> Yellow gradient.
    if t < 0.5:
        u = t / 0.5
        return (_lerp(0.08, 0.10, u), _lerp(0.35, 0.95, u), _lerp(1.00, 0.95, u), 1.0)

    u = (t - 0.5) / 0.5
    return (_lerp(0.10, 1.00, u), _lerp(0.95, 0.92, u), _lerp(0.95, 0.20, u), 1.0)


def _material_for_color(color):
    key = tuple(round(c, 3) for c in color)
    if key in _MATERIAL_CACHE:
        return _MATERIAL_CACHE[key]

    mat = bpy.data.materials.new(name=f"FrozenVecMat_{len(_MATERIAL_CACHE):03d}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.35
    _MATERIAL_CACHE[key] = mat
    return mat


def add_arrow(origin: Vector, vec_ut: Vector, index: int, min_mag: float, max_mag: float):
    scaled_vec_m, mag_ut = scale_magnetic_vector(vec_ut)
    if scaled_vec_m.length < 1e-9:
        return

    color = _magnitude_to_color(mag_ut, min_mag, max_mag)
    material = _material_for_color(color)

    # Center each arrow on the physical sensor location (origin).
    start = origin - (0.5 * scaled_vec_m)
    end = origin + (0.5 * scaled_vec_m)
    axis_vec = end - start

    direction = axis_vec.normalized()
    total_len = axis_vec.length
    head_len = min(HEAD_LENGTH, total_len * 0.45)
    shaft_len = max(total_len - head_len, total_len * 0.55)

    shaft_center = start + direction * (shaft_len * 0.5)
    head_center = start + direction * (shaft_len + head_len * 0.5)

    rot = direction.to_track_quat("Z", "Y").to_euler()

    bpy.ops.mesh.primitive_cylinder_add(
        radius=SHAFT_RADIUS,
        depth=shaft_len,
        location=shaft_center,
        rotation=rot,
    )
    shaft = bpy.context.object
    shaft.name = f"FrozenVec_{index:04d}_shaft"
    if len(shaft.data.materials) == 0:
        shaft.data.materials.append(material)
    else:
        shaft.data.materials[0] = material

    bpy.ops.mesh.primitive_cone_add(
        radius1=HEAD_RADIUS,
        radius2=0.0,
        depth=head_len,
        location=head_center,
        rotation=rot,
    )
    head = bpy.context.object
    head.name = f"FrozenVec_{index:04d}_head"
    if len(head.data.materials) == 0:
        head.data.materials.append(material)
    else:
        head.data.materials[0] = material



def build():
    if not FROZEN_CSV.exists():
        raise FileNotFoundError(f"Frozen CSV not found: {FROZEN_CSV}")

    # clear_scene()

    rows = read_frozen_rows(FROZEN_CSV)
    magnitudes = [vec.length for _, vec in rows if math.isfinite(vec.length)]
    min_mag = min(magnitudes) if magnitudes else 0.0
    max_mag = max(magnitudes) if magnitudes else 1.0

    for i, (origin, vec) in enumerate(rows):
        add_arrow(origin, vec, i, min_mag, max_mag)

    print(f"Loaded {len(rows)} frozen vectors from {FROZEN_CSV}")


if __name__ == "__main__":
    build()
