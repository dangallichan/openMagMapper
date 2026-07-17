"""3D analytic solenoid field + streamline exporter for Three.js particle paths.

This script extends the 2D analytic model to a full 3D Cartesian grid by using
the cylindrical analytic loop solution and mapping it to Cartesian components.

Outputs:
- Optional NPZ of field grid.
- JSON containing packed streamline arrays ready for Three.js BufferGeometry.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import special


@dataclass
class SolenoidParams:
    radius_m: float
    length_m: float
    turns: int
    current_a: float


def magnetic_field_loop_cyl(r, x, a):
    """Analytic field of one loop in cylindrical coordinates.

    Args:
        r: Radial distance from loop axis (m), scalar or ndarray.
        x: Axial offset from loop center (m), scalar or ndarray.
        a: Loop radius (m).

    Returns:
        Tuple (Br, Bx) for radial and axial components in normalized units.
    """
    rr = np.asarray(r, dtype=float)
    xx = np.asarray(x, dtype=float)

    rr_abs = np.abs(rr)
    den = (a + rr_abs) ** 2 + xx ** 2
    term1 = 1.0 / np.sqrt(den)
    term2 = 1.0 / ((a - rr_abs) ** 2 + xx ** 2)

    m = 4.0 * a * rr_abs / den
    m = np.clip(m, 0.0, 1.0)

    ell_k = special.ellipk(m)
    ell_e = special.ellipe(m)

    bx = (1.0 / np.pi) * term1 * (ell_e * (a**2 - rr_abs**2 - xx**2) * term2 + ell_k)

    rr_safe = np.where(rr_abs < 1e-12, 1.0, rr_abs)
    br = (xx / rr_safe) * (1.0 / np.pi) * term1 * (ell_e * (a**2 + rr_abs**2 + xx**2) * term2 - ell_k)
    br = np.where(rr_abs < 1e-9, 0.0, br)
    br = np.where(rr < 0.0, -br, br)

    return br, bx


def magnetic_field_solenoid_cyl(r, x, params: SolenoidParams):
    """Approximate finite solenoid field by summing many loops along x-axis."""
    rr = np.asarray(r, dtype=float)
    xx = np.asarray(x, dtype=float)

    bx_total = np.zeros_like(xx)
    br_total = np.zeros_like(rr)

    n_loops = max(1, int(params.turns))
    x_loops = np.linspace(-0.5 * params.length_m, 0.5 * params.length_m, n_loops)

    for x_loop in x_loops:
        br, bx = magnetic_field_loop_cyl(rr, xx - x_loop, params.radius_m)
        br_total += params.current_a * br
        bx_total += params.current_a * bx

    return br_total, bx_total


def make_grid(extent_x_m: float, extent_r_m: float, nx: int, ny: int, nz: int):
    xs = np.linspace(-extent_x_m, extent_x_m, int(nx))
    ys = np.linspace(-extent_r_m, extent_r_m, int(ny))
    zs = np.linspace(-extent_r_m, extent_r_m, int(nz))
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
    return xs, ys, zs, xg, yg, zg


def compute_field_cartesian(xg, yg, zg, params: SolenoidParams):
    """Compute Bx, By, Bz on grid for a solenoid oriented along +X axis."""
    rho = np.sqrt(yg**2 + zg**2)
    br, bx = magnetic_field_solenoid_cyl(rho, xg, params)

    rho_safe = np.where(rho < 1e-12, 1.0, rho)
    by = np.where(rho < 1e-9, 0.0, br * (yg / rho_safe))
    bz = np.where(rho < 1e-9, 0.0, br * (zg / rho_safe))
    bmag = np.sqrt(bx**2 + by**2 + bz**2)
    return bx, by, bz, bmag


def sample_field_trilinear(point_xyz, xs, ys, zs, bx, by, bz):
    x, y, z = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])

    if x < xs[0] or x > xs[-1] or y < ys[0] or y > ys[-1] or z < zs[0] or z > zs[-1]:
        return None

    ix1 = int(np.searchsorted(xs, x, side="right"))
    iy1 = int(np.searchsorted(ys, y, side="right"))
    iz1 = int(np.searchsorted(zs, z, side="right"))

    ix1 = min(max(ix1, 1), len(xs) - 1)
    iy1 = min(max(iy1, 1), len(ys) - 1)
    iz1 = min(max(iz1, 1), len(zs) - 1)

    ix0, iy0, iz0 = ix1 - 1, iy1 - 1, iz1 - 1

    dx = float(xs[ix1] - xs[ix0])
    dy = float(ys[iy1] - ys[iy0])
    dz = float(zs[iz1] - zs[iz0])
    if dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
        return None

    tx = (x - float(xs[ix0])) / dx
    ty = (y - float(ys[iy0])) / dy
    tz = (z - float(zs[iz0])) / dz

    def interp(c000, c001, c010, c011, c100, c101, c110, c111):
        c00 = c000 * (1.0 - tx) + c100 * tx
        c01 = c001 * (1.0 - tx) + c101 * tx
        c10 = c010 * (1.0 - tx) + c110 * tx
        c11 = c011 * (1.0 - tx) + c111 * tx
        c0 = c00 * (1.0 - ty) + c10 * ty
        c1 = c01 * (1.0 - ty) + c11 * ty
        return c0 * (1.0 - tz) + c1 * tz

    bxi = interp(
        bx[ix0, iy0, iz0], bx[ix0, iy0, iz1], bx[ix0, iy1, iz0], bx[ix0, iy1, iz1],
        bx[ix1, iy0, iz0], bx[ix1, iy0, iz1], bx[ix1, iy1, iz0], bx[ix1, iy1, iz1],
    )
    byi = interp(
        by[ix0, iy0, iz0], by[ix0, iy0, iz1], by[ix0, iy1, iz0], by[ix0, iy1, iz1],
        by[ix1, iy0, iz0], by[ix1, iy0, iz1], by[ix1, iy1, iz0], by[ix1, iy1, iz1],
    )
    bzi = interp(
        bz[ix0, iy0, iz0], bz[ix0, iy0, iz1], bz[ix0, iy1, iz0], bz[ix0, iy1, iz1],
        bz[ix1, iy0, iz0], bz[ix1, iy0, iz1], bz[ix1, iy1, iz0], bz[ix1, iy1, iz1],
    )

    return np.array([bxi, byi, bzi], dtype=float)


def integrate_streamline(seed_xyz, direction, xs, ys, zs, bx, by, bz, step_m, max_steps):
    """Integrate one streamline with midpoint RK2 in normalized field direction."""
    pts = [seed_xyz.astype(float)]
    p = seed_xyz.astype(float)

    for _ in range(int(max_steps)):
        b0 = sample_field_trilinear(p, xs, ys, zs, bx, by, bz)
        if b0 is None:
            break
        n0 = np.linalg.norm(b0)
        if n0 < 1e-14:
            break

        k1 = direction * (b0 / n0)
        pmid = p + 0.5 * step_m * k1

        b1 = sample_field_trilinear(pmid, xs, ys, zs, bx, by, bz)
        if b1 is None:
            break
        n1 = np.linalg.norm(b1)
        if n1 < 1e-14:
            break

        k2 = direction * (b1 / n1)
        p_next = p + step_m * k2

        if (
            p_next[0] < xs[0]
            or p_next[0] > xs[-1]
            or p_next[1] < ys[0]
            or p_next[1] > ys[-1]
            or p_next[2] < zs[0]
            or p_next[2] > zs[-1]
        ):
            break

        pts.append(p_next)
        p = p_next

    return np.asarray(pts)


def stitch_streamline(forward_pts, backward_pts):
    if len(forward_pts) < 2 and len(backward_pts) < 2:
        return np.zeros((0, 3), dtype=float)
    if len(backward_pts) > 1:
        back_rev = backward_pts[::-1]
        return np.vstack((back_rev[:-1], forward_pts)) if len(forward_pts) else back_rev
    return forward_pts


def polyline_length(points):
    if len(points) < 2:
        return 0.0
    d = points[1:] - points[:-1]
    return float(np.sum(np.linalg.norm(d, axis=1)))


def resample_polyline(points, ds_m):
    """Resample polyline to approximately constant spacing ds_m."""
    if len(points) < 2:
        return points

    ds = max(1e-6, float(ds_m))
    seg = points[1:] - points[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(cum[-1])

    if total <= ds:
        return points

    targets = np.arange(0.0, total + 0.5 * ds, ds)
    targets[-1] = total

    out = []
    j = 0
    for t in targets:
        while j < len(seg_len) - 1 and cum[j + 1] < t:
            j += 1
        l0, l1 = cum[j], cum[j + 1]
        if l1 <= l0 + 1e-12:
            out.append(points[j].copy())
            continue
        u = (t - l0) / (l1 - l0)
        p = points[j] * (1.0 - u) + points[j + 1] * u
        out.append(p)

    return np.asarray(out)


def make_ring_seeds(params: SolenoidParams, n_phi: int, n_x: int, radius_scale: float, x_span_scale: float):
    seed_r = max(1e-6, radius_scale * params.radius_m)
    half_span_x = 0.5 * params.length_m * max(0.0, x_span_scale)

    phis = np.linspace(0.0, 2.0 * np.pi, max(4, int(n_phi)), endpoint=False)
    xs = np.linspace(-half_span_x, half_span_x, max(1, int(n_x)))

    seeds = []
    for xv in xs:
        for phi in phis:
            yv = seed_r * np.cos(phi)
            zv = seed_r * np.sin(phi)
            seeds.append(np.array([xv, yv, zv], dtype=float))
    return seeds


def flatten_lines(lines):
    """Pack many polylines into flat buffers and offsets for WebGL upload."""
    offsets = [0]
    lengths = []
    flat = []

    for pts in lines:
        n = int(len(pts))
        lengths.append(n)
        offsets.append(offsets[-1] + n)
        if n:
            flat.extend(pts.astype(float).ravel().tolist())

    return {
        "offsets": offsets,
        "counts": lengths,
        "positions": flat,
    }


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="3D analytic solenoid field and streamline export for Three.js particle paths."
    )
    parser.add_argument("--radius", type=float, default=0.015, help="Solenoid radius in meters.")
    parser.add_argument("--length", type=float, default=0.05, help="Solenoid length in meters.")
    parser.add_argument("--turns", type=int, default=100, help="Number of loops for finite-solenoid approximation.")
    parser.add_argument("--current", type=float, default=1.0, help="Current scaling factor.")

    parser.add_argument("--extent-x", type=float, default=0.14, help="Half-extent along x (solenoid axis), meters.")
    parser.add_argument("--extent-r", type=float, default=0.14, help="Half-extent along y and z, meters.")
    parser.add_argument("--nx", type=int, default=64, help="Grid samples along x.")
    parser.add_argument("--ny", type=int, default=64, help="Grid samples along y.")
    parser.add_argument("--nz", type=int, default=64, help="Grid samples along z.")

    parser.add_argument("--seed-phi", type=int, default=36, help="Seeds per ring around x axis.")
    parser.add_argument("--seed-x", type=int, default=10, help="Number of seed rings along x.")
    parser.add_argument("--seed-radius-scale", type=float, default=5.0, help="Seed ring radius multiplier vs solenoid radius.")
    parser.add_argument("--seed-x-span-scale", type=float, default=1.2, help="Seed x-span multiplier vs solenoid length.")

    parser.add_argument("--stream-step", type=float, default=0.0035, help="Integration step size (m).")
    parser.add_argument("--stream-max-steps", type=int, default=900, help="Max integration steps per direction.")
    parser.add_argument("--stream-min-points", type=int, default=8, help="Discard lines shorter than this point count.")
    parser.add_argument("--resample-ds", type=float, default=0.003, help="Resampled spacing for particle path points (m).")

    parser.add_argument(
        "--out-json",
        default="simulatedFields/exports/solenoid_streamlines_3d.json",
        help="Output JSON path for streamline data.",
    )
    parser.add_argument(
        "--save-field-npz",
        default="",
        help="Optional NPZ output path for field grid (x/y/z + bx/by/bz + |B|).",
    )

    args = parser.parse_args()

    params = SolenoidParams(
        radius_m=float(args.radius),
        length_m=float(args.length),
        turns=int(args.turns),
        current_a=float(args.current),
    )

    xs, ys, zs, xg, yg, zg = make_grid(
        extent_x_m=float(args.extent_x),
        extent_r_m=float(args.extent_r),
        nx=int(args.nx),
        ny=int(args.ny),
        nz=int(args.nz),
    )
    bx, by, bz, bmag = compute_field_cartesian(xg, yg, zg, params)

    if args.save_field_npz:
        out_npz = Path(args.save_field_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_npz,
            x=xg,
            y=yg,
            z=zg,
            bx=bx,
            by=by,
            bz=bz,
            bmag=bmag,
            radius=params.radius_m,
            length=params.length_m,
            turns=params.turns,
            current=params.current_a,
        )
        print(f"Saved field grid to {out_npz}")

    seeds = make_ring_seeds(
        params=params,
        n_phi=int(args.seed_phi),
        n_x=int(args.seed_x),
        radius_scale=float(args.seed_radius_scale),
        x_span_scale=float(args.seed_x_span_scale),
    )

    raw_lines = []
    resampled_lines = []
    kept_seed_points = []

    for seed in seeds:
        fwd = integrate_streamline(
            seed_xyz=seed,
            direction=1.0,
            xs=xs,
            ys=ys,
            zs=zs,
            bx=bx,
            by=by,
            bz=bz,
            step_m=float(args.stream_step),
            max_steps=int(args.stream_max_steps),
        )
        bwd = integrate_streamline(
            seed_xyz=seed,
            direction=-1.0,
            xs=xs,
            ys=ys,
            zs=zs,
            bx=bx,
            by=by,
            bz=bz,
            step_m=float(args.stream_step),
            max_steps=int(args.stream_max_steps),
        )
        line = stitch_streamline(fwd, bwd)

        if len(line) < int(args.stream_min_points):
            continue

        line_resampled = resample_polyline(line, ds_m=float(args.resample_ds))
        if len(line_resampled) < 2:
            continue

        raw_lines.append(line)
        resampled_lines.append(line_resampled)
        kept_seed_points.append(seed)

    packed_raw = flatten_lines(raw_lines)
    packed_resampled = flatten_lines(resampled_lines)

    lengths_raw = [polyline_length(line) for line in raw_lines]
    lengths_resampled = [polyline_length(line) for line in resampled_lines]

    payload = {
        "meta": {
            "schema": "openmagmapper.streamlines.v1",
            "units": "meters",
            "frame": "right-handed, solenoid axis along +X",
        },
        "solenoid": {
            "radius_m": params.radius_m,
            "length_m": params.length_m,
            "turns": params.turns,
            "current_a": params.current_a,
        },
        "grid": {
            "shape": [int(args.nx), int(args.ny), int(args.nz)],
            "x_min_m": float(xs[0]),
            "x_max_m": float(xs[-1]),
            "y_min_m": float(ys[0]),
            "y_max_m": float(ys[-1]),
            "z_min_m": float(zs[0]),
            "z_max_m": float(zs[-1]),
        },
        "streamlines": {
            "count": len(raw_lines),
            "seed_points": np.asarray(kept_seed_points, dtype=float).ravel().tolist(),
            "raw": {
                "integration_step_m": float(args.stream_step),
                "lengths_m": lengths_raw,
                **packed_raw,
            },
            "resampled": {
                "sample_step_m": float(args.resample_ds),
                "lengths_m": lengths_resampled,
                **packed_resampled,
            },
        },
        "field_stats": {
            "b_min": float(np.min(bmag)),
            "b_max": float(np.max(bmag)),
            "b_mean": float(np.mean(bmag)),
        },
    }

    out_json = Path(args.out_json)
    write_json(out_json, payload)

    print(f"Saved streamline JSON to {out_json}")
    print(
        "Export summary: "
        f"streamlines={len(raw_lines)}, "
        f"raw_points={packed_raw['offsets'][-1] if packed_raw['offsets'] else 0}, "
        f"resampled_points={packed_resampled['offsets'][-1] if packed_resampled['offsets'] else 0}, "
        f"|B|min={np.min(bmag):.3e}, "
        f"|B|max={np.max(bmag):.3e}"
    )


if __name__ == "__main__":
    main()
