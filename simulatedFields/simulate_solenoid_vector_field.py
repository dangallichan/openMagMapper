# This version is pure AI-generated and works on 

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

MU0 = 4.0e-7 * np.pi


@dataclass
class SolenoidParams:
    radius_m: float
    length_m: float
    turns: int
    current_a: float
    segments_per_turn: int


def build_solenoid_segments(params: SolenoidParams):
    total_segments = max(8, int(params.turns * params.segments_per_turn))
    theta = np.linspace(0.0, 2.0 * np.pi * params.turns, total_segments + 1)
    y = np.linspace(-0.5 * params.length_m, 0.5 * params.length_m, total_segments + 1)

    x = params.radius_m * np.cos(theta)
    z = params.radius_m * np.sin(theta)

    pts = np.stack((x, y, z), axis=1)
    seg_start = pts[:-1]
    seg_end = pts[1:]
    dl = seg_end - seg_start
    mid = 0.5 * (seg_start + seg_end)
    return mid, dl, pts


def compute_biot_savart_field(points_xyz: np.ndarray, seg_mid: np.ndarray, dl: np.ndarray, current_a: float):
    # points_xyz: (P, 3), seg_mid/dl: (S, 3)
    r = points_xyz[:, None, :] - seg_mid[None, :, :]  # (P, S, 3)
    r_norm = np.linalg.norm(r, axis=2)

    # Avoid singular behavior at/near wire.
    eps = 1e-9
    r_norm = np.maximum(r_norm, eps)

    cross = np.cross(dl[None, :, :], r)
    inv_r3 = 1.0 / (r_norm ** 3)
    d_b = cross * inv_r3[:, :, None]

    prefactor = MU0 * current_a / (4.0 * np.pi)
    b = prefactor * np.sum(d_b, axis=1)
    return b


def make_grid(extent_xy_m: float, extent_z_m: float, n_xy: int, n_z: int):
    # Keep legacy CLI names but treat extent_z/n_z as the principal-axis (Y) range.
    xs = np.linspace(-extent_xy_m, extent_xy_m, n_xy)
    ys = np.linspace(-extent_z_m, extent_z_m, n_z)
    zs = np.linspace(-extent_xy_m, extent_xy_m, n_xy)
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack((xg.ravel(), yg.ravel(), zg.ravel()), axis=1)
    return xg, yg, zg, pts


def add_solenoid_wire(ax, wire_pts):
    ax.plot(wire_pts[:, 0], wire_pts[:, 1], wire_pts[:, 2], color="black", linewidth=1.5, alpha=0.85)


def sample_field_trilinear(point_xyz: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, bx: np.ndarray, by: np.ndarray, bz: np.ndarray):
    x, y, z = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])
    if x < xs[0] or x > xs[-1] or y < ys[0] or y > ys[-1] or z < zs[0] or z > zs[-1]:
        return None

    ix1 = int(np.searchsorted(xs, x, side="right"))
    iy1 = int(np.searchsorted(ys, y, side="right"))
    iz1 = int(np.searchsorted(zs, z, side="right"))

    ix1 = min(max(ix1, 1), len(xs) - 1)
    iy1 = min(max(iy1, 1), len(ys) - 1)
    iz1 = min(max(iz1, 1), len(zs) - 1)

    ix0 = ix1 - 1
    iy0 = iy1 - 1
    iz0 = iz1 - 1

    dx = float(xs[ix1] - xs[ix0])
    dy = float(ys[iy1] - ys[iy0])
    dz = float(zs[iz1] - zs[iz0])
    if dx <= 0 or dy <= 0 or dz <= 0:
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


def integrate_streamline(seed_xyz: np.ndarray, direction: float, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, bx: np.ndarray, by: np.ndarray, bz: np.ndarray, step_m: float, max_steps: int):
    pts = [seed_xyz.astype(float)]
    p = seed_xyz.astype(float)

    for _ in range(max_steps):
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
            p_next[0] < xs[0] or p_next[0] > xs[-1]
            or p_next[1] < ys[0] or p_next[1] > ys[-1]
            or p_next[2] < zs[0] or p_next[2] > zs[-1]
        ):
            break

        pts.append(p_next)
        p = p_next

    return np.array(pts)


def make_streamline_seeds(radius_m: float, y_half_extent_m: float, n_phi: int, n_y: int, seed_radius_scale: float):
    r = max(1e-6, float(seed_radius_scale) * float(radius_m))
    phis = np.linspace(0.0, 2.0 * np.pi, max(4, int(n_phi)), endpoint=False)
    ys = np.linspace(-y_half_extent_m, y_half_extent_m, max(1, int(n_y)))
    seeds = []
    for y in ys:
        for phi in phis:
            seeds.append(np.array([r * np.cos(phi), y, r * np.sin(phi)], dtype=float))
    return seeds


def main():
    parser = argparse.ArgumentParser(description="Simulate 3D magnetic vector field around a parameterized solenoid.")
    parser.add_argument("--radius", type=float, default=0.013, help="Solenoid radius in meters.")
    parser.add_argument("--length", type=float, default=0.05, help="Solenoid length in meters.")
    parser.add_argument("--turns", type=int, default=80, help="Number of turns.")
    parser.add_argument("--current", type=float, default=1.0, help="Current in amperes.")
    parser.add_argument("--segments-per-turn", type=int, default=24, help="Line segments per turn for Biot-Savart integration.")

    parser.add_argument("--extent-xy", type=float, default=0.12, help="Half-width of simulation region in x/y (m).")
    parser.add_argument("--extent-z", type=float, default=0.18, help="Half-width along solenoid principal axis Y (m).")
    parser.add_argument("--n-xy", type=int, default=50, help="Grid points along x and y.")
    parser.add_argument("--n-z", type=int, default=50, help="Grid points along solenoid principal axis Y.")

    parser.add_argument("--quiver-step", type=int, default=2, help="Subsample step for quiver arrows.")
    parser.add_argument("--scale", type=float, default=1.0, help="Arrow length scaling multiplier.")
    parser.add_argument("--no-streamlines", action="store_true", help="Disable streamline integration/plotting.")
    parser.add_argument("--stream-seeds-phi", type=int, default=32, help="Number of streamline seed points around each ring.")
    parser.add_argument("--stream-seeds-y", type=int, default=11, help="Number of seed rings along the Y axis.")
    parser.add_argument("--stream-seed-radius-scale", type=float, default=5, help="Seed-ring radius as a multiple of solenoid radius.")
    parser.add_argument("--stream-step", type=float, default=0.005, help="Streamline integration step in meters.")
    parser.add_argument("--stream-max-steps", type=int, default=600, help="Maximum integration steps in each direction.")
    parser.add_argument("--streamline-alpha", type=float, default=0.8, help="Alpha for plotted streamlines.")
    parser.add_argument("--streamline-lw", type=float, default=1.0, help="Line width for plotted streamlines.")
    parser.add_argument("--save-npz", default="", help="Optional output .npz path to save grid and vector field.")
    args = parser.parse_args()

    params = SolenoidParams(
        radius_m=float(args.radius),
        length_m=float(args.length),
        turns=int(args.turns),
        current_a=float(args.current),
        segments_per_turn=int(args.segments_per_turn),
    )

    seg_mid, dl, wire_pts = build_solenoid_segments(params)
    xg, yg, zg, points = make_grid(
        extent_xy_m=float(args.extent_xy),
        extent_z_m=float(args.extent_z),
        n_xy=int(args.n_xy),
        n_z=int(args.n_z),
    )

    b_vec = compute_biot_savart_field(points, seg_mid, dl, params.current_a)
    bx = b_vec[:, 0].reshape(xg.shape)
    by = b_vec[:, 1].reshape(xg.shape)
    bz = b_vec[:, 2].reshape(xg.shape)
    b_mag = np.linalg.norm(b_vec, axis=1).reshape(xg.shape)

    if args.save_npz:
        np.savez(
            args.save_npz,
            x=xg,
            y=yg,
            z=zg,
            bx=bx,
            by=by,
            bz=bz,
            bmag=b_mag,
            radius=params.radius_m,
            length=params.length_m,
            turns=params.turns,
            current=params.current_a,
        )
        print(f"Saved field data to: {args.save_npz}")

    # Subsample for cleaner quiver rendering.
    step = max(1, int(args.quiver_step))
    xs = xg[::step, ::step, ::step]
    ys = yg[::step, ::step, ::step]
    zs = zg[::step, ::step, ::step]
    us = bx[::step, ::step, ::step]
    vs = by[::step, ::step, ::step]
    ws = bz[::step, ::step, ::step]
    mags = b_mag[::step, ::step, ::step]

    # Normalize direction for visualization but keep color by magnitude.
    norm = np.sqrt(us ** 2 + vs ** 2 + ws ** 2)
    norm = np.maximum(norm, 1e-12)
    udir = us / norm
    vdir = vs / norm
    wdir = ws / norm

    mag_min = float(np.min(mags))
    mag_max = float(np.max(mags))
    mag_norm = (mags - mag_min) / (mag_max - mag_min + 1e-12)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.cm.plasma
    colors = cmap(mag_norm.ravel())

    arrow_len = 0.018 * float(args.scale)
    ax.quiver(
        xs.ravel(), ys.ravel(), zs.ravel(),
        udir.ravel(), vdir.ravel(), wdir.ravel(),
        length=arrow_len,
        normalize=False,
        colors=colors,
        linewidths=0.7,
        alpha=0.9,
    )

    add_solenoid_wire(ax, wire_pts)

    streamlines_plotted = 0
    if not args.no_streamlines:
        xs_axis = xg[:, 0, 0]
        ys_axis = yg[0, :, 0]
        zs_axis = zg[0, 0, :]

        seed_half_span_y = min(float(args.extent_z), 0.5 * float(args.length))
        seeds = make_streamline_seeds(
            radius_m=params.radius_m,
            y_half_extent_m=seed_half_span_y,
            n_phi=int(args.stream_seeds_phi),
            n_y=int(args.stream_seeds_y),
            seed_radius_scale=float(args.stream_seed_radius_scale),
        )

        for seed in seeds:
            fwd = integrate_streamline(
                seed,
                direction=1.0,
                xs=xs_axis,
                ys=ys_axis,
                zs=zs_axis,
                bx=bx,
                by=by,
                bz=bz,
                step_m=float(args.stream_step),
                max_steps=int(args.stream_max_steps),
            )
            bwd = integrate_streamline(
                seed,
                direction=-1.0,
                xs=xs_axis,
                ys=ys_axis,
                zs=zs_axis,
                bx=bx,
                by=by,
                bz=bz,
                step_m=float(args.stream_step),
                max_steps=int(args.stream_max_steps),
            )

            if len(fwd) < 2 and len(bwd) < 2:
                continue

            if len(bwd) > 1:
                bwd_rev = bwd[::-1]
                line_pts = np.vstack((bwd_rev[:-1], fwd)) if len(fwd) > 0 else bwd_rev
            else:
                line_pts = fwd

            if len(line_pts) >= 2:
                ax.plot(
                    line_pts[:, 0],
                    line_pts[:, 1],
                    line_pts[:, 2],
                    color="#66e0ff",
                    linewidth=float(args.streamline_lw),
                    alpha=float(args.streamline_alpha),
                )
                streamlines_plotted += 1

    ax.set_title("Parameterized Solenoid 3D Magnetic Field")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    lim_xy = float(args.extent_xy)
    lim_y = float(args.extent_z)
    ax.set_xlim(-lim_xy, lim_xy)
    ax.set_ylim(-lim_y, lim_y)
    ax.set_zlim(-lim_xy, lim_xy)
    ax.set_box_aspect((1, lim_y / max(lim_xy, 1e-6), 1))

    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(mag_norm.ravel())
    cbar = fig.colorbar(mappable, ax=ax, pad=0.08, shrink=0.75)
    cbar.set_label("Relative |B|")

    print(
        "Field stats: "
        f"|B|min={np.min(b_mag):.3e} T, "
        f"|B|max={np.max(b_mag):.3e} T, "
        f"segments={len(dl)}, "
        f"grid_points={len(points)}, "
        f"streamlines={streamlines_plotted}"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
