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
    z = np.linspace(-0.5 * params.length_m, 0.5 * params.length_m, total_segments + 1)

    x = params.radius_m * np.cos(theta)
    y = params.radius_m * np.sin(theta)

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
    xs = np.linspace(-extent_xy_m, extent_xy_m, n_xy)
    ys = np.linspace(-extent_xy_m, extent_xy_m, n_xy)
    zs = np.linspace(-extent_z_m, extent_z_m, n_z)
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack((xg.ravel(), yg.ravel(), zg.ravel()), axis=1)
    return xg, yg, zg, pts


def add_solenoid_wire(ax, wire_pts):
    ax.plot(wire_pts[:, 0], wire_pts[:, 1], wire_pts[:, 2], color="black", linewidth=1.5, alpha=0.85)


def main():
    parser = argparse.ArgumentParser(description="Simulate 3D magnetic vector field around a parameterized solenoid.")
    parser.add_argument("--radius", type=float, default=0.04, help="Solenoid radius in meters.")
    parser.add_argument("--length", type=float, default=0.12, help="Solenoid length in meters.")
    parser.add_argument("--turns", type=int, default=80, help="Number of turns.")
    parser.add_argument("--current", type=float, default=1.0, help="Current in amperes.")
    parser.add_argument("--segments-per-turn", type=int, default=24, help="Line segments per turn for Biot-Savart integration.")

    parser.add_argument("--extent-xy", type=float, default=0.12, help="Half-width of simulation region in x/y (m).")
    parser.add_argument("--extent-z", type=float, default=0.18, help="Half-width of simulation region in z (m).")
    parser.add_argument("--n-xy", type=int, default=13, help="Grid points along x and y.")
    parser.add_argument("--n-z", type=int, default=15, help="Grid points along z.")

    parser.add_argument("--quiver-step", type=int, default=2, help="Subsample step for quiver arrows.")
    parser.add_argument("--scale", type=float, default=1.0, help="Arrow length scaling multiplier.")
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

    ax.set_title("Parameterized Solenoid 3D Magnetic Field")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    lim_xy = float(args.extent_xy)
    lim_z = float(args.extent_z)
    ax.set_xlim(-lim_xy, lim_xy)
    ax.set_ylim(-lim_xy, lim_xy)
    ax.set_zlim(-lim_z, lim_z)
    ax.set_box_aspect((1, 1, lim_z / max(lim_xy, 1e-6)))

    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(mag_norm.ravel())
    cbar = fig.colorbar(mappable, ax=ax, pad=0.08, shrink=0.75)
    cbar.set_label("Relative |B|")

    print(
        "Field stats: "
        f"|B|min={np.min(b_mag):.3e} T, "
        f"|B|max={np.max(b_mag):.3e} T, "
        f"segments={len(dl)}, "
        f"grid_points={len(points)}"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
