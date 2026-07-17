"""Export axisymmetric psi-contours for Three.js cylindrical streamline generation.

This script computes the poloidal flux function Psi(x, rho) for a finite solenoid
using analytic loop expressions (elliptic integrals), extracts 2D contour lines
in the x-rho plane, and exports them for cylindrical revolution in Three.js.

The output is intentionally 2D: one family of contours is enough for full 3D
streamline rendering in an axisymmetric field by revolving around the x-axis.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import special


@dataclass
class SolenoidParams:
    radius_m: float
    length_m: float
    turns: int
    current_a: float


def loop_vector_potential_azimuthal(rho, x, a):
    """Analytic azimuthal vector potential A_phi for one circular current loop.

    Notes:
    - The absolute scale is not critical for streamline geometry because field
      lines are level sets of Psi = rho * A_phi.
    - This expression is valid for rho > 0; we handle rho -> 0 numerically.
    """
    rr = np.asarray(rho, dtype=float)
    xx = np.asarray(x, dtype=float)

    rr_abs = np.abs(rr)
    den = (a + rr_abs) ** 2 + xx**2
    m = np.clip(4.0 * a * rr_abs / den, 0.0, 1.0)

    k = np.sqrt(m)
    k_safe = np.where(k < 1e-12, 1e-12, k)
    rr_safe = np.where(rr_abs < 1e-12, 1e-12, rr_abs)

    ell_k = special.ellipk(m)
    ell_e = special.ellipe(m)

    pref = (2.0 / np.pi) * np.sqrt(a / rr_safe)
    # Standard loop A_phi form in terms of complete elliptic integrals.
    aphi = pref * ((1.0 - 0.5 * m) * ell_k - ell_e) / k_safe

    # On-axis limit is finite in Psi but A_phi itself trends to 0 with rho.
    aphi = np.where(rr_abs < 1e-9, 0.0, aphi)
    return aphi


def psi_solenoid(xg, rg, params: SolenoidParams):
    """Compute Psi(x, rho) by summing loop contributions along x-axis."""
    psi = np.zeros_like(xg, dtype=float)
    n_loops = max(1, int(params.turns))
    x_loops = np.linspace(-0.5 * params.length_m, 0.5 * params.length_m, n_loops)

    for x_loop in x_loops:
        aphi = params.current_a * loop_vector_potential_azimuthal(rg, xg - x_loop, params.radius_m)
        psi += rg * aphi

    return psi


def magnetic_field_loop_cyl(rho, x, a):
    """Analytic single-loop field components (Br, Bx) in cylindrical frame."""
    rr = np.asarray(rho, dtype=float)
    xx = np.asarray(x, dtype=float)

    rr_abs = np.abs(rr)
    den = (a + rr_abs) ** 2 + xx**2
    term1 = 1.0 / np.sqrt(den)
    term2 = 1.0 / ((a - rr_abs) ** 2 + xx**2)

    m = np.clip(4.0 * a * rr_abs / den, 0.0, 1.0)
    ell_k = special.ellipk(m)
    ell_e = special.ellipe(m)

    bx = (1.0 / np.pi) * term1 * (ell_e * (a**2 - rr_abs**2 - xx**2) * term2 + ell_k)

    rr_safe = np.where(rr_abs < 1e-12, 1.0, rr_abs)
    br = (xx / rr_safe) * (1.0 / np.pi) * term1 * (ell_e * (a**2 + rr_abs**2 + xx**2) * term2 - ell_k)
    br = np.where(rr_abs < 1e-9, 0.0, br)
    br = np.where(rr < 0.0, -br, br)
    return br, bx


def magnetic_field_solenoid_cyl(rho, x, params: SolenoidParams):
    """Finite-solenoid field by loop summation: returns (Br, Bx)."""
    rr = np.asarray(rho, dtype=float)
    xx = np.asarray(x, dtype=float)

    br_total = np.zeros_like(rr)
    bx_total = np.zeros_like(xx)

    n_loops = max(1, int(params.turns))
    x_loops = np.linspace(-0.5 * params.length_m, 0.5 * params.length_m, n_loops)

    for x_loop in x_loops:
        br, bx = magnetic_field_loop_cyl(rr, xx - x_loop, params.radius_m)
        br_total += params.current_a * br
        bx_total += params.current_a * bx

    return br_total, bx_total


def field_strength_xrho(xg, rg, params: SolenoidParams):
    """Return (Br, Bx, |B|) on x-rho grid."""
    br, bx = magnetic_field_solenoid_cyl(rg, xg, params)
    bmag = np.sqrt(br**2 + bx**2)
    return br, bx, bmag


def polyline_length_2d(points):
    if len(points) < 2:
        return 0.0
    d = points[1:] - points[:-1]
    return float(np.sum(np.linalg.norm(d, axis=1)))


def resample_polyline_2d(points, ds_m):
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


def flatten_lines_2d(lines):
    offsets = [0]
    counts = []
    flat = []

    for pts in lines:
        n = int(len(pts))
        counts.append(n)
        offsets.append(offsets[-1] + n)
        if n:
            flat.extend(pts.astype(float).ravel().tolist())

    return {
        "offsets": offsets,
        "counts": counts,
        "positions2d": flat,
    }


def flatten_scalar_lines(values_lines):
    """Pack scalar values per polyline point into flat buffer + offsets/counts."""
    offsets = [0]
    counts = []
    flat = []

    for vals in values_lines:
        n = int(len(vals))
        counts.append(n)
        offsets.append(offsets[-1] + n)
        if n:
            flat.extend(np.asarray(vals, dtype=float).ravel().tolist())

    return {
        "offsets": offsets,
        "counts": counts,
        "values": flat,
    }


def sample_scalar_bilinear_xrho(x, rho, xs, rhos, scalar_xrho):
    """Bilinear interpolation on scalar grid indexed as [x, rho]."""
    xv = float(x)
    rv = float(rho)
    if xv < xs[0] or xv > xs[-1] or rv < rhos[0] or rv > rhos[-1]:
        return np.nan

    ix1 = int(np.searchsorted(xs, xv, side="right"))
    ir1 = int(np.searchsorted(rhos, rv, side="right"))
    ix1 = min(max(ix1, 1), len(xs) - 1)
    ir1 = min(max(ir1, 1), len(rhos) - 1)
    ix0 = ix1 - 1
    ir0 = ir1 - 1

    dx = float(xs[ix1] - xs[ix0])
    dr = float(rhos[ir1] - rhos[ir0])
    if dx <= 0.0 or dr <= 0.0:
        return np.nan

    tx = (xv - float(xs[ix0])) / dx
    tr = (rv - float(rhos[ir0])) / dr

    c00 = scalar_xrho[ix0, ir0]
    c01 = scalar_xrho[ix0, ir1]
    c10 = scalar_xrho[ix1, ir0]
    c11 = scalar_xrho[ix1, ir1]

    c0 = c00 * (1.0 - tx) + c10 * tx
    c1 = c01 * (1.0 - tx) + c11 * tx
    return float(c0 * (1.0 - tr) + c1 * tr)


def sample_bmag_on_lines(lines, xs, rhos, bmag_xrho):
    """Return list of bmag arrays corresponding point-wise to each polyline."""
    all_vals = []
    for line in lines:
        vals = [sample_scalar_bilinear_xrho(p[0], p[1], xs, rhos, bmag_xrho) for p in line]
        all_vals.append(np.asarray(vals, dtype=float))
    return all_vals


def extract_psi_contours(xs, rhos, psi_xr, levels, min_points, min_rho_keep):
    """Return list of polylines in [x, rho] for each contour segment."""
    fig, ax = plt.subplots(figsize=(6, 4))
    # Matplotlib expects Z indexed as [y, x], so transpose from [x, rho].
    cs = ax.contour(xs, rhos, psi_xr.T, levels=levels)
    plt.close(fig)

    lines = []
    line_levels = []
    for level_idx, level_val in enumerate(cs.levels):
        segs = cs.allsegs[level_idx]
        for seg in segs:
            if seg is None or len(seg) < min_points:
                continue
            # seg columns are [x, rho]
            if np.max(seg[:, 1]) < min_rho_keep:
                continue
            lines.append(np.asarray(seg, dtype=float))
            line_levels.append(float(level_val))

    return lines, line_levels


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_contour_preview_png(path: Path, xs, rhos, psi_xr, levels, lines_resampled):
    """Save a quick diagnostic image of psi field and extracted 2D contours."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    # psi_xr is [x, rho], contour expects [rho, x] for Z shape.
    ax.contourf(xs, rhos, psi_xr.T, levels=36, cmap="viridis")
    ax.contour(xs, rhos, psi_xr.T, levels=levels, colors="white", linewidths=0.5, alpha=0.35)

    for line in lines_resampled:
        if len(line) < 2:
            continue
        ax.plot(line[:, 0], line[:, 1], color="#ff8c00", linewidth=1.1, alpha=0.9)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("rho (m)")
    ax.set_title("Psi Contours in x-rho Plane")
    ax.set_xlim(float(xs[0]), float(xs[-1]))
    ax.set_ylim(float(rhos[0]), float(rhos[-1]))
    ax.set_aspect("auto")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Export analytic Psi contours in x-rho plane for cylindrical revolution in Three.js."
    )
    parser.add_argument("--radius", type=float, default=0.015, help="Solenoid radius in meters.")
    parser.add_argument("--length", type=float, default=0.05, help="Solenoid length in meters.")
    parser.add_argument("--turns", type=int, default=100, help="Number of loops for finite-solenoid approximation.")
    parser.add_argument("--current", type=float, default=1.0, help="Current scaling factor.")

    parser.add_argument("--extent-x", type=float, default=0.14, help="Half-extent along x axis (m).")
    parser.add_argument("--extent-rho", type=float, default=0.14, help="Extent along rho >= 0 (m).")
    parser.add_argument("--nx", type=int, default=420, help="Grid samples along x.")
    parser.add_argument("--nrho", type=int, default=320, help="Grid samples along rho.")
    parser.add_argument("--rho-min", type=float, default=1e-5, help="Minimum rho to avoid axis singular behavior.")

    parser.add_argument("--contour-levels", type=int, default=44, help="Number of psi contour levels to extract.")
    parser.add_argument("--psi-level-qmin", type=float, default=0.08, help="Lower quantile for contour level range.")
    parser.add_argument("--psi-level-qmax", type=float, default=0.96, help="Upper quantile for contour level range.")
    parser.add_argument("--contour-min-points", type=int, default=18, help="Discard contour segments shorter than this point count.")
    parser.add_argument("--contour-min-rho", type=float, default=0.001, help="Discard contour segments entirely below this rho (m).")

    parser.add_argument("--resample-ds", type=float, default=0.0025, help="Resampled spacing for contour polylines (m).")
    parser.add_argument("--theta-suggest-samples", type=int, default=96, help="Suggested angular samples for Three.js cylindrical revolution.")

    parser.add_argument(
        "--out-json",
        default="simulatedFields/exports/solenoid_psi_contours_2d.json",
        help="Output JSON path for 2D contour data.",
    )
    parser.add_argument(
        "--save-psi-npz",
        default="",
        help="Optional NPZ output path with x/rho grids and psi array.",
    )
    parser.add_argument(
        "--plot-png",
        default="simulatedFields/exports/solenoid_psi_contours_2d_preview.png",
        help="Optional PNG path to save a diagnostic x-rho contour plot; pass empty string to disable.",
    )

    args = parser.parse_args()

    params = SolenoidParams(
        radius_m=float(args.radius),
        length_m=float(args.length),
        turns=int(args.turns),
        current_a=float(args.current),
    )

    nx = max(16, int(args.nx))
    nrho = max(16, int(args.nrho))
    xs = np.linspace(-float(args.extent_x), float(args.extent_x), nx)
    rho_min = max(1e-8, float(args.rho_min))
    rhos = np.linspace(rho_min, float(args.extent_rho), nrho)
    xg, rg = np.meshgrid(xs, rhos, indexing="ij")

    psi = psi_solenoid(xg, rg, params)
    br_xr, bx_xr, bmag_xr = field_strength_xrho(xg, rg, params)

    if args.save_psi_npz:
        out_npz = Path(args.save_psi_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_npz,
            x=xs,
            rho=rhos,
            psi=psi,
            br=br_xr,
            bx=bx_xr,
            bmag=bmag_xr,
            radius=params.radius_m,
            length=params.length_m,
            turns=params.turns,
            current=params.current_a,
        )
        print(f"Saved psi grid to {out_npz}")

    psi_vals = psi[np.isfinite(psi)]
    if len(psi_vals) < 10:
        raise RuntimeError("Insufficient valid psi samples to extract contours.")

    qmin = float(np.clip(args.psi_level_qmin, 0.0, 1.0))
    qmax = float(np.clip(args.psi_level_qmax, 0.0, 1.0))
    if qmax <= qmin:
        qmax = min(1.0, qmin + 0.2)

    lo = float(np.quantile(psi_vals, qmin))
    hi = float(np.quantile(psi_vals, qmax))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(psi_vals))
        hi = float(np.max(psi_vals))
    if hi <= lo:
        raise RuntimeError("Degenerate psi range; cannot define contour levels.")

    n_levels = max(4, int(args.contour_levels))
    levels = np.linspace(lo, hi, n_levels)

    raw_lines, raw_levels = extract_psi_contours(
        xs=xs,
        rhos=rhos,
        psi_xr=psi,
        levels=levels,
        min_points=max(2, int(args.contour_min_points)),
        min_rho_keep=max(0.0, float(args.contour_min_rho)),
    )

    resampled_lines = []
    resampled_levels = []
    for line, lv in zip(raw_lines, raw_levels):
        line_rs = resample_polyline_2d(line, ds_m=float(args.resample_ds))
        if len(line_rs) < 2:
            continue
        resampled_lines.append(line_rs)
        resampled_levels.append(float(lv))

    packed_raw = flatten_lines_2d(raw_lines)
    packed_rs = flatten_lines_2d(resampled_lines)
    raw_bmag_lines = sample_bmag_on_lines(raw_lines, xs, rhos, bmag_xr)
    rs_bmag_lines = sample_bmag_on_lines(resampled_lines, xs, rhos, bmag_xr)
    packed_raw_bmag = flatten_scalar_lines(raw_bmag_lines)
    packed_rs_bmag = flatten_scalar_lines(rs_bmag_lines)
    raw_lengths = [polyline_length_2d(line) for line in raw_lines]
    rs_lengths = [polyline_length_2d(line) for line in resampled_lines]

    bmag_vals = bmag_xr[np.isfinite(bmag_xr)]

    payload = {
        "meta": {
            "schema": "openmagmapper.psiContours2d.v1",
            "units": "meters",
            "description": "2D psi contours in (x, rho) for cylindrical revolution around +X axis.",
            "field_strength_units": "relative",
        },
        "solenoid": {
            "radius_m": params.radius_m,
            "length_m": params.length_m,
            "turns": params.turns,
            "current_a": params.current_a,
        },
        "frame": {
            "axis": "x",
            "radial_symbol": "rho",
            "revolve_about": "x",
            "threejs_mapping_hint": "x->x, rho->sqrt(y*y+z*z), theta around +X",
            "theta_suggest_samples": max(8, int(args.theta_suggest_samples)),
        },
        "grid": {
            "nx": nx,
            "nrho": nrho,
            "x_min_m": float(xs[0]),
            "x_max_m": float(xs[-1]),
            "rho_min_m": float(rhos[0]),
            "rho_max_m": float(rhos[-1]),
        },
        "psi_levels": {
            "requested_count": n_levels,
            "qmin": qmin,
            "qmax": qmax,
            "min_value": lo,
            "max_value": hi,
        },
        "contours2d": {
            "count": len(raw_lines),
            "raw": {
                "levels": raw_levels,
                "lengths_m": raw_lengths,
                "bmag_values": packed_raw_bmag["values"],
                **packed_raw,
            },
            "resampled": {
                "sample_step_m": float(args.resample_ds),
                "levels": resampled_levels,
                "lengths_m": rs_lengths,
                "bmag_values": packed_rs_bmag["values"],
                **packed_rs,
            },
        },
        "psi_stats": {
            "psi_min": float(np.min(psi_vals)),
            "psi_max": float(np.max(psi_vals)),
            "psi_mean": float(np.mean(psi_vals)),
        },
        "field_stats": {
            "bmag_min": float(np.min(bmag_vals)),
            "bmag_max": float(np.max(bmag_vals)),
            "bmag_mean": float(np.mean(bmag_vals)),
        },
    }

    out_json = Path(args.out_json)
    write_json(out_json, payload)

    if args.plot_png:
        plot_path = Path(args.plot_png)
        save_contour_preview_png(
            path=plot_path,
            xs=xs,
            rhos=rhos,
            psi_xr=psi,
            levels=levels,
            lines_resampled=resampled_lines,
        )
        print(f"Saved contour preview PNG to {plot_path}")

    print(f"Saved psi-contour JSON to {out_json}")
    print(
        "Export summary: "
        f"raw_contours={len(raw_lines)}, "
        f"resampled_contours={len(resampled_lines)}, "
        f"raw_points={packed_raw['offsets'][-1] if packed_raw['offsets'] else 0}, "
        f"resampled_points={packed_rs['offsets'][-1] if packed_rs['offsets'] else 0}, "
        f"psi_range=[{np.min(psi_vals):.3e}, {np.max(psi_vals):.3e}]"
    )


if __name__ == "__main__":
    main()
