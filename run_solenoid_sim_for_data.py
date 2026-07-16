import argparse
import csv
import subprocess
import sys
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Launch solenoid field simulation with parameters matched to your dataset."
    )

    parser.add_argument(
        "--sim-script",
        default="simulate_solenoid_vector_field.py",
        help="Path to simulator script (default: simulate_solenoid_vector_field.py).",
    )

    # Solenoid parameters (core physical model).
    parser.add_argument("--radius-mm", type=float, default=40.0, help="Solenoid radius in mm.")
    parser.add_argument("--length-mm", type=float, default=120.0, help="Solenoid length in mm.")
    parser.add_argument("--turns", type=int, default=80, help="Number of turns.")
    parser.add_argument("--current-a", type=float, default=1.0, help="Current in A.")
    parser.add_argument("--segments-per-turn", type=int, default=24, help="Line segments per turn.")

    # Grid/sampling parameters.
    parser.add_argument("--extent-xy", type=float, default=0.12, help="Half-width extent in x/y (m).")
    parser.add_argument("--extent-z", type=float, default=0.18, help="Half-width extent in z (m).")
    parser.add_argument("--n-xy", type=int, default=13, help="Grid samples along x/y.")
    parser.add_argument("--n-z", type=int, default=15, help="Grid samples along z.")
    parser.add_argument("--quiver-step", type=int, default=2, help="Quiver subsample step.")
    parser.add_argument("--scale", type=float, default=1.0, help="Arrow scale multiplier.")

    # Data matching helpers.
    parser.add_argument(
        "--match-frozen-csv",
        default="",
        help=(
            "Optional frozen vectors CSV. If set, extents are auto-fit to the"
            " sensor_table_* coordinate bounds."
        ),
    )
    parser.add_argument(
        "--extent-margin",
        type=float,
        default=0.03,
        help="Extra margin in meters added around data bounds when auto-fitting extents.",
    )

    parser.add_argument(
        "--save-npz",
        default="",
        help="Optional .npz path to save field arrays from the simulator.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing it.",
    )

    return parser


def infer_extents_from_frozen_csv(csv_path: Path, margin_m: float):
    xs = []
    ys = []
    zs = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                xs.append(float(row["sensor_table_x_m"]))
                ys.append(float(row["sensor_table_y_m"]))
                zs.append(float(row["sensor_table_z_m"]))
            except Exception:
                continue

    if not xs or not ys or not zs:
        raise ValueError("No valid sensor_table coordinates found in frozen CSV.")

    max_abs_xy = max(
        max(abs(min(xs)), abs(max(xs))),
        max(abs(min(ys)), abs(max(ys))),
    )
    max_abs_z = max(abs(min(zs)), abs(max(zs)))

    extent_xy = max_abs_xy + margin_m
    extent_z = max_abs_z + margin_m
    return extent_xy, extent_z


def main():
    parser = build_parser()
    args = parser.parse_args()

    sim_script = Path(args.sim_script).resolve()
    if not sim_script.exists():
        raise SystemExit(f"Simulator script not found: {sim_script}")

    extent_xy = float(args.extent_xy)
    extent_z = float(args.extent_z)

    if args.match_frozen_csv:
        frozen_csv = Path(args.match_frozen_csv).resolve()
        if not frozen_csv.exists():
            raise SystemExit(f"Frozen CSV not found: {frozen_csv}")
        extent_xy, extent_z = infer_extents_from_frozen_csv(frozen_csv, float(args.extent_margin))
        print(
            f"Auto-fit extents from {frozen_csv.name}: "
            f"extent_xy={extent_xy:.4f} m, extent_z={extent_z:.4f} m"
        )

    cmd = [
        sys.executable,
        str(sim_script),
        "--radius", str(float(args.radius_mm) / 1000.0),
        "--length", str(float(args.length_mm) / 1000.0),
        "--turns", str(int(args.turns)),
        "--current", str(float(args.current_a)),
        "--segments-per-turn", str(int(args.segments_per_turn)),
        "--extent-xy", str(extent_xy),
        "--extent-z", str(extent_z),
        "--n-xy", str(int(args.n_xy)),
        "--n-z", str(int(args.n_z)),
        "--quiver-step", str(int(args.quiver_step)),
        "--scale", str(float(args.scale)),
    ]

    if args.save_npz:
        cmd.extend(["--save-npz", str(args.save_npz)])

    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    print(f"Command:\n{printable}")

    if args.dry_run:
        return 0

    result = subprocess.run(cmd)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
