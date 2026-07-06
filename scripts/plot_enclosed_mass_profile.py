#!/usr/bin/env python3
"""Cumulative enclosed gas mass vs radius for one snapshot.

Loads gas cell masses, measures radius from the GC in the bar frame, and plots
M(<R) using only cells with min_r <= r <= max_r (defaults: 0 and box/2).
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PKG_ROOT))

from simviz.projections import rotate_to_bar_frame
from simviz.utils import CONSTANTS, read_snapshot_hdf5

SNAP_PREFIX = "phoenix_stinks_1Msun"
CODE_LENGTH_TO_KPC = CONSTANTS["arepoLength"] / CONSTANTS["pc2cm"] / 1.0e3


def cell_radius(x, y, z, metric="cylindrical"):
    """Radius of each cell from the origin (code units)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if metric == "cylindrical":
        return np.hypot(x, y)
    if metric == "spherical":
        return np.sqrt(x * x + y * y + z * z)
    raise ValueError(f"metric must be 'cylindrical' or 'spherical', got {metric!r}")


def filter_by_radius_range(radius, masses, min_r, max_r):
    """Keep only cells with min_r <= r <= max_r (code units)."""
    radius = np.asarray(radius, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    mask = (radius >= min_r) & (radius <= max_r)
    return radius[mask], masses[mask]


def cumulative_enclosed_mass(radius, masses):
    """Return sorted radii and M(<r) in the same mass units as ``masses``."""
    radius = np.asarray(radius, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if radius.shape != masses.shape:
        raise ValueError("radius and masses must have the same shape")
    finite = np.isfinite(radius) & np.isfinite(masses) & (masses > 0)
    radius = radius[finite]
    masses = masses[finite]
    if radius.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(radius, kind="mergesort")
    r_sorted = radius[order]
    m_sorted = masses[order]
    return r_sorted, np.cumsum(m_sorted)


def thin_curve(radius, enclosed, max_points=4000):
    """Downsample a monotonic curve for plotting."""
    n = radius.size
    if n <= max_points:
        return radius, enclosed
    idx = np.unique(np.linspace(0, n - 1, max_points, dtype=int))
    return radius[idx], enclosed[idx]


def load_gc_mass_profile(snap_path, metric="cylindrical"):
    """Load one snap and return radii (code units) and cell masses (M☉)."""
    data, header = read_snapshot_hdf5(snap_path, fields=("Coordinates", "Masses"))
    box = float(header["BoxSize"])
    t = float(header["Time"])
    x, y, z = np.asarray(data["Coordinates"], dtype=np.float64).T
    masses = np.asarray(data["Masses"], dtype=np.float64)
    x -= box / 2.0
    y -= box / 2.0
    z -= box / 2.0
    z0 = np.zeros_like(x)
    x, y, _, _ = rotate_to_bar_frame(x, y, z0, z0, t)
    radius = cell_radius(x, y, z, metric=metric)
    return radius, masses, header


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot cumulative enclosed gas mass M(<R) vs radius for one snapshot.",
    )
    parser.add_argument("-s", "--snap", type=Path, help="snapshot HDF5 path")
    parser.add_argument("--snap-dir", type=Path, help="directory with prefix_N.hdf5 files")
    parser.add_argument("--snap-number", type=int, help="snap index (with --snap-dir)")
    parser.add_argument("--snap-prefix", default=SNAP_PREFIX)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_PKG_ROOT / "example_output" / "enclosed_mass_profile.png",
    )
    parser.add_argument(
        "--metric",
        choices=("cylindrical", "spherical"),
        default="cylindrical",
        help="cylindrical: R=sqrt(x^2+y^2); spherical: r=|r| (bar frame, GC-centered)",
    )
    parser.add_argument(
        "--radius-unit",
        choices=("kpc", "code", "pc"),
        default="kpc",
        help="units for --min-r, --max-r, and the plot x-axis",
    )
    parser.add_argument(
        "--min-r",
        type=float,
        default=0.0,
        help="inner radius limit (same units as --radius-unit); default 0",
    )
    parser.add_argument(
        "--max-r",
        type=float,
        default=None,
        help="outer radius limit (same units as --radius-unit); default box/2",
    )
    parser.add_argument("--log-x", action="store_true", help="log radius axis")
    parser.add_argument("--log-y", action="store_true", help="log enclosed-mass axis")
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=4000,
        help="downsample curve for plotting (full snap still used in sum)",
    )
    return parser


def resolve_snap_path(args):
    if args.snap is not None:
        return Path(args.snap)
    if args.snap_dir is None or args.snap_number is None:
        print("ERROR: pass --snap or (--snap-dir and --snap-number)")
        sys.exit(1)
    path = args.snap_dir / f"{args.snap_prefix}_{args.snap_number}.hdf5"
    if not path.is_file():
        print(f"ERROR: not found: {path}")
        sys.exit(1)
    return path


def radius_scale_and_label(unit):
    if unit == "code":
        return 1.0, r"$R$ [100 pc]"
    if unit == "pc":
        return CONSTANTS["arepoLength"] / CONSTANTS["pc2cm"], r"$R$ [pc]"
    return CODE_LENGTH_TO_KPC, r"$R$ [kpc]"


def radius_to_code(value, unit):
    scale, _ = radius_scale_and_label(unit)
    return float(value) / scale


def radius_from_code(value, unit):
    scale, _ = radius_scale_and_label(unit)
    return float(value) * scale


def main():
    parser = build_parser()
    args = parser.parse_args()

    snap_path = resolve_snap_path(args)
    print(f"loading {snap_path}")
    radius_code, masses, header = load_gc_mass_profile(snap_path, metric=args.metric)
    box = float(header["BoxSize"])
    min_r_code = radius_to_code(args.min_r, args.radius_unit)
    if args.max_r is None:
        max_r_code = box / 2.0
    else:
        max_r_code = radius_to_code(args.max_r, args.radius_unit)
    if min_r_code < 0 or max_r_code <= min_r_code:
        print(f"ERROR: need 0 <= min_r < max_r, got min={min_r_code} max={max_r_code} (code units)")
        sys.exit(1)

    radius_code, masses = filter_by_radius_range(radius_code, masses, min_r_code, max_r_code)
    r_sorted, m_enc = cumulative_enclosed_mass(radius_code, masses)
    if r_sorted.size == 0:
        print(
            f"ERROR: no finite gas cells with positive mass in "
            f"[{min_r_code:.4g}, {max_r_code:.4g}] code units"
        )
        sys.exit(1)

    r_plot, m_plot = thin_curve(r_sorted, m_enc, max_points=args.max_plot_points)
    scale, xlabel = radius_scale_and_label(args.radius_unit)
    r_plot = r_plot * scale
    min_r_plot = radius_from_code(min_r_code, args.radius_unit)
    max_r_plot = radius_from_code(max_r_code, args.radius_unit)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    ax.plot(r_plot, m_plot, color="0.15", lw=1.2)
    ax.set_xlim(min_r_plot, max_r_plot)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Enclosed mass $M(<R)$ [$M_\odot$]")
    snap_n = snap_path.stem.split("_")[-1]
    time_myr = float(header["Time"]) * 98.7
    ax.set_title(
        f"snap {snap_n}  $t={time_myr:.1f}$ Myr  ({args.metric})\n"
        f"${min_r_plot:.3g} \\leq R \\leq {max_r_plot:.3g}$ "
        f"{args.radius_unit}"
    )
    if args.log_x:
        ax.set_xscale("log")
    if args.log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    print(f"  N_gas in annulus = {radius_code.size:,}")
    print(f"  R range = [{min_r_plot:.3g}, {max_r_plot:.3g}] {args.radius_unit}")
    print(f"  M_enc(max_r) = {m_enc[-1]:.4g} M_sun")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
