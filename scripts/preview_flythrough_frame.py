#!/usr/bin/env python3
"""one flythrough preview frame"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PKG_ROOT))

from simviz.field_plots import project_surface_density_camera
from simviz.projections import rotate_about_axis, rotate_to_bar_frame
from simviz.utils import read_snapshot_hdf5

# tune these
SNAP_PREFIX = "phoenix_stinks_1Msun"
FOV_X_DEG = 55.0
NX, NY = 700, 700
Z_NEAR, Z_FAR = 0.2, 30.0
VMIN, VMAX = 5e-4, 2e1
MASKED_FILL_SIGMAS = (0.8, 12.0)
MASKED_FILL_WEIGHT_SIGMA_PX = 1.5
MASKED_FILL_PERCENTILES = (30.0, 88.0)
MASKED_FILL_MASK_POWER = 2.0


def snap_num_from_name(path, prefix=SNAP_PREFIX):
    m = re.search(rf"{re.escape(prefix)}_(\d+)", Path(path).name, re.I)
    if not m:
        raise ValueError(f"expected {prefix}_<N>.hdf5, got {Path(path).name}")
    return int(m.group(1))


def load_gas_bar(snap_path):
    data, header = read_snapshot_hdf5(snap_path, fields=("Coordinates", "Masses"))
    t = float(header["Time"])
    box = float(header["BoxSize"])
    x, y, z = np.asarray(data["Coordinates"], dtype=np.float64).T
    masses = np.asarray(data["Masses"], dtype=np.float64)
    x -= box / 2.0
    y -= box / 2.0
    z -= box / 2.0
    z0 = np.zeros_like(x)
    x, y, _, _ = rotate_to_bar_frame(x, y, z0, z0, t)
    return x, y, z, masses, header


def camera_path(n_frames, r_start=12.0, r_end=6.0, n_turns=1.5, tilt_deg=35.0):
    theta = np.linspace(0, 2 * np.pi * n_turns, n_frames, endpoint=False)
    radius = np.linspace(r_start, r_end, n_frames)
    pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros(n_frames)])
    pts = rotate_about_axis(pts, axis=(1.0, 1.0, 0.3), angle=np.radians(tilt_deg))
    return pts


def project_mass_map(x, y, z, masses, cam, smooth=True):
    sigma, _ = project_surface_density_camera(
        x,
        y,
        z,
        masses,
        camera_position=cam,
        target=(0.0, 0.0, 0.0),
        up_hint=(0.0, 0.0, 1.0),
        fov_x_deg=FOV_X_DEG,
        nx=NX,
        ny=NY,
        z_near=Z_NEAR,
        z_far=Z_FAR,
        masked_fill_sigmas=MASKED_FILL_SIGMAS if smooth else None,
        masked_fill_weight_sigma_px=MASKED_FILL_WEIGHT_SIGMA_PX,
        masked_fill_percentiles=MASKED_FILL_PERCENTILES,
        masked_fill_mask_power=MASKED_FILL_MASK_POWER,
    )
    return sigma


def describe_map(sigma, label):
    """Print stage stats useful for diagnosing washed-out frames."""
    arr = np.asarray(sigma, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    pos = finite[finite > 0]
    nonzero_frac = pos.size / arr.size if arr.size else 0.0
    print(f"[{label}]")
    print(f"  shape={arr.shape}  nonzero_frac={nonzero_frac:.3f}")
    if pos.size:
        pcts = np.percentile(pos, [1, 50, 99, 99.5])
        print(
            f"  positive: min={pos.min():.3g}  max={pos.max():.3g}  "
            f"p1={pcts[0]:.3g}  p50={pcts[1]:.3g}  p99={pcts[2]:.3g}  p99.5={pcts[3]:.3g}"
        )
    else:
        print("  no positive pixels")


def color_limits(sigma, vmin_floor=1e-6):
    pos = sigma[sigma > 0]
    if pos.size == 0:
        return VMIN, VMAX
    vmin = max(float(np.percentile(pos, 1)), vmin_floor)
    vmax = float(np.percentile(pos, 99.5))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return vmin, vmax


def write_png(sigma, out_path, vmin, vmax, title=None):
    out = np.asarray(sigma, dtype=np.float64).copy()
    out[~np.isfinite(out)] = vmin
    out[out <= 0] = vmin

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    ax.imshow(
        out,
        origin="lower",
        extent=(-1, 1, -1, 1),
        norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        cmap="inferno",
        interpolation="nearest",
    )
    ax.set_xlabel("camera x")
    ax.set_ylabel("camera y")
    if title:
        ax.set_title(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_compare_png(sigma_raw, sigma_smoothed, out_path, vmin, vmax):
    """Side-by-side raw vs smoothed map at the same color scale."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    for ax, data, label in (
        (axes[0], sigma_raw, "raw histogram"),
        (axes[1], sigma_smoothed, "masked_fill"),
    ):
        out = np.asarray(data, dtype=np.float64).copy()
        out[~np.isfinite(out)] = vmin
        out[out <= 0] = vmin
        ax.imshow(
            out,
            origin="lower",
            extent=(-1, 1, -1, 1),
            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            cmap="inferno",
            interpolation="nearest",
        )
        ax.set_title(label)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--snap", type=Path, help="one snapshot hdf5")
    parser.add_argument("--snap-dir", type=Path, help="dir with prefix_N.hdf5 files")
    parser.add_argument("--snap-number", type=int)
    parser.add_argument("--snap-prefix", default=SNAP_PREFIX)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--n-frames", type=int, default=300)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--r-start", type=float, default=12.0)
    parser.add_argument("--r-end", type=float, default=6.0)
    parser.add_argument("--n-turns", type=float, default=1.5)
    parser.add_argument("--tilt-deg", type=float, default=35.0)
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--auto-scale", action="store_true")
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="render the raw histogram (skip masked_fill) to isolate the smoothing stage",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print stage-by-stage map stats and write a raw-vs-smoothed comparison PNG",
    )
    args = parser.parse_args()

    if args.snap is not None:
        snap_path = args.snap
    else:
        if args.snap_dir is None or args.snap_number is None:
            print("ERROR: need --snap or (--snap-dir and --snap-number)")
            sys.exit(1)
        snap_path = args.snap_dir / f"{args.snap_prefix}_{args.snap_number}.hdf5"

    if not snap_path.is_file():
        print(f"ERROR: not found: {snap_path}")
        sys.exit(1)

    out_path = args.output
    if out_path is None:
        out_path = _PKG_ROOT / "example_output" / "flythrough_preview.png"

    print(f"loading {snap_path}")
    x, y, z, masses, header = load_gas_bar(snap_path)
    print(f"  N_gas = {x.size:,}  Time = {header['Time']:.6g}")

    path = camera_path(
        args.n_frames,
        r_start=args.r_start,
        r_end=args.r_end,
        n_turns=args.n_turns,
        tilt_deg=args.tilt_deg,
    )
    cam = path[args.frame_index % args.n_frames]

    print("projecting...")
    sigma = project_mass_map(x, y, z, masses, cam, smooth=not args.no_smooth)

    if args.debug:
        sigma_raw = project_mass_map(x, y, z, masses, cam, smooth=False)
        describe_map(sigma_raw, "raw histogram")
        describe_map(sigma, "after masked_fill" if not args.no_smooth else "rendered (no smooth)")

    if args.auto_scale:
        vmin, vmax = color_limits(sigma)
        print(f"auto color scale: vmin={vmin:.4g}  vmax={vmax:.4g}")
    else:
        vmin = VMIN if args.vmin is None else args.vmin
        vmax = VMAX if args.vmax is None else args.vmax

    snap_n = snap_num_from_name(snap_path, prefix=args.snap_prefix)
    title = f"snap {snap_n}  frame {args.frame_index}  cam=({cam[0]:.2f},{cam[1]:.2f},{cam[2]:.2f})"
    write_png(sigma, out_path, vmin, vmax, title=title)
    print(f"wrote {out_path}")

    if args.debug and not args.no_smooth:
        cmp_path = Path(out_path).with_name(Path(out_path).stem + "_compare.png")
        _write_compare_png(sigma_raw, sigma, cmp_path, vmin, vmax)
        print(f"wrote {cmp_path}")


if __name__ == "__main__":
    main()
