#!/usr/bin/env python3
"""flythrough png sequence from snapshot hdf5 files"""

import argparse
import re
import sys
import time
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

# tune these (match preview_flythrough_frame.py)
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


def list_snaps(snap_dir, prefix=SNAP_PREFIX, first_snap_number=None, last_snap_number=None):
    paths = sorted(snap_dir.glob(f"{prefix}_*.hdf5"), key=lambda p: snap_num_from_name(p, prefix))
    if first_snap_number is not None:
        paths = [p for p in paths if snap_num_from_name(p, prefix) >= first_snap_number]
    if last_snap_number is not None:
        paths = [p for p in paths if snap_num_from_name(p, prefix) <= last_snap_number]
    if not paths:
        print(f"ERROR: no {prefix}_*.hdf5 in {snap_dir} (check --first-snap-number / --last-snap-number)")
        sys.exit(1)
    return paths


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


def project_mass_map(x, y, z, masses, cam):
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
        masked_fill_sigmas=MASKED_FILL_SIGMAS,
        masked_fill_weight_sigma_px=MASKED_FILL_WEIGHT_SIGMA_PX,
        masked_fill_percentiles=MASKED_FILL_PERCENTILES,
        masked_fill_mask_power=MASKED_FILL_MASK_POWER,
    )
    return sigma


def color_limits(sigma, vmin_floor=1e-6):
    """Log-scale limits from positive pixels (same as preview_flythrough_frame)."""
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
    if title:
        ax.set_title(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snap-dir", type=Path, required=True)
    parser.add_argument("--snap-prefix", default=SNAP_PREFIX)
    parser.add_argument(
        "--first-snap-number",
        type=int,
        default=None,
        help="only use snaps with this number or higher in the filename",
    )
    parser.add_argument(
        "--last-snap-number",
        type=int,
        default=None,
        help="only use snaps with this number or lower in the filename",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("flythrough_frames"))
    parser.add_argument("--n-frames", type=int, default=300)
    parser.add_argument("--frames-per-snap", type=int, default=5)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--r-start", type=float, default=12.0)
    parser.add_argument("--r-end", type=float, default=6.0)
    parser.add_argument("--n-turns", type=float, default=1.5)
    parser.add_argument("--tilt-deg", type=float, default=35.0)
    parser.add_argument("--vmin", type=float, default=VMIN)
    parser.add_argument("--vmax", type=float, default=VMAX)
    parser.add_argument(
        "--lock-color-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="autoscale vmin/vmax from the first rendered frame, then hold fixed; "
        "off by default so frames use the fixed --vmin/--vmax (matches preview defaults)",
    )
    args = parser.parse_args()

    snap_paths = list_snaps(
        args.snap_dir,
        prefix=args.snap_prefix,
        first_snap_number=args.first_snap_number,
        last_snap_number=args.last_snap_number,
    )
    frame_end = args.n_frames if args.frame_end is None else args.frame_end
    if frame_end <= args.frame_start:
        print("ERROR: frame-end must be > frame-start")
        sys.exit(1)
    n_render = frame_end - args.frame_start
    if n_render > len(snap_paths) * args.frames_per_snap:
        print(
            f"warning: {n_render} frames need more snaps than available "
            f"({len(snap_paths)} after filter); last snap will repeat"
        )

    cameras = camera_path(
        args.n_frames,
        r_start=args.r_start,
        r_end=args.r_end,
        n_turns=args.n_turns,
        tilt_deg=args.tilt_deg,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    snap_lo = snap_num_from_name(snap_paths[0], args.snap_prefix)
    snap_hi = snap_num_from_name(snap_paths[-1], args.snap_prefix)
    print(f"{len(snap_paths)} snaps after filter ({snap_lo} .. {snap_hi})")
    print(f"camera path length: {args.n_frames}  render frames {args.frame_start} .. {frame_end - 1}")
    print(f"output -> {args.output_dir}")

    cached = None
    x = y = z = masses = None
    vmin = args.vmin
    vmax = args.vmax
    scale_locked = False
    t0 = time.time()

    for i in range(args.frame_start, frame_end):
        sidx = min(i // args.frames_per_snap, len(snap_paths) - 1)
        if sidx != cached:
            snap_path = snap_paths[sidx]
            print(f"\nloading snap {snap_num_from_name(snap_path, args.snap_prefix)}: {snap_path.name}")
            x, y, z, masses, header = load_gas_bar(snap_path)
            print(f"  N_gas = {x.size:,}  Time = {header['Time']:.6g}")
            cached = sidx

        cam = cameras[i]
        sigma = project_mass_map(x, y, z, masses, cam)
        if args.lock_color_scale and not scale_locked:
            vmin, vmax = color_limits(sigma)
            scale_locked = True
            print(f"locked color scale from frame {i}: vmin={vmin:.4g}  vmax={vmax:.4g}")

        out_path = args.output_dir / f"frame_{i:04d}.png"
        title = f"frame {i:04d}  snap {snap_num_from_name(snap_paths[sidx], args.snap_prefix)}"
        write_png(sigma, out_path, vmin, vmax, title=title)

        if (i - args.frame_start) % 10 == 0:
            print(f"  wrote {out_path.name}")

    print(f"\ndone in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
