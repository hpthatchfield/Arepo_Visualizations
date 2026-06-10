#!/usr/bin/env python3
"""one flythrough preview frame

Renders a single frame using the exact same parameters and code paths as
render_flythrough_movie.py, so the preview matches what the movie produces.
"""

import argparse
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

from scripts.render_flythrough_movie import (
    CODE_TIME_TO_MYR,
    MASKED_FILL_BLEND_MODE,
    PATH_KEYFRAMES,
    PROJECTION_WEIGHT_FIELDS,
    SNAP_PREFIX,
    VMAX,
    VMIN,
    build_camera_path,
    color_limits,
    load_gas_bar,
    project_surface_map,
    snap_num_from_name,
    write_png,
)

project_mass_map = project_surface_map


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


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--snap", type=Path, help="one snapshot hdf5")
    parser.add_argument("--snap-dir", type=Path, help="dir with prefix_N.hdf5 files")
    parser.add_argument("--snap-number", type=int)
    parser.add_argument("--snap-prefix", default=SNAP_PREFIX)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--n-frames", type=int, default=300)
    parser.add_argument(
        "--path",
        choices=("orbit", "cinematic", "edge-orbit", "zoom-observe"),
        default="orbit",
        help="camera path: 'orbit' (default), 'cinematic', 'edge-orbit', or "
        "'zoom-observe' (keyframed); keyframed paths ignore "
        "--r-start/--r-end/--n-turns/--tilt-deg",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--r-start", type=float, default=12.0)
    parser.add_argument("--r-end", type=float, default=6.0)
    parser.add_argument("--n-turns", type=float, default=1.5)
    parser.add_argument("--tilt-deg", type=float, default=35.0)
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument(
        "--auto-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="autoscale vmin/vmax from this frame's percentiles (on by default, "
        "matches the movie); pass --no-auto-scale to use the fixed --vmin/--vmax",
    )
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
    parser.add_argument(
        "--projection-weight",
        choices=tuple(PROJECTION_WEIGHT_FIELDS),
        default="density",
        help="histogram weight per gas cell: 'density' (default) or 'mass'",
    )
    parser.add_argument(
        "--smooth-blend",
        choices=("detail", "linear"),
        default=MASKED_FILL_BLEND_MODE,
        help="masked_fill blend: 'detail' (default) or 'linear' (legacy)",
    )
    return parser


def main():
    parser = build_parser()
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
    x, y, z, weights, header = load_gas_bar(
        snap_path, projection_weight=args.projection_weight
    )
    print(f"  N_gas = {x.size:,}  Time = {header['Time']:.6g}")
    print(f"  projection_weight = {args.projection_weight}")

    if args.path in PATH_KEYFRAMES:
        path, ups = build_camera_path(args.path, args.n_frames)
    else:
        path, ups = build_camera_path(
            args.path,
            args.n_frames,
            r_start=args.r_start,
            r_end=args.r_end,
            n_turns=args.n_turns,
            tilt_deg=args.tilt_deg,
        )
    idx = args.frame_index % args.n_frames
    cam = path[idx]
    up = (0.0, 0.0, 1.0) if ups is None else ups[idx]

    print("projecting...")
    sigma = project_surface_map(
        x, y, z, weights, cam,
        smooth=not args.no_smooth, up=up, smooth_blend=args.smooth_blend,
    )

    if args.debug:
        sigma_raw = project_surface_map(
            x, y, z, weights, cam, smooth=False, up=up,
        )
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
    write_png(sigma, out_path, vmin, vmax, title=title,
              time_myr=float(header["Time"]) * CODE_TIME_TO_MYR)
    print(f"wrote {out_path}")

    if args.debug and not args.no_smooth:
        cmp_path = Path(out_path).with_name(Path(out_path).stem + "_compare.png")
        _write_compare_png(sigma_raw, sigma, cmp_path, vmin, vmax)
        print(f"wrote {cmp_path}")


if __name__ == "__main__":
    main()
