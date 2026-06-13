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
    COLUMN_DEPOSIT,
    COLUMN_SMOOTH_SIGMA_PX,
    DEFAULT_CMAP,
    MASKED_FILL_BLEND_MODE,
    PATH_KEYFRAMES,
    PROJECTION_METHOD,
    PROJECTION_METHODS,
    PROJECTION_WEIGHT_FIELDS,
    SIGMA_CODE_TO_MSUN_PC2,
    SIGMA_COLORBAR_LABEL,
    SNAP_PREFIX,
    VMAX,
    VMIN,
    build_camera_path,
    color_limits,
    load_gas_bar,
    project_flythrough_map,
    snap_num_from_name,
    write_png,
)
from simviz.colormaps import resolve_cmap

project_mass_map = project_flythrough_map


def resolve_preview_output(output, *, path, frame_index, tag=None):
    """Pick a preview PNG path that avoids accidental overwrites."""
    if output is not None:
        return Path(output)
    parts = [path, f"f{frame_index:04d}"]
    if tag:
        parts.append(tag)
    stem = "flythrough_preview_" + "_".join(parts)
    return _PKG_ROOT / "example_output" / f"{stem}.png"


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


def _write_deposit_compare_png(sigma_nearest, sigma_production, out_path, vmin, vmax):
    """Side-by-side nearest vs production column deposit at one color scale."""
    disp_vmin = vmin * SIGMA_CODE_TO_MSUN_PC2
    disp_vmax = vmax * SIGMA_CODE_TO_MSUN_PC2
    prod_label = (
        f"production ({COLUMN_DEPOSIT} + "
        f"σ={COLUMN_SMOOTH_SIGMA_PX:g}px blur)"
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=150)
    last_im = None
    for ax, data, label in (
        (axes[0], sigma_nearest, "nearest (no splat, no blur)"),
        (axes[1], sigma_production, prod_label),
    ):
        out = np.asarray(data, dtype=np.float64).copy()
        out[~np.isfinite(out)] = vmin
        out[out <= 0] = vmin
        last_im = ax.imshow(
            out * SIGMA_CODE_TO_MSUN_PC2,
            origin="lower",
            extent=(-1, 1, -1, 1),
            norm=colors.LogNorm(vmin=disp_vmin, vmax=disp_vmax),
            cmap=resolve_cmap(DEFAULT_CMAP),
            interpolation="nearest",
        )
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02, label=SIGMA_COLORBAR_LABEL)
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
            cmap=resolve_cmap(DEFAULT_CMAP),
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
    parser.add_argument(
        "--tag",
        default=None,
        help="suffix for the auto-generated output name (with --path and "
        "--frame-index); ignored when -o is set",
    )
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
        "--compare-deposit",
        action="store_true",
        help="write a side-by-side PNG: nearest (no splat/blur) vs production "
        "column deposit, then exit",
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
        help="masked_fill blend for --projection-method surface only",
    )
    parser.add_argument(
        "--projection-method",
        choices=PROJECTION_METHODS,
        default=PROJECTION_METHOD,
        help="surface or column (default: depth-integrated histogram)",
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

    out_path = resolve_preview_output(
        args.output,
        path=args.path,
        frame_index=args.frame_index,
        tag=args.tag,
    )

    print(f"loading {snap_path}")
    x, y, z, weights, header = load_gas_bar(
        snap_path, projection_weight=args.projection_weight
    )
    print(f"  N_gas = {x.size:,}  Time = {header['Time']:.6g}")
    print(f"  projection_weight = {args.projection_weight}")
    print(f"  projection_method = {args.projection_method}")

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

    if args.compare_deposit:
        if args.projection_method != "column":
            print("ERROR: --compare-deposit requires --projection-method column")
            sys.exit(1)
        print("projecting nearest (no splat, no blur)...")
        sigma_nearest = project_flythrough_map(
            x, y, z, weights, cam, smooth=False, up=up,
            projection_method=args.projection_method,
        )
        print("projecting production (splat + blur)...")
        sigma_production = project_flythrough_map(
            x, y, z, weights, cam, smooth=True, up=up,
            projection_method=args.projection_method,
            smooth_blend=args.smooth_blend,
        )
        describe_map(sigma_nearest, "nearest")
        describe_map(sigma_production, "production")
        if args.auto_scale:
            vmin, vmax = color_limits(sigma_production)
            print(f"shared color scale (from production): vmin={vmin:.4g}  vmax={vmax:.4g}")
        else:
            vmin = VMIN if args.vmin is None else args.vmin
            vmax = VMAX if args.vmax is None else args.vmax
        if args.output is not None:
            cmp_path = Path(args.output)
        else:
            cmp_path = resolve_preview_output(
                None,
                path=args.path,
                frame_index=args.frame_index,
                tag=args.tag or "deposit_compare",
            )
        _write_deposit_compare_png(
            sigma_nearest, sigma_production, cmp_path, vmin, vmax,
        )
        print(f"wrote {cmp_path}")
        return

    print("projecting...")
    sigma = project_flythrough_map(
        x, y, z, weights, cam,
        smooth=not args.no_smooth, up=up,
        projection_method=args.projection_method,
        smooth_blend=args.smooth_blend,
    )

    if args.debug:
        sigma_raw = project_flythrough_map(
            x, y, z, weights, cam, smooth=False, up=up,
            projection_method=args.projection_method,
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
