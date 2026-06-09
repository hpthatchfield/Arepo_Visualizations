#!/usr/bin/env python3
"""Render a fly-through PNG sequence from snapshot HDF5 files.

Progress is printed to stdout with ``flush=True`` so lines appear immediately
when run under SLURM, SSH, or piped to a log. Use ``python -u`` if output still
looks buffered. Typical phases:

1. Startup banner (snap range, camera path, output dir)
2. Per-snapshot load lines with load time and particle count
3. Rolling frame progress (``--progress-every N``) with elapsed time and ETA
4. Final ``done in ...`` summary

Encode frames with ffmpeg, e.g.::

    ffmpeg -y -framerate 24 -i flythrough_frames/frame_%04d.png \\
        -c:v libx264 -pix_fmt yuv420p flythrough.mp4
"""

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
CODE_TIME_TO_MYR = 98.7


def _log(msg=""):
    print(msg, flush=True)


def format_progress_line(*, done, total, frame_index, snap_number, elapsed_s):
    """One-line progress summary for a rendered frame."""
    pct = 100.0 * done / total if total else 0.0
    eta_s = elapsed_s / done * (total - done) if done > 0 else 0.0
    return (
        f"[{done}/{total} {pct:5.1f}%]  frame {frame_index:04d}  snap {snap_number}  "
        f"elapsed={elapsed_s / 60:.1f}m  eta={eta_s / 60:.1f}m"
    )


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


# (fraction, radius, azimuth_deg, elevation_deg); elevation is the angle above the
# disk plane (0 = edge-on / l-b-like, +90 = top-down, negative = below the plane).
DEFAULT_CINEMATIC_KEYFRAMES = (
    (0.00, 18.0, 0.0, 20.0),     # start: slight angle, far out
    (0.25, 8.0, 90.0, 20.0),     # slow zoom-in toward the GC
    (0.55, 8.0, 270.0, 75.0),    # orbit around, rising to near top-down
    (0.80, 10.0, 360.0, 0.0),    # dip down to edge-on (l-b view)
    (1.00, 18.0, 360.0, -30.0),  # continue below while zooming back out
)


def _smooth_interp(t, fracs, values):
    """Piecewise interpolation with smoothstep easing in/out at each keyframe."""
    t = np.asarray(t, dtype=np.float64)
    fracs = np.asarray(fracs, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(t)
    for j in range(len(fracs) - 1):
        lo, hi = fracs[j], fracs[j + 1]
        if j == len(fracs) - 2:
            m = (t >= lo) & (t <= hi)
        else:
            m = (t >= lo) & (t < hi)
        span = hi - lo
        u = (t[m] - lo) / span if span > 0 else np.zeros(int(np.count_nonzero(m)))
        s = u * u * (3.0 - 2.0 * u)
        out[m] = values[j] + (values[j + 1] - values[j]) * s
    return out


def _camera_up(pos):
    """Galactic-north-ish up vector that stays valid near (but not at) the poles."""
    pos = np.asarray(pos, dtype=np.float64)
    norm = np.linalg.norm(pos)
    if norm == 0:
        return np.array([0.0, 0.0, 1.0])
    forward = -pos / norm
    world_z = np.array([0.0, 0.0, 1.0])
    up = world_z - np.dot(world_z, forward) * forward
    if np.linalg.norm(up) < 1e-3:
        alt = np.array([0.0, 1.0, 0.0])
        up = alt - np.dot(alt, forward) * forward
    return up / np.linalg.norm(up)


def cinematic_camera_path(n_frames, keyframes=DEFAULT_CINEMATIC_KEYFRAMES):
    """Multi-phase keyframed path: zoom-in, orbit from above, dip to edge-on, exit below.

    Each keyframe is ``(fraction, radius, azimuth_deg, elevation_deg)``. Returns
    ``(positions (N, 3), ups (N, 3))`` with the camera always facing the origin.
    """
    keyframes = np.asarray(keyframes, dtype=np.float64)
    fracs = keyframes[:, 0]
    t = np.linspace(0.0, 1.0, n_frames)
    r = _smooth_interp(t, fracs, keyframes[:, 1])
    az = np.radians(_smooth_interp(t, fracs, keyframes[:, 2]))
    el = np.radians(_smooth_interp(t, fracs, keyframes[:, 3]))
    pts = np.column_stack([
        r * np.cos(el) * np.cos(az),
        r * np.cos(el) * np.sin(az),
        r * np.sin(el),
    ])
    ups = np.array([_camera_up(p) for p in pts])
    return pts, ups


def project_mass_map(x, y, z, masses, cam, smooth=True, up=(0.0, 0.0, 1.0)):
    sigma, _ = project_surface_density_camera(
        x,
        y,
        z,
        masses,
        camera_position=cam,
        target=(0.0, 0.0, 0.0),
        up_hint=up,
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


def write_png(sigma, out_path, vmin, vmax, title=None, time_myr=None):
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
    if time_myr is not None:
        ax.text(0.02, 0.98, f"{time_myr:.1f} Myr", transform=ax.transAxes,
                color="white", ha="left", va="top", fontsize=12)
    if title:
        ax.set_title(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_parser():
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
    parser.add_argument(
        "--path",
        choices=("orbit", "cinematic"),
        default="orbit",
        help="camera path: 'orbit' (tilted circle, default) or 'cinematic' "
        "(keyframed zoom-in / orbit from above / dip to edge-on / exit below); "
        "cinematic ignores --r-start/--r-end/--n-turns/--tilt-deg",
    )
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
        default=True,
        help="autoscale vmin/vmax from the first rendered frame, then hold fixed "
        "(on by default); pass --no-lock-color-scale to use the fixed --vmin/--vmax",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="print a progress line every N rendered frames (default: 1)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    snap_paths = list_snaps(
        args.snap_dir,
        prefix=args.snap_prefix,
        first_snap_number=args.first_snap_number,
        last_snap_number=args.last_snap_number,
    )
    frame_end = args.n_frames if args.frame_end is None else args.frame_end
    if frame_end <= args.frame_start:
        _log("ERROR: frame-end must be > frame-start")
        sys.exit(1)
    if args.progress_every < 1:
        _log("ERROR: --progress-every must be >= 1")
        sys.exit(1)
    n_render = frame_end - args.frame_start
    if n_render > len(snap_paths) * args.frames_per_snap:
        _log(
            f"warning: {n_render} frames need more snaps than available "
            f"({len(snap_paths)} after filter); last snap will repeat"
        )

    if args.path == "cinematic":
        cameras, ups = cinematic_camera_path(args.n_frames)
        _log("camera path: cinematic (keyframed); "
             "--r-start/--r-end/--n-turns/--tilt-deg ignored")
    else:
        cameras = camera_path(
            args.n_frames,
            r_start=args.r_start,
            r_end=args.r_end,
            n_turns=args.n_turns,
            tilt_deg=args.tilt_deg,
        )
        ups = None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    snap_lo = snap_num_from_name(snap_paths[0], args.snap_prefix)
    snap_hi = snap_num_from_name(snap_paths[-1], args.snap_prefix)
    _log("=== render_flythrough_movie ===")
    _log(f"{len(snap_paths)} snaps after filter ({snap_lo} .. {snap_hi})")
    _log(f"camera path length: {args.n_frames}  render frames {args.frame_start} .. {frame_end - 1}")
    _log(f"frames_per_snap={args.frames_per_snap}  progress_every={args.progress_every}")
    _log(f"output -> {args.output_dir.resolve()}")
    _log("starting render loop...")

    cached = None
    x = y = z = masses = None
    header = None
    vmin = args.vmin
    vmax = args.vmax
    scale_locked = False
    t0 = time.time()

    for i in range(args.frame_start, frame_end):
        sidx = min(i // args.frames_per_snap, len(snap_paths) - 1)
        snap_n = snap_num_from_name(snap_paths[sidx], args.snap_prefix)
        if sidx != cached:
            snap_path = snap_paths[sidx]
            _log(f"\nloading snap {snap_n}: {snap_path.name}")
            t_load = time.time()
            x, y, z, masses, header = load_gas_bar(snap_path)
            _log(
                f"  loaded in {time.time() - t_load:.1f}s  "
                f"N_gas = {x.size:,}  Time = {header['Time']:.6g}"
            )
            cached = sidx

        cam = cameras[i]
        up = (0.0, 0.0, 1.0) if ups is None else ups[i]
        sigma = project_mass_map(x, y, z, masses, cam, up=up)
        if args.lock_color_scale and not scale_locked:
            vmin, vmax = color_limits(sigma)
            scale_locked = True
            _log(f"locked color scale from frame {i}: vmin={vmin:.4g}  vmax={vmax:.4g}")

        out_path = args.output_dir / f"frame_{i:04d}.png"
        title = f"frame {i:04d}  snap {snap_n}"
        write_png(sigma, out_path, vmin, vmax, title=title,
                  time_myr=float(header["Time"]) * CODE_TIME_TO_MYR)

        done = i - args.frame_start + 1
        is_last = i == frame_end - 1
        if done % args.progress_every == 0 or is_last:
            elapsed = time.time() - t0
            _log(format_progress_line(
                done=done,
                total=n_render,
                frame_index=i,
                snap_number=snap_n,
                elapsed_s=elapsed,
            ))

    _log(f"\ndone: {n_render} frames in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
