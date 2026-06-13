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

    ffmpeg -y -framerate 24 -i example_output/flythrough_frames/frame_%04d.png \\
        -c:v libx264 -pix_fmt yuv420p example_output/flythrough.mp4
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

from simviz.field_plots import (
    project_column_density_camera,
    project_surface_density_camera,
)
from simviz.colormaps import resolve_cmap
from simviz.projections import rotate_about_axis, rotate_to_bar_frame
from simviz.utils import (
    code_density_sum_to_n_mol_cm2,
    n_mol_cm2_to_code_density_sum,
    read_snapshot_hdf5,
)

# tune these (match preview_flythrough_frame.py)
SNAP_PREFIX = "phoenix_stinks_1Msun"
FOV_X_DEG = 55.0
NX, NY = 700, 700
Z_NEAR, Z_FAR = 0.2, 30.0
VMAX = 2e1
COLOR_VMIN_PERCENTILE = 20.0
COLOR_VMAX_PERCENTILE = 99.0
MASKED_FILL_SIGMAS = (0.0, 12.0)
MASKED_FILL_WEIGHT_SIGMA_PX = 1.5
MASKED_FILL_PERCENTILES = (25.0, 90.0)
MASKED_FILL_MASK_POWER = 1.0
MASKED_FILL_BLEND_MODE = "detail"
MASKED_FILL_DENSE_THRESHOLD = 0.35
PROJECTION_METHOD = "column"
COLUMN_DEPTH_BINS = 48
COLUMN_DEPOSIT = "linear_xy"
COLUMN_SMOOTH_SIGMA_PX = 0.35
COLUMN_DEPTH_CODE = Z_FAR - Z_NEAR
# Log-scale display floor in cm⁻² (~10 M☉ pc⁻² under default sightline depth).
SIGMA_DISPLAY_FLOOR_N_MOL_CM2 = 5.4e20
VMIN_FLOOR = n_mol_cm2_to_code_density_sum(
    SIGMA_DISPLAY_FLOOR_N_MOL_CM2, COLUMN_DEPTH_CODE
)
VMIN = VMIN_FLOOR
SIGMA_CODE_TO_N_MOL_CM2 = float(
    code_density_sum_to_n_mol_cm2(1.0, COLUMN_DEPTH_CODE)
)
SIGMA_COLORBAR_LABEL = r"$N_\mathrm{mol}$ [cm$^{-2}$]"
CODE_TIME_TO_MYR = 98.7

PROJECTION_METHODS = ("surface", "column")
ARRAYS_SUBDIR = "arrays"
DEFAULT_CMAP = "pride"


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


PROJECTION_WEIGHT_FIELDS = {
    "density": "Density",
    "mass": "Masses",
}


def load_gas_bar(snap_path, projection_weight="density"):
    """Load gas coordinates in the bar frame and per-cell histogram weights."""
    if projection_weight not in PROJECTION_WEIGHT_FIELDS:
        raise ValueError(
            f"projection_weight must be one of {sorted(PROJECTION_WEIGHT_FIELDS)}; "
            f"got {projection_weight!r}"
        )
    weight_key = PROJECTION_WEIGHT_FIELDS[projection_weight]
    data, header = read_snapshot_hdf5(snap_path, fields=("Coordinates", weight_key))
    t = float(header["Time"])
    box = float(header["BoxSize"])
    x, y, z = np.asarray(data["Coordinates"], dtype=np.float64).T
    weights = np.asarray(data[weight_key], dtype=np.float64)
    x -= box / 2.0
    y -= box / 2.0
    z -= box / 2.0
    z0 = np.zeros_like(x)
    x, y, _, _ = rotate_to_bar_frame(x, y, z0, z0, t)
    return x, y, z, weights, header


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

# Shorter loop: edge-on (mock from-the-sun view) -> tilt to 30 deg -> one orbit -> back.
DEFAULT_EDGE_ORBIT_KEYFRAMES = (
    (0.00, 12.0, 0.0, 0.0),      # start edge-on, in the disk plane
    (0.12, 10.0, 0.0, 30.0),     # tilt up to 30 deg above the plane
    (0.88, 10.0, 360.0, 30.0),   # one full orbit at 30 deg
    (1.00, 12.0, 360.0, 0.0),    # dip back to edge-on (closes the loop)
)

# Far-out galaxy view -> zoom to CMZ -> partial orbit -> dip to edge-on and hold.
# Zoom completes at fraction 0.22 (fast camera + fast snap advance); orbit/GC phase
# from 0.22 onward uses --frames-per-snap for detailed evolution.
DEFAULT_ZOOM_OBSERVE_KEYFRAMES = (
    (0.00, 32.0, 0.0, 50.0),     # far out — more outer disk in view
    (0.22, 9.0, 0.0, 35.0),      # quick zoom to the center (was 0.40 / r=22)
    (0.60, 8.0, 55.0, 35.0),     # partial rotation (~55 deg)
    (0.85, 9.0, 70.0, 8.0),      # dip toward the midplane
    (1.00, 9.0, 70.0, 0.0),      # end edge-on (mock observational view)
)
ZOOM_OBSERVE_ZOOM_END_FRACTION = 0.22
ZOOM_OBSERVE_FRAMES_PER_SNAP_ZOOM = 8

PATH_KEYFRAMES = {
    "cinematic": DEFAULT_CINEMATIC_KEYFRAMES,
    "edge-orbit": DEFAULT_EDGE_ORBIT_KEYFRAMES,
    "zoom-observe": DEFAULT_ZOOM_OBSERVE_KEYFRAMES,
}


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


def build_camera_path(path, n_frames, r_start=12.0, r_end=6.0, n_turns=1.5, tilt_deg=35.0):
    """Return ``(positions, ups)`` for a named camera path. ``ups`` is None for orbit."""
    if path in PATH_KEYFRAMES:
        return cinematic_camera_path(n_frames, keyframes=PATH_KEYFRAMES[path])
    return camera_path(
        n_frames,
        r_start=r_start,
        r_end=r_end,
        n_turns=n_turns,
        tilt_deg=tilt_deg,
    ), None


def build_snap_indices(n_frames, n_snaps, path, frames_per_snap=2):
    """Map each frame index to a snapshot list index.

    For ``zoom-observe``, snapshots advance quickly during the zoom-in phase
    (``ZOOM_OBSERVE_FRAMES_PER_SNAP_ZOOM``) and at ``frames_per_snap`` afterward
    so simulation evolution is easier to follow near the GC.
    """
    indices = np.zeros(n_frames, dtype=int)
    if path != "zoom-observe":
        for i in range(n_frames):
            indices[i] = min(i // frames_per_snap, n_snaps - 1)
        return indices

    zoom_end = max(1, int(round(ZOOM_OBSERVE_ZOOM_END_FRACTION * n_frames)))
    snap = 0
    since_change = 0
    for i in range(n_frames):
        if i == zoom_end:
            since_change = 0
        indices[i] = min(snap, n_snaps - 1)
        in_zoom = i < zoom_end
        block = (
            ZOOM_OBSERVE_FRAMES_PER_SNAP_ZOOM if in_zoom else frames_per_snap
        )
        since_change += 1
        if since_change >= block and snap < n_snaps - 1:
            snap += 1
            since_change = 0
    return indices


def resolve_color_lock_frame(path, n_frames, frame_start, frame_end):
    """Pick which frame sets the locked log color scale for this render batch."""
    if path == "zoom-observe":
        ideal = max(0, min(n_frames - 1, int(round(ZOOM_OBSERVE_ZOOM_END_FRACTION * n_frames))))
    else:
        ideal = frame_start
    if ideal < frame_start:
        return frame_start
    if ideal >= frame_end:
        return frame_end - 1
    return ideal


def project_frame_at_index(
    frame_index,
    snap_indices,
    snap_paths,
    cameras,
    ups,
    *,
    projection_weight,
    projection_method,
    smooth_blend,
    snap_prefix=SNAP_PREFIX,
):
    """Load the snap and projection for one frame index."""
    sidx = int(snap_indices[frame_index])
    snap_path = snap_paths[sidx]
    x, y, z, weights, header = load_gas_bar(
        snap_path, projection_weight=projection_weight
    )
    cam = cameras[frame_index]
    up = (0.0, 0.0, 1.0) if ups is None else ups[frame_index]
    sigma = project_flythrough_map(
        x, y, z, weights, cam, up=up,
        projection_method=projection_method,
        smooth_blend=smooth_blend,
    )
    snap_n = snap_num_from_name(snap_path, prefix=snap_prefix)
    return sigma, snap_n, header


def max_snap_index_used(snap_indices, frame_end):
    """Highest snapshot index referenced through ``frame_end - 1``."""
    if frame_end <= 0:
        return 0
    end = min(frame_end, len(snap_indices))
    return int(snap_indices[end - 1])


def project_surface_map(
    x, y, z, weights, cam, smooth=True, up=(0.0, 0.0, 1.0),
    smooth_blend=MASKED_FILL_BLEND_MODE,
):
    sigma, _ = project_surface_density_camera(
        x,
        y,
        z,
        weights,
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
        masked_fill_blend_mode=smooth_blend,
        masked_fill_dense_threshold=MASKED_FILL_DENSE_THRESHOLD,
    )
    return sigma


def project_flythrough_map(
    x,
    y,
    z,
    weights,
    cam,
    smooth=True,
    up=(0.0, 0.0, 1.0),
    projection_method=PROJECTION_METHOD,
    smooth_blend=MASKED_FILL_BLEND_MODE,
):
    """Dispatch to surface splat (legacy) or column histogram integration."""
    if projection_method == "column":
        sigma, _ = project_column_density_camera(
            x,
            y,
            z,
            weights,
            camera_position=cam,
            target=(0.0, 0.0, 0.0),
            up_hint=up,
            fov_x_deg=FOV_X_DEG,
            nx=NX,
            ny=NY,
            nz=COLUMN_DEPTH_BINS,
            z_near=Z_NEAR,
            z_far=Z_FAR,
            smooth_sigma_px=COLUMN_SMOOTH_SIGMA_PX if smooth else None,
            deposit=COLUMN_DEPOSIT if smooth else "nearest",
        )
        return sigma
    if projection_method == "surface":
        return project_surface_map(
            x, y, z, weights, cam, smooth=smooth, up=up, smooth_blend=smooth_blend,
        )
    raise ValueError(f"projection_method must be one of {PROJECTION_METHODS}")


project_mass_map = project_flythrough_map


def color_limits(
    sigma,
    vmin_floor=VMIN_FLOOR,
    vmin_percentile=COLOR_VMIN_PERCENTILE,
    vmax_percentile=COLOR_VMAX_PERCENTILE,
):
    """Log-scale limits from positive pixels (same as preview_flythrough_frame)."""
    pos = sigma[sigma > 0]
    if pos.size == 0:
        return VMIN, VMAX
    vmin = max(float(np.percentile(pos, vmin_percentile)), vmin_floor)
    vmax = float(np.percentile(pos, vmax_percentile))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return vmin, vmax


def frame_index_from_array_name(path):
    m = re.search(r"frame_(\d+)\.npz$", Path(path).name, re.I)
    if not m:
        raise ValueError(f"expected frame_<N>.npz, got {Path(path).name}")
    return int(m.group(1))


def frame_array_path(output_dir, frame_index):
    return Path(output_dir) / ARRAYS_SUBDIR / f"frame_{frame_index:04d}.npz"


def arrays_dir_for(output_dir):
    return Path(output_dir) / ARRAYS_SUBDIR


def save_frame_array(sigma, out_path, *, snap_number, time_myr, frame_index):
    """Save a raw projection map for fast colormap / scale re-rendering."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        sigma=np.asarray(sigma, dtype=np.float32),
        snap_number=np.int32(snap_number),
        time_myr=np.float64(time_myr),
        frame_index=np.int32(frame_index),
    )


def load_frame_array(path):
    with np.load(path) as data:
        return {
            "sigma": np.asarray(data["sigma"], dtype=np.float64),
            "snap_number": int(data["snap_number"]),
            "time_myr": float(data["time_myr"]),
            "frame_index": int(data["frame_index"]),
        }


def list_frame_arrays(arrays_dir):
    paths = sorted(
        Path(arrays_dir).glob("frame_*.npz"),
        key=frame_index_from_array_name,
    )
    if not paths:
        raise FileNotFoundError(f"no frame_*.npz in {arrays_dir}")
    return paths


def sigma_code_to_n_mol_cm2(sigma):
    """Convert column-histogram sums to μ-weighted particle column density (cm⁻²)."""
    return code_density_sum_to_n_mol_cm2(sigma, COLUMN_DEPTH_CODE)


def write_png(
    sigma,
    out_path,
    vmin,
    vmax,
    title=None,
    time_myr=None,
    cmap=DEFAULT_CMAP,
    show_colorbar=True,
    colorbar_label=SIGMA_COLORBAR_LABEL,
):
    out = np.asarray(sigma, dtype=np.float64).copy()
    out[~np.isfinite(out)] = vmin
    out[out <= 0] = vmin
    display = sigma_code_to_n_mol_cm2(out)
    disp_vmin = vmin * SIGMA_CODE_TO_N_MOL_CM2
    disp_vmax = vmax * SIGMA_CODE_TO_N_MOL_CM2

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 6), dpi=150)
    im = ax.imshow(
        display,
        origin="lower",
        extent=(-1, 1, -1, 1),
        norm=colors.LogNorm(vmin=disp_vmin, vmax=disp_vmax),
        cmap=resolve_cmap(cmap),
        interpolation="nearest",
    )
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
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
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=_PKG_ROOT / "example_output" / "flythrough_frames",
    )
    parser.add_argument(
        "--path",
        choices=("orbit", "cinematic", "edge-orbit", "zoom-observe"),
        default="orbit",
        help="camera path: 'orbit' (tilted circle, default), 'cinematic' "
        "(zoom-in / orbit from above / dip to edge-on / exit below), "
        "'edge-orbit' (edge-on -> 30 deg -> one orbit -> edge-on loop), or "
        "'zoom-observe' (far-out galaxy -> zoom to CMZ -> partial orbit -> "
        "edge-on observational end); keyframed paths ignore "
        "--r-start/--r-end/--n-turns/--tilt-deg",
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
        help="autoscale vmin/vmax from a reference frame, then hold fixed (on by "
        "default); zoom-observe uses the CMZ zoom-arrival frame; pass "
        "--no-lock-color-scale to use the fixed --vmin/--vmax",
    )
    parser.add_argument(
        "--color-lock-frame",
        type=int,
        default=None,
        help="frame index for color-scale lock (default: first frame, or CMZ "
        "zoom-arrival frame for zoom-observe)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="print a progress line every N rendered frames (default: 1)",
    )
    parser.add_argument(
        "--projection-weight",
        choices=tuple(PROJECTION_WEIGHT_FIELDS),
        default="density",
        help="histogram weight per gas cell: 'density' (default, matches "
        "project_column_density_xy) or 'mass'",
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
        help="surface: 2D image histogram + masked_fill; column (default): sum "
        "density along depth per pixel (like project_column_density_xy)",
    )
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="also save raw projection maps as compressed .npz under "
        "<output-dir>/arrays/ for fast colormap / scale iteration "
        "(see render_flythrough_from_arrays.py)",
    )
    parser.add_argument(
        "--skip-png",
        action="store_true",
        help="skip PNG output (use with --save-arrays to avoid duplicate work "
        "during long projection runs)",
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
    if args.skip_png and not args.save_arrays:
        _log("ERROR: --skip-png requires --save-arrays")
        sys.exit(1)
    n_render = frame_end - args.frame_start
    snap_indices = build_snap_indices(
        args.n_frames, len(snap_paths), args.path, args.frames_per_snap
    )
    snaps_needed = max_snap_index_used(snap_indices, frame_end) + 1
    if snaps_needed > len(snap_paths):
        _log(
            f"warning: frames through {frame_end - 1} need {snaps_needed} snaps "
            f"but only {len(snap_paths)} available after filter; last snap will repeat"
        )
    elif n_render > len(snap_paths) * args.frames_per_snap and args.path != "zoom-observe":
        _log(
            f"warning: {n_render} frames need more snaps than available "
            f"({len(snap_paths)} after filter); last snap will repeat"
        )

    if args.path in PATH_KEYFRAMES:
        cameras, ups = build_camera_path(args.path, args.n_frames)
        _log(f"camera path: {args.path} (keyframed); "
             "--r-start/--r-end/--n-turns/--tilt-deg ignored")
    else:
        cameras, ups = build_camera_path(
            args.path,
            args.n_frames,
            r_start=args.r_start,
            r_end=args.r_end,
            n_turns=args.n_turns,
            tilt_deg=args.tilt_deg,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    snap_lo = snap_num_from_name(snap_paths[0], args.snap_prefix)
    snap_hi = snap_num_from_name(snap_paths[-1], args.snap_prefix)
    _log("=== render_flythrough_movie ===")
    _log(f"{len(snap_paths)} snaps after filter ({snap_lo} .. {snap_hi})")
    _log(f"camera path length: {args.n_frames}  render frames {args.frame_start} .. {frame_end - 1}")
    _log(f"frames_per_snap={args.frames_per_snap}  progress_every={args.progress_every}")
    if args.path == "zoom-observe":
        zoom_frames = max(1, int(round(ZOOM_OBSERVE_ZOOM_END_FRACTION * args.n_frames)))
        _log(
            f"zoom-observe pacing: frames 0..{zoom_frames - 1} use "
            f"frames_per_snap={ZOOM_OBSERVE_FRAMES_PER_SNAP_ZOOM}, "
            f"then {zoom_frames}..{args.n_frames - 1} use frames_per_snap="
            f"{args.frames_per_snap}"
        )
    _log(f"projection_weight={args.projection_weight}  projection_method={args.projection_method}")
    if args.projection_method == "surface":
        _log(f"smooth_blend={args.smooth_blend}")
    if args.save_arrays:
        _log(f"arrays -> {arrays_dir_for(args.output_dir).resolve()}")
    if args.skip_png:
        _log("PNG output disabled (--skip-png)")
    _log(f"output -> {args.output_dir.resolve()}")

    vmin = args.vmin
    vmax = args.vmax
    if args.lock_color_scale:
        if args.color_lock_frame is not None:
            if not (args.frame_start <= args.color_lock_frame < frame_end):
                _log(
                    f"warning: --color-lock-frame {args.color_lock_frame} outside "
                    f"render range [{args.frame_start}, {frame_end}); clamping"
                )
            lock_frame = max(
                args.frame_start, min(args.color_lock_frame, frame_end - 1)
            )
        else:
            lock_frame = resolve_color_lock_frame(
                args.path, args.n_frames, args.frame_start, frame_end
            )
        cmz_frame = int(round(ZOOM_OBSERVE_ZOOM_END_FRACTION * args.n_frames))
        _log(f"precomputing color scale from frame {lock_frame}...")
        sigma_ref, snap_n_lock, _ = project_frame_at_index(
            lock_frame,
            snap_indices,
            snap_paths,
            cameras,
            ups,
            projection_weight=args.projection_weight,
            projection_method=args.projection_method,
            smooth_blend=args.smooth_blend,
            snap_prefix=args.snap_prefix,
        )
        vmin, vmax = color_limits(sigma_ref)
        if args.path == "zoom-observe" and lock_frame == cmz_frame:
            _log(
                f"locked color scale at CMZ zoom arrival: frame {lock_frame}  "
                f"snap {snap_n_lock}  vmin={vmin:.4g}  vmax={vmax:.4g}"
            )
        else:
            _log(
                f"locked color scale from frame {lock_frame}  snap {snap_n_lock}  "
                f"vmin={vmin:.4g}  vmax={vmax:.4g}"
            )
            if args.path == "zoom-observe" and cmz_frame < args.frame_start:
                _log(
                    f"  (CMZ arrival is frame {cmz_frame}, before this batch; "
                    "pass --vmin/--vmax from an earlier batch for consistency)"
                )

    _log("starting render loop...")

    cached = None
    x = y = z = weights = None
    header = None
    t0 = time.time()

    for i in range(args.frame_start, frame_end):
        sidx = int(snap_indices[i])
        snap_n = snap_num_from_name(snap_paths[sidx], args.snap_prefix)
        if sidx != cached:
            snap_path = snap_paths[sidx]
            _log(f"\nloading snap {snap_n}: {snap_path.name}")
            t_load = time.time()
            x, y, z, weights, header = load_gas_bar(
                snap_path, projection_weight=args.projection_weight
            )
            _log(
                f"  loaded in {time.time() - t_load:.1f}s  "
                f"N_gas = {x.size:,}  Time = {header['Time']:.6g}"
            )
            cached = sidx

        cam = cameras[i]
        up = (0.0, 0.0, 1.0) if ups is None else ups[i]
        sigma = project_flythrough_map(
            x, y, z, weights, cam, up=up,
            projection_method=args.projection_method,
            smooth_blend=args.smooth_blend,
        )

        if args.save_arrays:
            array_path = frame_array_path(args.output_dir, i)
            save_frame_array(
                sigma,
                array_path,
                snap_number=snap_n,
                time_myr=float(header["Time"]) * CODE_TIME_TO_MYR,
                frame_index=i,
            )

        if not args.skip_png:
            out_path = args.output_dir / f"frame_{i:04d}.png"
            title = f"frame {i:04d}  snap {snap_n}"
            write_png(
                sigma,
                out_path,
                vmin,
                vmax,
                title=title,
                time_myr=float(header["Time"]) * CODE_TIME_TO_MYR,
            )

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
