"""flythrough script helpers"""

from pathlib import Path

import numpy as np
import pytest

import scripts.preview_flythrough_frame as preview
import scripts.render_flythrough_movie as movie
from scripts.preview_flythrough_frame import build_parser as build_preview_parser
from scripts.preview_flythrough_frame import resolve_preview_output
from scripts.preview_flythrough_frame import snap_num_from_name
from scripts.render_flythrough_movie import camera_path
from scripts.render_flythrough_movie import (
    build_camera_path,
    build_parser,
    build_snap_indices,
    cinematic_camera_path,
    color_limits,
    format_progress_line,
    frame_array_path,
    list_snaps,
    load_frame_array,
    resolve_color_lock_frame,
    save_frame_array,
    ZOOM_OBSERVE_FRAMES_PER_SNAP_ZOOM,
    ZOOM_OBSERVE_ZOOM_END_FRACTION,
)


def test_snap_num_from_name():
    assert snap_num_from_name("phoenix_stinks_1Msun_999.hdf5") == 999
    assert snap_num_from_name("/data/phoenix_stinks_1Msun_1000.hdf5") == 1000
    assert snap_num_from_name("whole_disk_300.hdf5", prefix="whole_disk") == 300


def test_snap_num_from_name_bad():
    with pytest.raises(ValueError):
        snap_num_from_name("snapshot_300.hdf5")


def test_camera_path_shape():
    path = camera_path(12, r_start=10.0, r_end=8.0, n_turns=1.0)
    assert path.shape == (12, 3)


def test_list_snaps_first_snap_number(tmp_path):
    for n in (100, 200, 300):
        (tmp_path / f"phoenix_stinks_1Msun_{n}.hdf5").write_bytes(b"")
    paths = list_snaps(tmp_path, first_snap_number=200)
    nums = [snap_num_from_name(p) for p in paths]
    assert nums == [200, 300]


def test_color_limits_percentiles():
    sigma = np.logspace(-2, 2, 1000)
    floor = movie.vmin_floor_code(movie.COLUMN_DEPTH_CODE)
    vmin, vmax = color_limits(sigma, vmin_floor=floor)
    assert vmin > 0 and vmax > vmin
    assert vmin >= max(np.percentile(sigma, movie.COLOR_VMIN_PERCENTILE), floor)
    assert vmax >= np.percentile(sigma, movie.COLOR_VMAX_PERCENTILE - 1)


def test_resolve_cmap_rainforest():
    from simviz.colormaps import resolve_cmap

    cmap = resolve_cmap("rainforest")
    assert hasattr(cmap, "name")


def test_frames_per_snap_index():
    n_snaps = 10
    for i, expected in [(0, 0), (4, 0), (5, 1), (49, 9)]:
        sidx = min(i // 5, n_snaps - 1)
        assert sidx == expected


def test_movie_defaults_to_autoscale_lock():
    args = build_parser().parse_args(["--snap-dir", "/tmp"])
    assert args.lock_color_scale is True
    assert args.vmin == movie.VMIN
    assert args.vmax == movie.VMAX
    assert args.progress_every == 1


def test_format_progress_line():
    line = format_progress_line(
        done=10,
        total=100,
        frame_index=9,
        snap_number=500,
        elapsed_s=600.0,
    )
    assert "[10/100" in line
    assert "frame 0009" in line
    assert "snap 500" in line
    assert "elapsed=10.0m" in line
    assert "eta=90.0m" in line


def test_movie_no_lock_uses_fixed_limits():
    args = build_parser().parse_args(["--snap-dir", "/tmp", "--no-lock-color-scale"])
    assert args.lock_color_scale is False


def test_preview_reuses_movie_definitions():
    """Preview must use the movie's shared params/helpers, not its own copies."""
    assert preview.VMIN is movie.VMIN
    assert preview.VMAX is movie.VMAX
    assert preview.project_flythrough_map is movie.project_flythrough_map
    assert preview.build_camera_path is movie.build_camera_path
    assert preview.color_limits is movie.color_limits


def test_movie_defaults_to_density_projection_weight():
    args = build_parser().parse_args(["--snap-dir", "/tmp"])
    assert args.projection_weight == "density"
    assert args.smooth_blend == "detail"
    assert args.projection_method == "column"


def test_preview_defaults_to_autoscale():
    args = build_preview_parser().parse_args(["-s", "/tmp/x.hdf5"])
    assert args.auto_scale is True


def test_preview_no_auto_scale_uses_fixed():
    args = build_preview_parser().parse_args(["-s", "/tmp/x.hdf5", "--no-auto-scale"])
    assert args.auto_scale is False


def test_default_path_is_orbit():
    assert build_parser().parse_args(["--snap-dir", "/tmp"]).path == "orbit"
    assert build_preview_parser().parse_args(["-s", "/tmp/x.hdf5"]).path == "orbit"


def test_cinematic_camera_path_shape_and_ups():
    pts, ups = cinematic_camera_path(50)
    assert pts.shape == (50, 3)
    assert ups.shape == (50, 3)
    # ups are unit length
    assert np.allclose(np.linalg.norm(ups, axis=1), 1.0)
    # up never parallel to the view direction (camera faces the origin)
    forward = -pts / np.linalg.norm(pts, axis=1, keepdims=True)
    assert np.all(np.abs(np.sum(ups * forward, axis=1)) < 0.999)


def test_cinematic_camera_path_endpoints():
    pts, _ = cinematic_camera_path(200)
    r0 = np.linalg.norm(pts[0])
    r_end = np.linalg.norm(pts[-1])
    assert np.isclose(r0, 18.0, atol=0.2)
    assert np.isclose(r_end, 18.0, atol=0.2)
    assert pts[0][2] > 0          # starts above the disk plane
    assert pts[-1][2] < 0         # ends below the disk plane


def test_edge_orbit_path_endpoints():
    pts, ups = build_camera_path("edge-orbit", 240)
    assert pts.shape == (240, 3)
    assert ups.shape == (240, 3)
    assert np.isclose(pts[0][2], 0.0, atol=0.05)   # starts in-plane (edge-on)
    assert np.isclose(pts[-1][2], 0.0, atol=0.05)   # ends in-plane
    assert np.allclose(pts[0], pts[-1], atol=0.15)  # smooth loop closure
    mid = int(0.5 * 240)
    assert pts[mid][2] > 3.0                      # ~30 deg orbit mid-point


def test_mock_sun_view_az_el_deg():
    from simviz.projections import (
        GALACTIC_ORIGIN,
        MOCK_SUN_VIEW_BAR_OFFSET_DEG,
        mock_sun_view_az_el_deg,
    )

    az, el = mock_sun_view_az_el_deg()
    assert MOCK_SUN_VIEW_BAR_OFFSET_DEG == 30.0
    assert np.isclose(el, 0.0, atol=1e-9)
    assert GALACTIC_ORIGIN["xsun"] < 0.0
    # GC→Sun axis in the disk plane, rotated +30° CCW: (-cos30, -sin30, 0).
    assert az < -90.0 or az > 90.0  # third/fourth quadrant (sun side)
    r = 9.0
    x = r * np.cos(np.radians(el)) * np.cos(np.radians(az))
    y = r * np.cos(np.radians(el)) * np.sin(np.radians(az))
    assert x < 0.0 and y < 0.0


def test_opening_camera_radius():
    r = movie.opening_camera_radius(disk_half_width_code=125.0)
    assert 200.0 < r < 280.0
    assert movie.column_z_far_for_camera((r, 0.0, r * 0.996)) > r


def test_zoom_observe_path_endpoints():
    from simviz.projections import mock_sun_view_az_el_deg

    pts, _ = build_camera_path("zoom-observe", 240)
    sun_az, sun_el = mock_sun_view_az_el_deg()
    r0 = np.linalg.norm(pts[0])
    r_open = movie.opening_camera_radius()
    r_end = np.linalg.norm(pts[-1])
    end_az = np.degrees(np.arctan2(pts[-1][1], pts[-1][0]))
    end_el = np.degrees(np.arcsin(np.clip(pts[-1][2] / r_end, -1.0, 1.0)))
    start_el = np.degrees(np.arcsin(np.clip(pts[0][2] / r0, -1.0, 1.0)))
    assert np.isclose(r0, r_open, rtol=0.02)
    assert start_el > 80.0
    assert r0 > r_end + 5.0
    assert np.isclose(end_az, sun_az, atol=0.5)     # ends on mock solar l-b axis
    assert np.isclose(end_el, sun_el, atol=0.5)
    assert pts[-1][0] < 0.0                         # sun side of GC (xsun < 0)
    assert not np.allclose(pts[0], pts[-1], atol=1.0)  # does not loop back
    zoom_frame = int(round(ZOOM_OBSERVE_ZOOM_END_FRACTION * 240))
    r_zoom = np.linalg.norm(pts[zoom_frame])
    assert r_zoom < r0 - 10.0                       # zoom completes well before orbit


def test_zoom_observe_snap_indices_advance_faster_during_zoom():
    n_frames = 120
    n_snaps = 100
    indices = build_snap_indices(n_frames, n_snaps, "zoom-observe", frames_per_snap=2)
    zoom_end = max(1, int(round(ZOOM_OBSERVE_ZOOM_END_FRACTION * n_frames)))
    zoom_snaps = indices[zoom_end - 1] + 1
    detail_span = n_frames - zoom_end
    detail_snaps = indices[-1] - indices[zoom_end] + 1
    zoom_rate = zoom_end / max(zoom_snaps, 1)
    detail_rate = detail_span / max(detail_snaps, 1)
    assert zoom_rate < detail_rate
    assert zoom_rate <= ZOOM_OBSERVE_FRAMES_PER_SNAP_ZOOM + 0.5
    assert detail_rate >= 2.0 - 0.5


def test_resolve_color_lock_frame_zoom_observe():
    n_frames = 1202
    cmz = int(round(ZOOM_OBSERVE_ZOOM_END_FRACTION * n_frames))
    assert resolve_color_lock_frame("zoom-observe", n_frames, 0, 1202) == cmz
    assert resolve_color_lock_frame("zoom-observe", n_frames, 0, 300) == cmz
    assert resolve_color_lock_frame("zoom-observe", n_frames, 720, 1202) == 720
    assert resolve_color_lock_frame("orbit", n_frames, 100, 200) == 100


def test_build_camera_path_orbit_has_no_ups():
    pts, ups = build_camera_path("orbit", 12)
    assert pts.shape == (12, 3)
    assert ups is None


def test_save_and_load_frame_array(tmp_path):
    sigma = np.logspace(-2, 2, 16).reshape(4, 4)
    path = frame_array_path(tmp_path, 7)
    save_frame_array(
        sigma,
        path,
        snap_number=880,
        time_myr=12.3,
        frame_index=7,
        column_depth_code=42.5,
    )
    loaded = load_frame_array(path)
    assert loaded["frame_index"] == 7
    assert loaded["snap_number"] == 880
    assert loaded["time_myr"] == pytest.approx(12.3)
    assert loaded["column_depth_code"] == pytest.approx(42.5)
    assert loaded["sigma"].shape == (4, 4)
    assert np.allclose(loaded["sigma"], sigma, rtol=1e-5)


def test_movie_save_arrays_flags():
    args = build_parser().parse_args(["--snap-dir", "/tmp", "--save-arrays"])
    assert args.save_arrays is True
    assert args.skip_png is False


def test_movie_skip_png_requires_save_arrays():
    args = build_parser().parse_args(
        ["--snap-dir", "/tmp", "--save-arrays", "--skip-png"]
    )
    assert args.skip_png is True
    assert args.save_arrays is True


def test_preview_auto_output_name():
    path = resolve_preview_output(
        None, path="zoom-observe", frame_index=264, tag=None
    )
    assert path.name == "flythrough_preview_zoom-observe_f0264.png"


def test_preview_auto_output_name_with_tag():
    path = resolve_preview_output(
        None, path="zoom-observe", frame_index=0, tag="far_out"
    )
    assert path.name == "flythrough_preview_zoom-observe_f0000_far_out.png"


def test_preview_explicit_output_ignores_tag():
    path = resolve_preview_output(
        Path("custom.png"),
        path="zoom-observe",
        frame_index=0,
        tag="ignored",
    )
    assert path == Path("custom.png")


def test_preview_compare_deposit_flag():
    args = build_preview_parser().parse_args(
        ["--snap-dir", "/tmp", "--snap-number", "500", "--compare-deposit"]
    )
    assert args.compare_deposit is True
