"""flythrough script helpers"""

import numpy as np
import pytest

import scripts.preview_flythrough_frame as preview
import scripts.render_flythrough_movie as movie
from scripts.preview_flythrough_frame import build_parser as build_preview_parser
from scripts.preview_flythrough_frame import camera_path, snap_num_from_name
from scripts.render_flythrough_movie import (
    build_parser,
    cinematic_camera_path,
    color_limits,
    list_snaps,
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
    vmin, vmax = color_limits(sigma)
    assert vmin > 0 and vmax > vmin
    assert vmin <= np.percentile(sigma, 5)
    assert vmax >= np.percentile(sigma, 95)


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


def test_movie_no_lock_uses_fixed_limits():
    args = build_parser().parse_args(["--snap-dir", "/tmp", "--no-lock-color-scale"])
    assert args.lock_color_scale is False


def test_preview_reuses_movie_definitions():
    """Preview must use the movie's shared params/helpers, not its own copies."""
    assert preview.VMIN is movie.VMIN
    assert preview.VMAX is movie.VMAX
    assert preview.project_mass_map is movie.project_mass_map
    assert preview.camera_path is movie.camera_path
    assert preview.color_limits is movie.color_limits


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
