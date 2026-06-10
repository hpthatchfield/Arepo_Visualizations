"""tests for annoying functions that aren't really working how I want them to."""

import numpy as np
import pytest

from simviz.field_plots import (
    adaptive_gaussian_blend_mass_map,
    masked_fill_mass_map,
    project_column_density_camera,
    project_surface_density_camera,
)


def test_adaptive_gaussian_blend_mass_map():
    rng = np.random.default_rng(1)
    sigma = rng.random((32, 32)) * 1e-3
    sigma[10:20, 10:20] += 10.0
    out = adaptive_gaussian_blend_mass_map(
        sigma, sigma_sharp_px=0.5, sigma_smooth_px=2.0, weight_sigma_px=0.5
    )
    assert out.shape == sigma.shape
    assert np.all(np.isfinite(out))


def test_project_surface_density_camera_smooth_sigma():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(-5.0, 5.0, n)
    y = rng.uniform(-5.0, 5.0, n)
    z = rng.uniform(1.0, 20.0, n)
    m = rng.uniform(0.01, 1.0, n)
    cam = np.array([8.0, 4.0, -15.0])

    s0, _ = project_surface_density_camera(
        x, y, z, m, camera_position=cam, nx=48, ny=48, z_far=100.0, smooth_sigma_px=None
    )
    s1, _ = project_surface_density_camera(
        x, y, z, m, camera_position=cam, nx=48, ny=48, z_far=100.0, smooth_sigma_px=1.5
    )
    assert s0.shape == s1.shape == (48, 48)
    assert np.all(np.isfinite(s1))


def test_masked_fill_mass_map():
    rng = np.random.default_rng(3)
    sigma = rng.random((32, 32)) * 1e-3
    sigma[10:20, 10:20] += 10.0
    out = masked_fill_mass_map(
        sigma, sigma_fill_px=4.0, sigma_sharp_px=0.0, weight_sigma_px=0.5
    )
    assert out.shape == sigma.shape
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    assert out[15, 15] > out[0, 0]


def test_masked_fill_mass_map_mask_power_runs():
    sigma = np.full((32, 32), 1e-4, dtype=np.float64)
    sigma[16, 16] = 1e-2
    out = masked_fill_mass_map(
        sigma,
        sigma_fill_px=6.0,
        sigma_sharp_px=0.0,
        weight_sigma_px=1.0,
        percentiles=(20.0, 85.0),
        mask_power=2.0,
    )
    assert out.shape == sigma.shape
    assert np.all(np.isfinite(out))


def test_masked_fill_detail_blend_avoids_shadow_trough():
    """Detail blend must not fall below the wide-blur fill at boundaries."""
    sigma = np.zeros((64, 64), dtype=np.float64)
    sigma[32, 32] = 100.0
    linear = masked_fill_mass_map(
        sigma,
        sigma_fill_px=8.0,
        sigma_sharp_px=0.0,
        weight_sigma_px=1.0,
        percentiles=(20.0, 85.0),
        blend_mode="linear",
    )
    detail = masked_fill_mass_map(
        sigma,
        sigma_fill_px=8.0,
        sigma_sharp_px=0.0,
        weight_sigma_px=1.0,
        percentiles=(20.0, 85.0),
        blend_mode="detail",
    )
    fill = masked_fill_mass_map(
        sigma,
        sigma_fill_px=8.0,
        sigma_sharp_px=0.0,
        weight_sigma_px=0.0,
        percentiles=(0.0, 100.0),
        blend_mode="linear",
    )
    # detail should never be below fill envelope
    assert np.all(detail + 1e-12 >= fill)
    # linear blend often dips below fill near bright peaks
    assert np.any(linear < fill - 1e-6)
    # dense peak should stay sharp (not replaced by wide blur)
    assert detail[32, 32] >= 0.95 * sigma[32, 32]


def test_project_surface_density_camera_masked_fill():
    rng = np.random.default_rng(4)
    n = 400
    x = rng.uniform(-5.0, 5.0, n)
    y = rng.uniform(-5.0, 5.0, n)
    z = rng.uniform(1.0, 20.0, n)
    m = rng.uniform(0.01, 1.0, n)
    cam = np.array([8.0, 4.0, -15.0])

    s_mask, _ = project_surface_density_camera(
        x,
        y,
        z,
        m,
        camera_position=cam,
        nx=48,
        ny=48,
        z_far=100.0,
        masked_fill_sigmas=(0.5, 3.0),
        masked_fill_weight_sigma_px=0.5,
    )
    assert s_mask.shape == (48, 48)
    assert np.all(np.isfinite(s_mask))


def test_project_surface_density_camera_adaptive_sigmas():
    rng = np.random.default_rng(2)
    n = 400
    x = rng.uniform(-5.0, 5.0, n)
    y = rng.uniform(-5.0, 5.0, n)
    z = rng.uniform(1.0, 20.0, n)
    m = rng.uniform(0.01, 1.0, n)
    cam = np.array([8.0, 4.0, -15.0])

    s_adapt, _ = project_surface_density_camera(
        x,
        y,
        z,
        m,
        camera_position=cam,
        nx=48,
        ny=48,
        z_far=100.0,
        adaptive_smooth_sigmas=(0.8, 2.5),
        adaptive_weight_sigma_px=0.5,
    )
    assert s_adapt.shape == (48, 48)
    assert np.all(np.isfinite(s_adapt))


def test_project_column_density_camera_sums_along_depth():
    """Two cells on the same sightline should add, not compete as surface splats."""
    cam = np.array([0.0, 0.0, -10.0])
    up = (0.0, 1.0, 0.0)
    x = np.array([0.0, 0.0])
    y = np.array([0.0, 0.0])
    z = np.array([4.0, 8.0])
    weights = np.array([1.0, 2.0])

    col, _ = project_column_density_camera(
        x, y, z, weights, camera_position=cam, up_hint=up, nx=32, ny=32, nz=8,
        z_near=0.5, z_far=20.0, smooth_sigma_px=None,
    )
    assert col.shape == (32, 32)
    assert float(col.max()) == pytest.approx(3.0)
    assert float(col.sum()) == pytest.approx(3.0)


def test_project_surface_density_camera_density_vs_mass_weights():
    rng = np.random.default_rng(5)
    n = 500
    x = rng.uniform(-4.0, 4.0, n)
    y = rng.uniform(-4.0, 4.0, n)
    z = rng.uniform(2.0, 18.0, n)
    rho = rng.uniform(1e-4, 1.0, n)
    mass = rho * rng.uniform(0.5, 2.0, n)
    cam = np.array([7.0, 3.0, -12.0])

    s_rho, _ = project_surface_density_camera(
        x, y, z, rho, camera_position=cam, nx=48, ny=48, z_far=100.0,
        smooth_sigma_px=None,
    )
    s_mass, _ = project_surface_density_camera(
        x, y, z, mass, camera_position=cam, nx=48, ny=48, z_far=100.0,
        smooth_sigma_px=None,
    )
    assert s_rho.shape == s_mass.shape == (48, 48)
    assert not np.allclose(s_rho, s_mass)
