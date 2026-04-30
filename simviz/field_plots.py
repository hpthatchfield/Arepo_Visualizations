"""2D histogram maps, field projections, and B-field visualizations."""

import numpy as np
try:
    from fast_histogram import histogram2d
except ImportError:  # pragma: no cover
    from numpy import histogram2d as _np_histogram2d

    def histogram2d(x, y, range, bins, weights=None):
        hist, _, _ = _np_histogram2d(x, y, bins=bins, range=range, weights=weights)
        return hist


DEFAULT_LBV_GRID = {
    "lmin": -2.3,
    "lmax": 2.3,
    "dl": 0.001,
    "bmin": -1.5,
    "bmax": 1.5,
    "db": 0.001,
    "vmin": -350.0,
    "vmax": 350.0,
    "dv": 1.7,
}

DEFAULT_LB_BFIELD_GRID = {
    "lmin": -2.3,
    "lmax": 2.3,
    "dl": 0.025,
    "bmin": -0.85,
    "bmax": 0.85,
    "db": 0.025,
}

DEFAULT_XYZ_GRID = {
    "xmin": -3.0,
    "xmax": 3.0,
    "dx": 0.005,
    "ymin": -3.0,
    "ymax": 3.0,
    "dy": 0.005,
    "zmin": -2.0,
    "zmax": 2.0,
    "dz": 0.005,
}


def _grid_centers_and_edges(vmin, vmax, dv):
    centers = np.arange(vmin, vmax + dv, dv)
    edges = np.arange(vmin - dv / 2.0, vmax + 3.0 * dv / 2.0, dv)
    return centers, edges


def create_lbrv_maps(l, b, vr, r, masses, grid=None):
    """Create l-b and l-v weighted histogram maps using fast_histogram.

    Parameters
    ----------
    l, b, vr, r, masses : array-like
        Longitude, latitude, radial velocity, radius, and mass arrays.
    grid : dict, optional
        Grid parameters. Uses ``DEFAULT_LBV_GRID`` when omitted.

    Returns
    -------
    lb_map, lv_map, ls, bs, vs : ndarray
        l-b and l-v maps and their axis center arrays.
    """
    cfg = DEFAULT_LBV_GRID.copy()
    if grid:
        cfg.update(grid)

    ls, l_edges = _grid_centers_and_edges(cfg["lmin"], cfg["lmax"], cfg["dl"])
    bs, b_edges = _grid_centers_and_edges(cfg["bmin"], cfg["bmax"], cfg["db"])
    vs, v_edges = _grid_centers_and_edges(cfg["vmin"], cfg["vmax"], cfg["dv"])

    weights = masses / r**2

    lb_map = histogram2d(
        l,
        b,
        range=[[cfg["lmin"], cfg["lmax"]], [cfg["bmin"], cfg["bmax"]]],
        bins=[len(l_edges), len(b_edges)],
        weights=weights,
    )
    lv_map = histogram2d(
        l,
        vr,
        range=[[cfg["lmin"], cfg["lmax"]], [cfg["vmin"], cfg["vmax"]]],
        bins=[len(l_edges), len(v_edges)],
        weights=weights,
    )

    return lb_map.T, lv_map.T, ls, bs, vs


def find_xy_map(points_xyz, column_density, grid=None):
    """Create x-y and x-z weighted maps using fast_histogram."""
    cfg = DEFAULT_XYZ_GRID.copy()
    if grid:
        cfg.update(grid)

    x, y, z = np.asarray(points_xyz).T

    xs, x_edges = _grid_centers_and_edges(cfg["xmin"], cfg["xmax"], cfg["dx"])
    ys, y_edges = _grid_centers_and_edges(cfg["ymin"], cfg["ymax"], cfg["dy"])
    zs, z_edges = _grid_centers_and_edges(cfg["zmin"], cfg["zmax"], cfg["dz"])

    h_xy = histogram2d(
        x,
        y,
        range=[[cfg["xmin"], cfg["xmax"]], [cfg["ymin"], cfg["ymax"]],],
        bins=(len(x_edges), len(y_edges)),
        weights=column_density,
    )
    h_xz = histogram2d(
        x,
        z,
        range=[[cfg["xmin"], cfg["xmax"]], [cfg["zmin"], cfg["zmax"]],],
        bins=(len(x_edges), len(z_edges)),
        weights=column_density,
    )

    return h_xy.T, h_xz.T, xs, ys, zs, x_edges, y_edges, z_edges


def plot_surface_density(
    surface_density,
    extent,
    ax=None,
    cmap="Greys",
    vmin=1e-2,
    vmax=1e2,
):
    """Plot a 2D surface-density map.

    Parameters
    ----------
    surface_density : ndarray
        2D map to render with ``imshow``.
    extent : tuple
        ``(xmin, xmax, ymin, ymax)`` passed to ``imshow``.
    ax : matplotlib.axes.Axes, optional
        Existing axes. A new figure and axes are created when omitted.
    cmap : str, optional
        Colormap name.
    vmin, vmax : float, optional
        Log-normalization range.

    Returns
    -------
    fig, ax, im : tuple
        Figure, axes, and image artist.
    """
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    else:
        fig = ax.figure

    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(surface_density, extent=extent, norm=norm, cmap=cmap, origin="lower")
    return fig, ax, im


plot_surface_density_map = plot_surface_density


def project_surface_density_camera(
    x,
    y,
    z,
    masses,
    camera_position,
    target=(0.0, 0.0, 0.0),
    up_hint=(0.0, 0.0, 1.0),
    fov_x_deg=60.0,
    fov_y_deg=None,
    nx=600,
    ny=600,
    z_near=1e-3,
    z_far=None,
):
    """Project mass to a perspective camera image plane.

    This creates a surface-density-like map in camera coordinates by summing mass
    from all particles along each line of sight.

    Parameters
    ----------
    x, y, z : array-like
        Particle positions in a common Cartesian frame.
    masses : array-like
        Particle masses (code units).
    camera_position : array-like, shape (3,)
        Camera position in the same coordinate frame as ``x,y,z``.
    target : array-like, shape (3,), optional
        Point the camera looks at; default origin.
    up_hint : array-like, shape (3,), optional
        Approximate up direction for camera roll control.
    fov_x_deg, fov_y_deg : float, optional
        Horizontal/vertical field of view in degrees. If ``fov_y_deg`` is None,
        it is set from aspect ratio ``ny/nx``.
    nx, ny : int, optional
        Output image resolution.
    z_near, z_far : float or None, optional
        Near/far depth clipping in camera coordinates.

    Returns
    -------
    sigma : ndarray, shape (ny, nx)
        Perspective-projected mass map.
    extent : tuple
        Fixed image-plane extent ``(-1, 1, -1, 1)`` for plotting.
    """
    from .projections import world_to_camera

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)

    x_cam, y_cam, z_cam = world_to_camera(
        x, y, z, camera_position=camera_position, target=target, up_hint=up_hint
    )

    fov_y = fov_y_deg
    if fov_y is None:
        fov_y = fov_x_deg * (ny / float(nx))

    tx = np.tan(np.radians(fov_x_deg) / 2.0)
    ty = np.tan(np.radians(fov_y) / 2.0)

    depth_ok = z_cam > z_near
    if z_far is not None:
        depth_ok &= z_cam < z_far

    x_img = x_cam[depth_ok] / (z_cam[depth_ok] * tx)
    y_img = y_cam[depth_ok] / (z_cam[depth_ok] * ty)
    m_img = masses[depth_ok]

    in_view = (np.abs(x_img) <= 1.0) & (np.abs(y_img) <= 1.0)
    sigma = histogram2d(
        x_img[in_view],
        y_img[in_view],
        range=[[-1.0, 1.0], [-1.0, 1.0]],
        bins=(nx, ny),
        weights=m_img[in_view],
    ).T

    return sigma, (-1.0, 1.0, -1.0, 1.0)


def project_bfield_xy(coords_xyz, bfield_xyz, density_code, grid=None):
    """Density-weighted projection of magnetic field components onto the XY plane.

    Parameters
    ----------
    coords_xyz : array-like, shape (N, 3)
        Particle x, y, z positions in code units (bar-frame centred on GC).
    bfield_xyz : array-like, shape (N, 3)
        Bx, By, Bz components in physical units (e.g. µG from ``calc_bfield_uG``).
    density_code : array-like, shape (N,)
        Gas density in code units, used as projection weight.
    grid : dict, optional
        Override keys from ``DEFAULT_XYZ_GRID``.  Only x/y keys are used.

    Returns
    -------
    Bx_proj, By_proj, Bz_proj : ndarray, shape (Ny, Nx)
        Density-weighted mean B components on the 2D XY grid.
    B_mag : ndarray, shape (Ny, Nx)
        Projected field magnitude ``sqrt(Bx² + By²)``.
    xs, ys : ndarray
        Grid centre arrays.
    """
    cfg = DEFAULT_XYZ_GRID.copy()
    if grid:
        cfg.update(grid)

    x, y, _ = np.asarray(coords_xyz).T
    Bx, By, Bz = np.asarray(bfield_xyz).T
    rho = np.asarray(density_code)

    xs, x_edges = _grid_centers_and_edges(cfg["xmin"], cfg["xmax"], cfg["dx"])
    ys, y_edges = _grid_centers_and_edges(cfg["ymin"], cfg["ymax"], cfg["dy"])

    xy_range = [[cfg["xmin"], cfg["xmax"]], [cfg["ymin"], cfg["ymax"]]]
    bins = (len(x_edges), len(y_edges))

    rho_sum = histogram2d(x, y, range=xy_range, bins=bins, weights=rho)
    Bx_sum  = histogram2d(x, y, range=xy_range, bins=bins, weights=rho * Bx)
    By_sum  = histogram2d(x, y, range=xy_range, bins=bins, weights=rho * By)
    Bz_sum  = histogram2d(x, y, range=xy_range, bins=bins, weights=rho * Bz)

    # Avoid divide by zero; empty bins -> 0.
    safe  = rho_sum > 0
    denom = np.where(safe, rho_sum, 1.0)
    Bx_proj = np.where(safe, Bx_sum / denom, 0.0).T
    By_proj = np.where(safe, By_sum / denom, 0.0).T
    Bz_proj = np.where(safe, Bz_sum / denom, 0.0).T
    B_mag   = np.sqrt(Bx_proj**2 + By_proj**2)

    return Bx_proj, By_proj, Bz_proj, B_mag, xs, ys


def project_column_density_xy(coords_xyz, density_code, grid=None):
    """Gas column-density proxy on the same XY grid as ``project_bfield_xy``.

    Each cell contains the sum of ``density_code`` for particles falling in that bin
    (surface-density proxy up to a constant factor).

    Parameters
    ----------
    coords_xyz : array-like, shape (N, 3)
        Particle positions in code units (same frame as ``project_bfield_xy``).
    density_code : array-like, shape (N,)
        Density or mass proxy used as weight.
    grid : dict, optional
        Same keys as ``DEFAULT_XYZ_GRID`` / ``project_bfield_xy``.

    Returns
    -------
    sigma_xy : ndarray, shape (Ny, Nx)
    xs, ys : ndarray
        Grid centres matching ``project_bfield_xy``.
    """
    cfg = DEFAULT_XYZ_GRID.copy()
    if grid:
        cfg.update(grid)

    x, y, _ = np.asarray(coords_xyz).T
    rho = np.asarray(density_code)

    xs, x_edges = _grid_centers_and_edges(cfg["xmin"], cfg["xmax"], cfg["dx"])
    ys, y_edges = _grid_centers_and_edges(cfg["ymin"], cfg["ymax"], cfg["dy"])

    xy_range = [[cfg["xmin"], cfg["xmax"]], [cfg["ymin"], cfg["ymax"]]]
    bins = (len(x_edges), len(y_edges))
    rho_sum = histogram2d(x, y, range=xy_range, bins=bins, weights=rho)
    return rho_sum.T, xs, ys


def project_bfield_lb(x, y, z, Bx, By, Bz, density, grid=None):
    """Project B-field orientation onto the l-b sky plane using Stokes parameters.

    Integrates along lines of sight (l-b bins) using the polarimetric Q/U formalism
    so that cancelling field directions don't spuriously wash each other out — same
    approach as the original Stokes-parameter notebook workflow, applied to the sky projection.

    Parameters
    ----------
    x, y, z : array-like
        Particle positions in bar-frame code units (GC-centred, post rotate_to_bar_frame).
    Bx, By, Bz : array-like
        B components in the same frame (µG).
    density : array-like
        Gas density (code units) used as projection weight.
    grid : dict, optional
        Override keys from ``DEFAULT_LB_BFIELD_GRID`` (l/b in degrees). Smaller
        ``dl`` and ``db`` yield finer angular bins (more pixels; slower LIC).

    Returns
    -------
    ux, uy : ndarray, shape (Nb, Nl)
        Stokes-derived plane-of-sky orientation unit vectors.
    B_perp : ndarray, shape (Nb, Nl)
        Density-weighted mean |B| in the plane of sky per bin (µG).
    sigma : ndarray, shape (Nb, Nl)
        Density sum per bin (surface-density proxy).
    ls, bs : ndarray
        Grid centres in degrees.
    """
    from .projections import xyz_basis_from_sun, GALACTIC_ORIGIN

    cfg = DEFAULT_LB_BFIELD_GRID.copy()
    if grid:
        cfg.update(grid)

    x   = np.asarray(x,       dtype=np.float64)
    y   = np.asarray(y,       dtype=np.float64)
    z   = np.asarray(z,       dtype=np.float64)
    Bx  = np.asarray(Bx,      dtype=np.float64)
    By  = np.asarray(By,      dtype=np.float64)
    Bz  = np.asarray(Bz,      dtype=np.float64)
    rho = np.asarray(density, dtype=np.float64)

    xsun = GALACTIC_ORIGIN["xsun"]
    ysun = GALACTIC_ORIGIN["ysun"]
    zsun = GALACTIC_ORIGIN["zsun"]
    xhat, yhat, zhat = xyz_basis_from_sun(xsun, ysun, zsun)

    # B in heliocentric frame (vector transform, no offset)
    B_X = Bx * xhat[0] + By * xhat[1] + Bz * xhat[2]
    B_Y = Bx * yhat[0] + By * yhat[1] + Bz * yhat[2]
    B_Z = Bx * zhat[0] + By * zhat[1] + Bz * zhat[2]

    # Heliocentric coords to l, b per particle
    dx = x - xsun
    dy = y - ysun
    dz = z - zsun
    X = dx * xhat[0] + dy * xhat[1] + dz * xhat[2]
    Y = dx * yhat[0] + dy * yhat[1] + dz * yhat[2]
    Z = dx * zhat[0] + dy * zhat[1] + dz * zhat[2]
    r = np.sqrt(X**2 + Y**2 + Z**2)
    safe_r = np.where(r > 0, r, 1.0)
    l = np.arctan2(Y, X)
    b = np.pi / 2.0 - np.arccos(np.clip(Z / safe_r, -1.0, 1.0))

    # Sky-plane B: project onto l-hat and b-hat at each particle's (l, b)
    # l_hat = [-sin(l), cos(l), 0]
    # b_hat = [-cos(b)*cos(l), -cos(b)*sin(l), sin(b)]
    B_l = B_X * (-np.sin(l)) + B_Y * np.cos(l)
    B_b = (B_X * (-np.cos(b) * np.cos(l))
           + B_Y * (-np.cos(b) * np.sin(l))
           + B_Z * np.sin(b))

    B_perp_sq = B_l**2 + B_b**2
    safe_b = B_perp_sq > 0
    denom_b = np.where(safe_b, B_perp_sq, 1.0)

    # Stokes weights (normalised so |B| does not bias orientation).
    Q_w    = rho * np.where(safe_b, (B_l**2 - B_b**2) / denom_b, 0.0)
    U_w    = rho * np.where(safe_b, 2.0 * B_l * B_b / denom_b,   0.0)
    Bperp_w = rho * np.sqrt(B_perp_sq)

    l_deg_arr = np.degrees(l)
    b_deg_arr = np.degrees(b)
    ls, l_edges = _grid_centers_and_edges(cfg["lmin"], cfg["lmax"], cfg["dl"])
    bs, b_edges = _grid_centers_and_edges(cfg["bmin"], cfg["bmax"], cfg["db"])
    lb_range = [[cfg["lmin"], cfg["lmax"]], [cfg["bmin"], cfg["bmax"]]]
    bins = (len(l_edges), len(b_edges))

    rho_sum  = histogram2d(l_deg_arr, b_deg_arr, range=lb_range, bins=bins, weights=rho)
    Q_sum    = histogram2d(l_deg_arr, b_deg_arr, range=lb_range, bins=bins, weights=Q_w)
    U_sum    = histogram2d(l_deg_arr, b_deg_arr, range=lb_range, bins=bins, weights=U_w)
    Bp_sum   = histogram2d(l_deg_arr, b_deg_arr, range=lb_range, bins=bins, weights=Bperp_w)

    safe2 = rho_sum > 0
    d2 = np.where(safe2, rho_sum, 1.0)
    Q_map      = np.where(safe2, Q_sum  / d2, 0.0).T
    U_map      = np.where(safe2, U_sum  / d2, 0.0).T
    B_perp_map = np.where(safe2, Bp_sum / d2, 0.0).T
    sigma_map  = rho_sum.T

    psi = 0.5 * np.arctan2(U_map, Q_map)
    ux  = np.cos(psi)
    uy  = np.sin(psi)

    return ux, uy, B_perp_map, sigma_map, ls, bs


def project_bfield_plane(x, y, z, Bx, By, Bz, density, plane='xy', grid=None):
    """Project B-field orientation onto a Cartesian plane using Stokes parameters.

    Works for any of the three principal planes. Uses the same per-particle
    Q/U formalism as ``project_bfield_lb`` — field reversals along the line of
    sight cancel rather than averaging to zero.

    Parameters
    ----------
    x, y, z : array-like
        Positions (code units, GC-centred).
    Bx, By, Bz : array-like
        B-field components (µG, same frame as positions).
    density : array-like
        Projection weight (gas density, code units).
    plane : {'xy', 'xz', 'yz'}
        Projection plane.
    grid : dict, optional
        Grid keys ``{ax1}min``, ``{ax1}max``, ``d{ax1}``, ``{ax2}min``, ``{ax2}max``,
        ``d{ax2}`` where ``ax1``, ``ax2`` are the two in-plane axes. Defaults fall
        back to ``DEFAULT_XYZ_GRID``.

    Returns
    -------
    ux, uy : ndarray, shape (N2, N1)
        Stokes-derived unit orientation vectors (N1 = first-axis bins, N2 = second).
    B_perp : ndarray, shape (N2, N1)
        Density-weighted mean in-plane |B| per bin (µG).
    sigma : ndarray, shape (N2, N1)
        Density sum per bin (surface-density proxy).
    ax1s, ax2s : ndarray
        Grid centres along the first and second in-plane axes.
    """
    plane = plane.lower()
    if plane == 'xy':
        h1, h2 = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        B1, B2 = np.asarray(Bx, dtype=np.float64), np.asarray(By, dtype=np.float64)
        k1, k2 = 'x', 'y'
    elif plane == 'xz':
        h1, h2 = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)
        B1, B2 = np.asarray(Bx, dtype=np.float64), np.asarray(Bz, dtype=np.float64)
        k1, k2 = 'x', 'z'
    elif plane == 'yz':
        h1, h2 = np.asarray(y, dtype=np.float64), np.asarray(z, dtype=np.float64)
        B1, B2 = np.asarray(By, dtype=np.float64), np.asarray(Bz, dtype=np.float64)
        k1, k2 = 'y', 'z'
    else:
        raise ValueError(f"plane must be 'xy', 'xz', or 'yz'; got {plane!r}")

    rho = np.asarray(density, dtype=np.float64)

    cfg = {
        f'{k1}min': DEFAULT_XYZ_GRID[f'{k1}min'],
        f'{k1}max': DEFAULT_XYZ_GRID[f'{k1}max'],
        f'd{k1}':   DEFAULT_XYZ_GRID[f'd{k1}'],
        f'{k2}min': DEFAULT_XYZ_GRID[f'{k2}min'],
        f'{k2}max': DEFAULT_XYZ_GRID[f'{k2}max'],
        f'd{k2}':   DEFAULT_XYZ_GRID[f'd{k2}'],
    }
    if grid:
        cfg.update(grid)

    B_perp_sq = B1**2 + B2**2
    safe_b    = B_perp_sq > 0
    denom_b   = np.where(safe_b, B_perp_sq, 1.0)

    Q_w     = rho * np.where(safe_b, (B1**2 - B2**2) / denom_b, 0.0)
    U_w     = rho * np.where(safe_b, 2.0 * B1 * B2 / denom_b,   0.0)
    Bperp_w = rho * np.sqrt(B_perp_sq)

    ax1s, ax1_edges = _grid_centers_and_edges(cfg[f'{k1}min'], cfg[f'{k1}max'], cfg[f'd{k1}'])
    ax2s, ax2_edges = _grid_centers_and_edges(cfg[f'{k2}min'], cfg[f'{k2}max'], cfg[f'd{k2}'])
    h_range = [[cfg[f'{k1}min'], cfg[f'{k1}max']], [cfg[f'{k2}min'], cfg[f'{k2}max']]]
    bins    = (len(ax1_edges), len(ax2_edges))

    rho_sum  = histogram2d(h1, h2, range=h_range, bins=bins, weights=rho)
    Q_sum    = histogram2d(h1, h2, range=h_range, bins=bins, weights=Q_w)
    U_sum    = histogram2d(h1, h2, range=h_range, bins=bins, weights=U_w)
    Bp_sum   = histogram2d(h1, h2, range=h_range, bins=bins, weights=Bperp_w)

    safe2 = rho_sum > 0
    d2    = np.where(safe2, rho_sum, 1.0)
    Q_map      = np.where(safe2, Q_sum  / d2, 0.0).T
    U_map      = np.where(safe2, U_sum  / d2, 0.0).T
    B_perp_map = np.where(safe2, Bp_sum / d2, 0.0).T
    sigma_map  = rho_sum.T

    psi = 0.5 * np.arctan2(U_map, Q_map)
    ux  = np.cos(psi)
    uy  = np.sin(psi)

    return ux, uy, B_perp_map, sigma_map, ax1s, ax2s


def _colorbar_horizontal_above_ax(fig, ax, mappable, cmap, norm, colorbar_label):
    """Add a horizontal colorbar above *ax* with ticks/label on top (shared layout)."""
    import matplotlib.colors as mcolors
    import matplotlib.ticker as ticker

    pos = ax.get_position()
    cb_w = pos.width * 0.55
    cb_h = 0.012
    cb_x = pos.x0 + (pos.width - cb_w) * 0.5
    cb_y = min(pos.y1 + 0.045, 0.965 - cb_h)
    cb_ax = fig.add_axes([cb_x, cb_y, cb_w, cb_h])
    cb = fig.colorbar(mappable, cax=cb_ax, orientation="horizontal", label=colorbar_label)
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(axis="x", pad=1)
    # Log-style decade ticks for LogNorm and BoundaryNorm (e.g. log-spaced levels).
    use_log_ticks = isinstance(norm, mcolors.LogNorm) or (
        hasattr(norm, "boundaries")
        and norm.boundaries is not None
        and float(np.min(norm.boundaries)) > 0
    )
    if not use_log_ticks:
        return
    if hasattr(norm, "boundaries") and norm.boundaries is not None:
        vmin = float(np.min(norm.boundaries))
        vmax = float(np.max(norm.boundaries))
    else:
        vmin = float(getattr(norm, "vmin", np.nan))
        vmax = float(getattr(norm, "vmax", np.nan))
    if np.isfinite(vmin) and np.isfinite(vmax) and vmin > 0 and vmax > vmin:
        pmin = int(np.ceil(np.log10(vmin)))
        pmax = int(np.floor(np.log10(vmax)))
        if pmax >= pmin:
            ticks = np.logspace(pmin, pmax, pmax - pmin + 1)
        else:
            ticks = np.array([np.sqrt(vmin * vmax)])
        cb.set_ticks(ticks)
        cb.ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))


def plot_threepanel_bfield(
    x, y, z, Bx, By, Bz, density,
    xlim=(-5.0, 5.0),
    ylim=(-5.0, 5.0),
    zlim=(-2.0, 2.0),
    dpix=0.01,
    dpix_z=None,
    norm=None,
    norm_lim=(-0.5, 3.8),
    cmap="RdYlBu_r",
    lic_kernels=(21, 51),
    seed=None,
    colorbar_label=r"$\Sigma\,[\mathrm{M_\odot\,pc^{-2}}]$",
    scalebar_length=None,
    scalebar_label=None,
    smooth_sigma_px=0.0,
    interpolation="nearest",
    projection_method="nearest_nd",
):
    """Three-panel B-field orientation figure matching the reference layout.

    Face-on XY (top-left, large) + XZ edge-on (bottom strip) + YZ edge-on (right
    strip). Panel sizes are proportional to physical extents so the coordinate axes
    align across panels. Background is gas column density; texture is LIC along
    the Stokes-derived orientation.

    Parameters
    ----------
    x, y, z : array-like
        Positions (code units, GC-centred, after ``rotate_to_bar_frame``).
    Bx, By, Bz : array-like
        B-field in the same frame (µG).
    density : array-like
        Gas density (code units), used as projection weight.
    xlim, ylim, zlim : (float, float)
        Spatial bounds of the region to plot (code units).
    dpix : float
        Pixel size in x/y (code units). Default 0.01 (1 pc pixels), matching
        the high-resolution reference setup over the default extents.
    dpix_z : float, optional
        Pixel size in z. Defaults to ``dpix``.
    norm : matplotlib.colors.Normalize, optional
        Colormap normalization. Overrides ``norm_lim``.
    norm_lim : (float, float), optional
        ``(log10_vmin, log10_vmax)`` for the ``BoundaryNorm`` auto-norm.
        Default ``(-0.5, 4.5)`` matches the reference ``logspace(-0.5, 4.5, 256)``.
    cmap : str
    lic_kernels : (int, int)
        LIC kernel lengths for the two-pass convolution.
    seed : int, optional
        RNG seed for reproducible textures.
    colorbar_label : str or None
    scalebar_length : float, optional
        Length of a scalebar to draw on the XY panel (code units).
    scalebar_label : str, optional
        Label beside the scalebar.

    Returns
    -------
    fig : Figure
    axes : (ax_xy, ax_xz, ax_yz)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from matplotlib import gridspec

    if dpix_z is None:
        dpix_z = dpix

    xmin, xmax = xlim
    ymin, ymax = ylim
    zmin, zmax = zlim

    # Clip to the 3D box (same slab logic as the reference voxel grid).
    # Each panel integrates only gas inside the corresponding slab (e.g. XY uses zlim).
    _x = np.asarray(x, dtype=np.float64)
    _y = np.asarray(y, dtype=np.float64)
    _z = np.asarray(z, dtype=np.float64)
    _Bx = np.asarray(Bx, dtype=np.float64)
    _By = np.asarray(By, dtype=np.float64)
    _Bz = np.asarray(Bz, dtype=np.float64)
    _rho = np.asarray(density, dtype=np.float64)
    box = ((_x >= xmin) & (_x <= xmax) &
           (_y >= ymin) & (_y <= ymax) &
           (_z >= zmin) & (_z <= zmax))
    _x, _y, _z   = _x[box],  _y[box],  _z[box]
    _Bx, _By, _Bz = _Bx[box], _By[box], _Bz[box]
    _rho = _rho[box]

    grid_xy = {'xmin': xmin, 'xmax': xmax, 'dx': dpix,
               'ymin': ymin, 'ymax': ymax, 'dy': dpix}
    grid_xz = {'xmin': xmin, 'xmax': xmax, 'dx': dpix,
               'zmin': zmin, 'zmax': zmax, 'dz': dpix_z}
    grid_yz = {'ymin': ymin, 'ymax': ymax, 'dy': dpix,
               'zmin': zmin, 'zmax': zmax, 'dz': dpix_z}

    if projection_method == "histogram":
        ux_xy, uy_xy, _, sigma_xy, xs, ys = project_bfield_plane(
            _x, _y, _z, _Bx, _By, _Bz, _rho, plane='xy', grid=grid_xy)
        ux_xz, uy_xz, _, sigma_xz, xs2, zs = project_bfield_plane(
            _x, _y, _z, _Bx, _By, _Bz, _rho, plane='xz', grid=grid_xz)
        ux_yz, uy_yz, _, sigma_yz, ys2, zs2 = project_bfield_plane(
            _x, _y, _z, _Bx, _By, _Bz, _rho, plane='yz', grid=grid_yz)
    elif projection_method == "nearest_nd":
        try:
            from scipy.interpolate import NearestNDInterpolator
        except Exception as exc:
            raise ImportError(
                "projection_method='nearest_nd' requires scipy (NearestNDInterpolator)."
            ) from exc

        nx = max(2, int(np.round((xmax - xmin) / dpix)))
        ny = max(2, int(np.round((ymax - ymin) / dpix)))
        nz = max(2, int(np.round((zmax - zmin) / dpix_z)))

        XGRID, YGRID, ZGRID = np.mgrid[
            xmin:xmax:(nx * 1j),
            ymin:ymax:(ny * 1j),
            zmin:zmax:(nz * 1j),
        ]
        points = np.column_stack([_x, _y, _z])

        rho_grid = NearestNDInterpolator(points, _rho)(XGRID, YGRID, ZGRID)
        bx_grid = NearestNDInterpolator(points, _Bx)(XGRID, YGRID, ZGRID)
        by_grid = NearestNDInterpolator(points, _By)(XGRID, YGRID, ZGRID)
        bz_grid = NearestNDInterpolator(points, _Bz)(XGRID, YGRID, ZGRID)

        # Column/surface density proxies (before physical-unit conversion below).
        sigma_xy = np.rot90(rho_grid.sum(axis=2))
        sigma_xz = np.rot90(rho_grid.sum(axis=1))
        sigma_yz = np.rot90(rho_grid.sum(axis=0))

        # Density-weighted projected components.
        d_xy = np.where(rho_grid.sum(axis=2) > 0, rho_grid.sum(axis=2), 1.0)
        d_xz = np.where(rho_grid.sum(axis=1) > 0, rho_grid.sum(axis=1), 1.0)
        d_yz = np.where(rho_grid.sum(axis=0) > 0, rho_grid.sum(axis=0), 1.0)
        bx_xy = np.rot90((bx_grid * rho_grid).sum(axis=2) / d_xy)
        by_xy = np.rot90((by_grid * rho_grid).sum(axis=2) / d_xy)
        by_xz = np.rot90((by_grid * rho_grid).sum(axis=1) / d_xz)
        bz_xz = np.rot90((bz_grid * rho_grid).sum(axis=1) / d_xz)
        bx_yz = np.rot90((bx_grid * rho_grid).sum(axis=0) / d_yz)
        bz_yz = np.rot90((bz_grid * rho_grid).sum(axis=0) / d_yz)

        # Headless orientation (same Stokes-angle relation used elsewhere).
        psi_xy = 0.5 * np.arctan2(2.0 * bx_xy * by_xy, bx_xy**2 - by_xy**2)
        psi_xz = 0.5 * np.arctan2(2.0 * by_xz * bz_xz, by_xz**2 - bz_xz**2)
        psi_yz = 0.5 * np.arctan2(2.0 * bx_yz * bz_yz, bx_yz**2 - bz_yz**2)
        ux_xy, uy_xy = np.cos(psi_xy), np.sin(psi_xy)
        ux_xz, uy_xz = np.cos(psi_xz), np.sin(psi_xz)
        ux_yz, uy_yz = np.cos(psi_yz), np.sin(psi_yz)
    else:
        raise ValueError("projection_method must be 'nearest_nd' or 'histogram'")

    # Convert from code density sum to physical surface density [Msun / pc^2],
    # matching the reference pixel-depth scaling.
    # density is in Msun/(100 pc)^3 and code lengths are in 100 pc.
    sigma_xy = sigma_xy * dpix_z / 1.0e4
    sigma_xz = sigma_xz * dpix   / 1.0e4
    sigma_yz = sigma_yz * dpix   / 1.0e4

    if smooth_sigma_px and smooth_sigma_px > 0:
        try:
            from scipy.ndimage import gaussian_filter
            sigma_xy = gaussian_filter(sigma_xy, smooth_sigma_px)
            sigma_xz = gaussian_filter(sigma_xz, smooth_sigma_px)
            sigma_yz = gaussian_filter(sigma_yz, smooth_sigma_px)
        except Exception:
            pass

    if norm is None:
        pos = sigma_xy[sigma_xy > 0]
        if pos.size > 0:
            if norm_lim is not None:
                rhomin, rhomax = norm_lim
            else:
                log_med = float(np.log10(np.median(pos)))
                rhomin  = log_med - 2.5
                rhomax  = log_med + 2.5
            levels = np.logspace(rhomin, rhomax, 256)
            norm = colors.BoundaryNorm(levels, 256)
        else:
            norm = colors.Normalize(vmin=0.0, vmax=1.0)

    rng    = np.random.default_rng(seed)
    mod_xy = lic_modulation_planck_style(ux_xy, uy_xy, kernels=lic_kernels, rng=rng)
    mod_xz = lic_modulation_planck_style(ux_xz, uy_xz, kernels=lic_kernels, rng=rng)
    mod_yz = lic_modulation_planck_style(ux_yz, uy_yz, kernels=lic_kernels, rng=rng)

    rgba_xy = rgba_intensity_with_lic_striations(sigma_xy, mod_xy, norm, cmap)
    rgba_xz = rgba_intensity_with_lic_striations(sigma_xz, mod_xz, norm, cmap)
    rgba_yz = rgba_intensity_with_lic_striations(sigma_yz, mod_yz, norm, cmap)

    DX = xmax - xmin
    DY = ymax - ymin
    DZ = zmax - zmin

    # Figure size scaled to physical aspect ratio
    total_w = DX + DZ
    total_h = DY + DZ
    base_in = 7.0
    fig = plt.figure(figsize=(base_in * total_w / max(total_w, total_h),
                               base_in * total_h / max(total_w, total_h)))
    gs  = gridspec.GridSpec(2, 2, wspace=0., hspace=0.,
                            height_ratios=[DY, DZ], width_ratios=[DX, DZ])
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[1, 0])
    ax_yz = fig.add_subplot(gs[0, 1])

    ext_xy = (xmin, xmax, ymin, ymax)
    ext_xz = (xmin, xmax, zmin, zmax)
    ext_yz = (zmin, zmax, ymin, ymax)

    ax_xy.imshow(rgba_xy, origin='lower', extent=ext_xy, aspect='auto', interpolation=interpolation)
    ax_xz.imshow(rgba_xz, origin='lower', extent=ext_xz, aspect='auto', interpolation=interpolation)
    # rot90 so y is vertical and z is horizontal in the right-hand panel
    ax_yz.imshow(np.rot90(rgba_yz), origin='lower', extent=ext_yz, aspect='auto', interpolation=interpolation)

    ax_xy.set_ylabel(r"$y\,[\mathrm{100\,pc}]$", fontsize=8)
    ax_xz.set_xlabel(r"$x\,[\mathrm{100\,pc}]$", fontsize=8)
    ax_xz.set_ylabel(r"$z\,[\mathrm{100\,pc}]$", fontsize=8)
    ax_yz.set_xlabel(r"$z\,[\mathrm{100\,pc}]$", fontsize=8)
    ax_xy.tick_params(labelbottom=False)
    ax_yz.tick_params(labelleft=False, labelbottom=False)

    if scalebar_length is not None:
        y0 = ymin + 0.05 * DY
        x0 = xmin + 0.05 * DX
        ax_xy.plot([x0, x0 + scalebar_length], [y0, y0], 'w-', lw=2)
        if scalebar_label:
            ax_xy.text(x0 + scalebar_length / 2, y0 + 0.02 * DY,
                       scalebar_label, color='w', ha='center', va='bottom', fontsize=7)

    if colorbar_label is not None:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(sigma_xy)
        _colorbar_horizontal_above_ax(fig, ax_xy, mappable, cmap, norm, colorbar_label)

    return fig, (ax_xy, ax_xz, ax_yz)


def plot_bfield_xy(B_mag, Bx_proj, By_proj, xs, ys, ax=None,
                   vmin=1e-2, vmax=1e3, cmap=None,
                   show_orientation=True, orientation_stride=30):
    """Plot projected B field magnitude with optional field-orientation bars.

    Orientation bars are headless (``pivot='middle'``, no arrowhead) to show
    field direction without implying a sense — consistent with polarimetry
    convention used in astrophysics.

    Parameters
    ----------
    B_mag : ndarray
        2D projected field magnitude from ``project_bfield_xy``.
    Bx_proj, By_proj : ndarray
        Density-weighted mean B components from ``project_bfield_xy``.
    xs, ys : ndarray
        Grid centre arrays from ``project_bfield_xy``.
    ax : Axes, optional
        Existing axes.  A new figure is created when omitted.
    vmin, vmax : float, optional
        Log normalization range in the same units as ``B_mag``; default ``(1e-2, 1e3)`` µG.
    cmap : colormap, optional
        Defaults to ``cmasher.ember`` if available, else ``"plasma"``.
    show_orientation : bool, optional
        Overlay headless bars showing projected field orientation; default True.
    orientation_stride : int, optional
        Subsampling stride on the grid for orientation bars; default 30.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    if cmap is None:
        try:
            import cmasher as cmr
            cmap = cmr.ember
        except ImportError:
            cmap = "plasma"

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    else:
        fig = ax.figure

    norm   = colors.LogNorm(vmin=vmin, vmax=vmax)
    extent = (xs.min(), xs.max(), ys.min(), ys.max())
    im = ax.imshow(B_mag, origin="lower", extent=extent, cmap=cmap, norm=norm, aspect="equal")

    if show_orientation:
        s   = orientation_stride
        xq  = xs[::s]
        yq  = ys[::s]
        XQ, YQ = np.meshgrid(xq, yq)
        Bxq = Bx_proj[::s, ::s]
        Byq = By_proj[::s, ::s]
        Bmag_q = np.sqrt(Bxq**2 + Byq**2)
        safe   = Bmag_q > 0
        ux = np.where(safe, Bxq / np.where(safe, Bmag_q, 1.0), 0.0)
        uy = np.where(safe, Byq / np.where(safe, Bmag_q, 1.0), 0.0)
        ax.quiver(
            XQ, YQ, ux, uy,
            color="white", alpha=0.55, pivot="middle", scale=40,
            headwidth=0, headlength=0, headaxislength=0,
        )

    ax.set_xlabel(r"$x\,[\mathrm{100\,pc}]$")
    ax.set_ylabel(r"$y\,[\mathrm{100\,pc}]$")
    fig.colorbar(im, ax=ax, label=r"$|B|\,[\mu\mathrm{G}]$", pad=0.02)

    return fig, ax


# LIC + HSV striation (ported from the Stokes-parameter reference notebook).
# Pure Python: keep grid sizes moderate, or use a compiled LIC path for large maps.


def _dda_lic_step(vx, vy, x, y, fx, fy, w, h):
    """One LIC streamline DDA step (advance to next cell boundary)."""
    if vx >= 0:
        tx = (1.0 - fx) / vx if abs(vx) > 1e-15 else float("inf")
    else:
        tx = (-fx) / vx if abs(vx) > 1e-15 else float("inf")
    if vy >= 0:
        ty = (1.0 - fy) / vy if abs(vy) > 1e-15 else float("inf")
    else:
        ty = (-fy) / vy if abs(vy) > 1e-15 else float("inf")
    if tx < ty:
        if vx >= 0:
            x = x + 1
            fx = 0.0
        else:
            x = x - 1
            fx = 1.0
        fy = fy + tx * vy
    else:
        if vy >= 0:
            y = y + 1
            fy = 0.0
        else:
            y = y - 1
            fy = 1.0
        fx = fx + ty * vx
    if x >= w:
        x = w - 1
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if y >= h:
        y = h - 1
    return x, y, fx, fy


def line_integral_convolution(vectors, texture, kernel):
    """Line integral convolution (LIC) on a 2D vector field.

    ``vectors`` shape ``(h, w, 2)``: ``[..., 0]`` is x, ``[..., 1]`` is y.
    ``texture`` shape ``(w, h)`` — note column-major indexing ``texture[x, y]``.

    Parameters
    ----------
    vectors : ndarray, shape (h, w, 2), float32
        Direction field (not necessarily unit-length).
    texture : ndarray, shape (w, h), float32
        Noise texture.
    kernel : 1d ndarray
        Convolution kernel along streamlines.

    Returns
    -------
    ndarray, shape (h, w)
        LIC intensity image.

    Notes
    -----
    Runtime scales as ``h * w * len(kernel)`` — downsample large maps for interactive use.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    texture = np.asarray(texture, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    h, w, t = vectors.shape
    if t != 2:
        raise ValueError("vectors must have shape (h, w, 2)")
    kernellen = kernel.shape[0]
    result = np.zeros((h, w), dtype=np.float32)
    kh = kernellen // 2

    for i in range(h):
        for j in range(w):
            x = j
            y = i
            fx = 0.5
            fy = 0.5
            k = kh
            result[i, j] += kernel[k] * texture[x, y]
            while k < kernellen - 1:
                vx = float(vectors[y, x, 0])
                vy = float(vectors[y, x, 1])
                x, y, fx, fy = _dda_lic_step(vx, vy, x, y, fx, fy, w, h)
                k += 1
                result[i, j] += kernel[k] * texture[x, y]

            x = j
            y = i
            fx = 0.5
            fy = 0.5
            while k > 0:
                vx = float(vectors[y, x, 0])
                vy = float(vectors[y, x, 1])
                x, y, fx, fy = _dda_lic_step(-vx, -vy, x, y, fx, fy, w, h)
                k -= 1
                result[i, j] += kernel[k] * texture[x, y]

    return result


def histogram_equalize(image, n_bins=256):
    """Histogram-equalize a 2D array (flattened), preserving shape."""
    flat = np.asarray(image).ravel()
    hist, bins = np.histogram(flat, n_bins)
    cdf = np.cumsum(hist).astype(np.float64)
    if cdf[-1] <= 0:
        return np.asarray(image).copy()
    cdf = cdf / cdf[-1]
    out = np.interp(flat, bins[:-1], cdf)
    return out.reshape(image.shape)


def lic_modulation_planck_style(ux, uy, kernels=(21, 51), rng=None):
    """Two-pass LIC + histogram equalization (reference notebook workflow).

    Parameters
    ----------
    ux, uy : ndarray, shape (Ny, Nx)
        Projected unit-vector components (e.g. Bx/|B|, By/|B| in the map plane).
    kernels : tuple of int
        Odd-length sine kernels for pass 1 and pass 2.
    rng : numpy.random.Generator, optional
        Random texture seed.

    Returns
    -------
    ndarray, shape (Ny, Nx)
        Striation modulation in ``[0, 1]`` after clipping internal range.
    """
    rng = rng if rng is not None else np.random.default_rng()
    ux = np.asarray(ux, dtype=np.float32)
    uy = np.asarray(uy, dtype=np.float32)
    h, w = ux.shape
    mag = np.sqrt(ux * ux + uy * uy)
    safe = mag > 1e-15
    ux = np.where(safe, ux / np.where(safe, mag, 1.0), 1.0)
    uy = np.where(safe, uy / np.where(safe, mag, 1.0), 0.0)
    vectors = np.zeros((h, w, 2), dtype=np.float32)
    vectors[..., 0] = ux
    vectors[..., 1] = uy

    texture = rng.random((w, h), dtype=np.float32)
    k0 = np.sin(np.arange(kernels[0]) * np.pi / kernels[0]).astype(np.float32)
    lic1 = line_integral_convolution(vectors, texture, k0)
    eq1 = histogram_equalize(lic1)

    k1 = np.sin(np.arange(kernels[1]) * np.pi / kernels[1]).astype(np.float32)
    lic2 = line_integral_convolution(vectors, eq1.T.astype(np.float32), k1)
    eq2 = histogram_equalize(lic2)

    mi, ma = 0.2, 1.0
    mod = (np.clip(eq2, mi, ma) - mi) / (ma - mi)
    return mod.astype(np.float64)


def rgba_intensity_with_lic_striations(
    intensity,
    lic_modulation,
    norm,
    cmap,
    saturation_scale=0.9,
    rgb_dimming=0.08,
):
    """Combine a scalar background image with LIC striations via HSV (Planck-style).

    Parameters
    ----------
    intensity : ndarray, shape (Ny, Nx)
        Physical scalar (e.g. Σ_gas or |B|).
    lic_modulation : ndarray, shape (Ny, Nx)
        Values in ``[0, 1]`` from ``lic_modulation_planck_style``.
    norm : matplotlib.colors.Normalize
        Normalization for ``intensity``.
    cmap : str or Colormap
        Base colormap for the scalar field.
    saturation_scale : float
        Weight on modulation for saturation boost (reference default 0.9).
    rgb_dimming : float
        Per-channel darkening along striations (reference default 0.08).

    Returns
    -------
    ndarray, shape (Ny, Nx, 4)
        RGBA image suitable for ``imshow``.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    intensity = np.asarray(intensity)
    lic_modulation = np.asarray(lic_modulation)
    if intensity.shape != lic_modulation.shape:
        raise ValueError("intensity and lic_modulation must have the same shape")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba = sm.to_rgba(intensity)
    rgb = rgba[..., :3]
    hsv = mcolors.rgb_to_hsv(rgb)
    hsv[..., 1] = np.clip(hsv[..., 1] * (saturation_scale * lic_modulation + 1.0), 0.0, 1.0)
    rgb_out = mcolors.hsv_to_rgb(hsv)
    dim = 1.0 - lic_modulation * rgb_dimming
    rgb_out[..., 0] *= dim
    rgb_out[..., 1] *= dim
    rgb_out[..., 2] *= dim
    out = np.zeros(intensity.shape + (4,), dtype=np.float64)
    out[..., :3] = rgb_out
    out[..., 3] = 1.0
    return out


def plot_planck_style_bfield(
    intensity,
    ux,
    uy,
    xs,
    ys,
    ax=None,
    figsize=(7.0, 7.0),
    norm=None,
    norm_lim=None,
    cmap="RdYlBu_r",
    lic_kernels=(21, 51),
    seed=None,
    saturation_scale=0.9,
    rgb_dimming=0.08,
    colorbar_label=r"$|B|\,[\mu\mathrm{G}]$",
    interpolation="nearest",
    smooth_sigma_px=0.0,
):
    """Plot scalar field with LIC orientation striations (Planck / polarization style).

    Uses density- or field-strength ``intensity`` as the colored background and the
    in-plane unit vectors ``(ux, uy)`` to generate streamline textures.

    Parameters
    ----------
    intensity : ndarray, shape (Ny, Nx)
        e.g. gas surface density or ``|B|``.
    ux, uy : ndarray, shape (Ny, Nx)
        Normalized in-plane direction (``Bx / sqrt(Bx²+By²)``, etc.).
    xs, ys : ndarray
        Axis centres from ``project_bfield_xy`` (used for ``extent``).
    ax : Axes, optional
    figsize : (float, float)
        Figure size in inches when ``ax`` is ``None`` (default ``(7, 7)``).
    norm : matplotlib.colors.Normalize, optional
        Overrides ``norm_lim`` when given.
    norm_lim : (float, float) or None, optional
        ``(log10 vmin, log10 vmax)`` for ``LogNorm`` on positive ``intensity``.
        Ignored if ``norm`` is passed.
    cmap : colormap name or Colormap
    lic_kernels : tuple of int
        LIC kernel lengths (odd recommended).
    seed : int, optional
        RNG seed for reproducible textures.
    saturation_scale, rgb_dimming : float
        Striation strength (reference notebook defaults).
    colorbar_label : str or None
        Scalar colorbar label; ``None`` hides the colorbar.
    interpolation : str
        Passed to ``imshow`` (e.g. ``\"nearest\"``, ``\"bicubic\"``).
    smooth_sigma_px : float
        Optional Gaussian blur sigma (pixels) on ``intensity`` before LIC.

    Returns
    -------
    fig, ax, rgba : Figure, Axes, ndarray
        RGBA image array used for ``imshow``.
    """
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    intensity = np.asarray(intensity, dtype=np.float64)
    if smooth_sigma_px and smooth_sigma_px > 0:
        try:
            from scipy.ndimage import gaussian_filter

            intensity = gaussian_filter(intensity, smooth_sigma_px)
        except Exception:
            pass

    if norm is None:
        if norm_lim is not None:
            lo, hi = norm_lim
            norm = colors.LogNorm(vmin=10.0**lo, vmax=10.0**hi)
        else:
            pos = intensity[np.isfinite(intensity) & (intensity > 0)]
            if pos.size > 0:
                vmin = float(np.nanpercentile(pos, 5))
                vmax = float(np.nanpercentile(pos, 99))
                if vmin <= 0:
                    vmin = max(float(np.nanmin(pos)), 1e-30)
                if vmax <= vmin:
                    vmax = vmin * 10.0
                norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = colors.Normalize(
                    vmin=float(np.nanmin(intensity)),
                    vmax=float(np.nanmax(intensity)),
                )

    rng = np.random.default_rng(seed)
    mod = lic_modulation_planck_style(ux, uy, kernels=lic_kernels, rng=rng)
    rgba = rgba_intensity_with_lic_striations(
        intensity,
        mod,
        norm,
        cmap,
        saturation_scale=saturation_scale,
        rgb_dimming=rgb_dimming,
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    extent = (xs.min(), xs.max(), ys.min(), ys.max())
    ax.imshow(
        rgba,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation=interpolation,
    )
    if colorbar_label is not None:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(intensity)
        _colorbar_horizontal_above_ax(fig, ax, mappable, cmap, norm, colorbar_label)
    ax.set_xlabel(r"$x\,[\mathrm{100\,pc}]$")
    ax.set_ylabel(r"$y\,[\mathrm{100\,pc}]$")
    return fig, ax, rgba


def plot_lb_lv(lb_map, lv_map, ls, bs, vs, axes=None, vmin=1e-7, vmax=1e-3, cmap=None,
               b_lim=(-1.0, 1.0)):
    """Plot l-b and l-v histogram maps sharing the longitude axis.

    Parameters
    ----------
    lb_map, lv_map : ndarray
        2D maps from ``create_lbrv_maps``.
    ls, bs, vs : ndarray
        Longitude, latitude, and velocity center arrays from ``create_lbrv_maps``.
    axes : array-like of two Axes, optional
        Existing axes ``(ax_lv, ax_lb)``.  A new figure is created when omitted.
    vmin, vmax : float, optional
        Colormap boundary norm range; default ``(1e-7, 1e-3)``.
    cmap : colormap, optional
        Matplotlib colormap.  Falls back to ``cmasher.bubblegum`` if available,
        otherwise ``"viridis"``.
    b_lim : tuple of float, optional
        Latitude display range in degrees; default ``(-0.85, 0.85)`` matching the
        movie-frame scripts.  Does not affect binning — only what is shown.

    Returns
    -------
    fig : Figure
    axes : tuple of Axes
        ``(ax_lv, ax_lb)`` — velocity panel first, latitude panel second.
    """
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    if cmap is None:
        try:
            import cmasher as cmr
            cmap = cmr.bubblegum
        except ImportError:
            cmap = "viridis"

    if axes is None:
        figwidth = 10.0
        l_range = ls.max() - ls.min()
        b_disp  = b_lim[1] - b_lim[0]
        # lb strip height ~ proportional in deg; 0.82 leaves room for colorbar/margins.
        lb_h = figwidth * 0.82 * (b_disp / l_range)
        lv_h = max(lb_h * 1.5, 3.5)
        fig, axes = plt.subplots(
            2, 1,
            figsize=(figwidth, lv_h + lb_h + 0.8),
            sharex=True,
            gridspec_kw={"height_ratios": [lv_h, lb_h], "hspace": 0.12},
        )
    else:
        fig = axes[0].figure

    ax_lv, ax_lb = axes

    # Clip b_lim to data so a tight custom grid does not leave blank latitude bands.
    b_lim = (max(b_lim[0], float(bs.min())), min(b_lim[1], float(bs.max())))

    # take log10, flooring zeros/negatives at vmin so the colormap stays finite
    vmin_log = np.log10(vmin)
    vmax_log = np.log10(vmax)
    log_lv = np.log10(np.clip(lv_map, vmin, None))
    log_lb = np.log10(np.clip(lb_map, vmin, None))
    norm = colors.Normalize(vmin=vmin_log, vmax=vmax_log)

    # integer decade positions for clean tick labels
    cb_ticks = list(range(int(np.ceil(vmin_log)), int(np.floor(vmax_log)) + 1))

    extent_lv = (ls.min(), ls.max(), vs.min(), vs.max())
    extent_lb = (ls.min(), ls.max(), bs.min(), bs.max())

    im_lv = ax_lv.imshow(log_lv, origin="lower", extent=extent_lv, cmap=cmap, norm=norm, aspect="auto")
    im_lb = ax_lb.imshow(log_lb, origin="lower", extent=extent_lb, cmap=cmap, norm=norm, aspect="auto")

    # galactic longitude increases to the left; b clipped to display range
    ax_lb.set_xlim(ls.max(), ls.min())
    ax_lb.set_ylim(*b_lim)

    ax_lv.set_ylabel(r"$v_r\,[\mathrm{km\,s^{-1}}]$")
    ax_lb.set_xlabel(r"$l\,[\mathrm{deg}]$")
    ax_lb.set_ylabel(r"$b\,[\mathrm{deg}]$")

    cb_label = r"$\log_{10}(m_{\rm CO}\,/\,r^2)$"
    fig.colorbar(im_lv, ax=ax_lv, label=cb_label, ticks=cb_ticks, pad=0.02)
    fig.colorbar(im_lb, ax=ax_lb, label=cb_label, ticks=cb_ticks, pad=0.02)

    return fig, axes
