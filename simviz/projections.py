"""Coordinate transforms: GC frame ↔ bar frame ↔ heliocentric galactic (l, b, r)."""

import numpy as np


GALACTIC_ORIGIN = {
    "xsun": -80.0,
    "ysun": 0.0,
    "zsun": 0.0,
    "vxsun": 0.0,
    "vysun": 220.0,
    "vzsun": 0.0,
}


def rotate_xy(x, y, theta):
    """Rotate x and y coordinates by an angle.

    Parameters
    ----------
    x, y : array-like
        Cartesian coordinates.
    theta : float
        Rotation angle in radians.

    Returns
    -------
    xprime, yprime : ndarray
        Rotated coordinates.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    xprime = x * np.cos(theta) - y * np.sin(theta)
    yprime = x * np.sin(theta) + y * np.cos(theta)
    return xprime, yprime


def xyz_basis_from_sun(xsun, ysun, zsun):
    """Return heliocentric basis vectors in the simulation frame."""
    rsun = np.sqrt(xsun**2 + ysun**2 + zsun**2)
    rsun_xy = np.sqrt(xsun**2 + ysun**2)

    sintheta = rsun_xy / rsun
    costheta = zsun / rsun
    sinphi = ysun / rsun_xy
    cosphi = xsun / rsun_xy

    xhat = np.array([-cosphi * sintheta, -sinphi * sintheta, -costheta])
    yhat = np.array([+sinphi, -cosphi, 0.0])
    zhat = np.array([-cosphi * costheta, -sinphi * costheta, +sintheta])
    return xhat, yhat, zhat


def rblhat(l, b):
    """Return r-hat, b-hat, l-hat components in Cartesian frame."""
    theta = np.pi / 2.0 - b
    rhat = np.array([np.sin(theta) * np.cos(l), np.sin(theta) * np.sin(l), np.cos(theta)])
    bhat = np.array([-np.cos(theta) * np.cos(l), -np.cos(theta) * np.sin(l), np.sin(theta)])
    lhat = np.array([-np.sin(l), np.cos(l), np.zeros_like(l)])
    return rhat, bhat, lhat


def xyz_to_xyz_sunframe(x, y, z, vx, vy, vz, xsun, ysun, zsun, vxsun, vysun, vzsun):
    """Transform simulation frame xyz to sun-centered XYZ."""
    xhat, yhat, zhat = xyz_basis_from_sun(xsun, ysun, zsun)

    deltax = x - xsun
    deltay = y - ysun
    deltaz = z - zsun
    deltavx = vx - vxsun
    deltavy = vy - vysun
    deltavz = vz - vzsun

    x_out = deltax * xhat[0] + deltay * xhat[1] + deltaz * xhat[2]
    y_out = deltax * yhat[0] + deltay * yhat[1] + deltaz * yhat[2]
    z_out = deltax * zhat[0] + deltay * zhat[1] + deltaz * zhat[2]

    vx_out = deltavx * xhat[0] + deltavy * xhat[1] + deltavz * xhat[2]
    vy_out = deltavx * yhat[0] + deltavy * yhat[1] + deltavz * yhat[2]
    vz_out = deltavx * zhat[0] + deltavy * zhat[1] + deltavz * zhat[2]

    return x_out, y_out, z_out, vx_out, vy_out, vz_out


def xyz_sunframe_to_lbr(X, Y, Z, vX, vY, vZ):
    """Convert heliocentric Cartesian phase-space to l,b,r and velocities."""
    r = np.sqrt(X**2 + Y**2 + Z**2)
    l = np.arctan2(Y, X)
    theta = np.arccos(Z / r)
    b = np.pi / 2.0 - theta

    rhat, bhat, lhat = rblhat(l, b)
    vr = vX * rhat[0] + vY * rhat[1] + vZ * rhat[2]
    vb = vX * bhat[0] + vY * bhat[1] + vZ * bhat[2]
    vl = vX * lhat[0] + vY * lhat[1] + vZ * lhat[2]
    return l, b, r, vl, vb, vr


def rotate_to_bar_frame(x, y, vx, vy, t, omega=4.0, phi_deg=20.0):
    """Rotate coordinates and velocities into the rotating bar frame.

    Bar-frame angle is ``theta = omega * t - phi``; applied to both positions
    and velocities so everything stays consistent before the galactic transform.

    Parameters
    ----------
    x, y : array-like
        Cartesian coordinates in the GC-centered frame (code units).
    vx, vy : array-like
        Velocity components in the GC-centered frame (km/s).
    t : float
        Simulation time in code units.
    omega : float, optional
        Bar pattern speed in code units; default 4.0.
    phi_deg : float, optional
        Bar initial phase offset in degrees; default 20.0.

    Returns
    -------
    x_rot, y_rot, vx_rot, vy_rot : ndarray
        Rotated positions and velocities.
    """
    theta = omega * t - np.radians(phi_deg)
    x_rot, y_rot = rotate_xy(x, y, theta)
    vx_rot, vy_rot = rotate_xy(vx, vy, theta)
    return x_rot, y_rot, vx_rot, vy_rot


def xyz_to_lbr(x, y, z, vx, vy, vz, **origin):
    """Convert GC-centered xyz and velocities to l,b,r and line-of-sight velocities."""
    params = GALACTIC_ORIGIN.copy()
    params.update(origin)
    X, Y, Z, vX, vY, vZ = xyz_to_xyz_sunframe(
        x, y, z, vx, vy, vz,
        params["xsun"], params["ysun"], params["zsun"],
        params["vxsun"], params["vysun"], params["vzsun"],
    )
    return xyz_sunframe_to_lbr(X, Y, Z, vX, vY, vZ)


def _normalize(vec):
    """Return a unit vector."""
    arr = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        raise ValueError("Zero-length vector cannot be normalized.")
    return arr / norm


def rotate_about_axis(vectors, axis, angle):
    """Rotate vector(s) about ``axis`` by ``angle`` radians using Rodrigues."""
    v = np.asarray(vectors, dtype=np.float64)
    k = _normalize(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    return v * c + np.cross(k, v) * s + k * np.sum(v * k, axis=-1, keepdims=True) * (1.0 - c)


def camera_frame(camera_position, target=(0.0, 0.0, 0.0), up_hint=(0.0, 0.0, 1.0)):
    """Return orthonormal camera basis vectors in world coordinates.

    Returns ``(right, up, forward)`` where forward points from camera to target.
    """
    cam = np.asarray(camera_position, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    up = _normalize(up_hint)

    forward = _normalize(tgt - cam)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-12:
        raise ValueError("up_hint is parallel to view direction; choose a different up_hint.")
    right = _normalize(right)
    up = _normalize(np.cross(right, forward))
    return right, up, forward


def world_to_camera(x, y, z, camera_position, target=(0.0, 0.0, 0.0), up_hint=(0.0, 0.0, 1.0)):
    """Project world coordinates into camera coordinates.

    Positive z in camera coordinates points forward from the camera.
    """
    right, up, forward = camera_frame(camera_position, target=target, up_hint=up_hint)
    cam = np.asarray(camera_position, dtype=np.float64)
    pts = np.column_stack([x, y, z]) - cam
    x_cam = pts @ right
    y_cam = pts @ up
    z_cam = pts @ forward
    return x_cam, y_cam, z_cam
