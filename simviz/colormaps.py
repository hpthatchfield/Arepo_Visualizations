"""Colormap and norm helpers."""

import matplotlib.colors as colors


def make_lognorm(vmin, vmax):
    """Log normalization between vmin and vmax."""
    return colors.LogNorm(vmin=vmin, vmax=vmax)


def make_boundary_norm(levels, ncolors=256):
    """Boundary norm for a fixed set of discrete levels."""
    return colors.BoundaryNorm(levels, ncolors)
