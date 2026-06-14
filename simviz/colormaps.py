"""Colormap and norm helpers."""

import matplotlib.colors as colors
import matplotlib.pyplot as plt


def resolve_cmap(name):
    """Return a matplotlib Colormap from a cmasher or matplotlib name."""
    if not isinstance(name, str):
        return name
    try:
        import cmasher as cm

        if hasattr(cm, name):
            return getattr(cm, name)
    except ImportError:
        pass
    return plt.get_cmap(name)


def make_lognorm(vmin, vmax):
    """Log normalization between vmin and vmax."""
    return colors.LogNorm(vmin=vmin, vmax=vmax)


def make_boundary_norm(levels, ncolors=256):
    """Boundary norm for a fixed set of discrete levels."""
    return colors.BoundaryNorm(levels, ncolors)
