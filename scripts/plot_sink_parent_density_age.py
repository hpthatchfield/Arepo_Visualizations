#!/usr/bin/env python3
"""2D histogram: parent gas density vs time since sink formation, SNe as stars."""

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

from simviz.utils import (
    CONSTANTS,
    attach_parent_fields_from_hdf5,
    build_sink_data_sinkwise,
    read_sink_snap_binary,
)

SINK_SNAP_TEMPLATE = "sink_snap_{snap}"
GAS_SNAP_PREFIX = "phoenix_stinks_1Msun"
DEFAULT_MAX_SNE = 2000
DEFAULT_MAX_ACCRETION = 50
DEFAULT_CODE_TIME_TO_MYR = 98.7


def gas_snap_path(gas_dir, isnap, prefix=GAS_SNAP_PREFIX):
    """Resolve gas HDF5 for snapshot index (tries padded and unpadded names)."""
    candidates = (
        gas_dir / f"{prefix}_{isnap}.hdf5",
        gas_dir / f"{prefix}_{isnap:03d}.hdf5",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def age_since_formation_myr(snap_time, formation_time, code_time_to_myr):
    """Time since sink creation in Myr (simulation time minus FormationTime)."""
    return (float(snap_time) - float(formation_time)) * float(code_time_to_myr)


def nearest_time_index(times, t_query):
    """Index of ``times`` closest to ``t_query`` (ignores NaN)."""
    t = np.asarray(times, dtype=np.float64)
    ok = np.isfinite(t)
    if not np.any(ok):
        return None
    idx_ok = np.where(ok)[0]
    j = int(np.argmin(np.abs(t[ok] - float(t_query))))
    return int(idx_ok[j])


def collect_density_age_samples(sink_tracks, code_time_to_myr, use_cgs=False):
    """Flatten (age Myr, log10 density) for every finite track point."""
    ages = []
    log_rho = []
    for tr in sink_tracks.values():
        ft = tr["formationTime"]
        if not np.isfinite(ft):
            continue
        for i in range(tr["time"].size):
            rho = tr["rho_parent"][i]
            if not np.isfinite(rho) or rho <= 0:
                continue
            if not np.isfinite(tr["mass"][i]):
                continue
            age = age_since_formation_myr(tr["time"][i], ft, code_time_to_myr)
            if age < 0:
                continue
            rho_plot = float(rho)
            if use_cgs:
                rho_plot *= CONSTANTS["arepoDensity"]
            ages.append(age)
            log_rho.append(np.log10(rho_plot))
    return np.asarray(ages, dtype=np.float64), np.asarray(log_rho, dtype=np.float64)


def collect_sne_overlay(sink_tracks, code_time_to_myr, use_cgs=False):
    """Unique SNe per sink: (age Myr, log10 density) at nearest snap with rho."""
    ages = []
    log_rho = []
    for tr in sink_tracks.values():
        ft = tr["formationTime"]
        if not np.isfinite(ft):
            continue
        seen = set()
        for i in range(tr["time"].size):
            sne_list = tr["sne_times"][i]
            if sne_list is None or len(sne_list) == 0:
                continue
            for t_sn in np.asarray(sne_list, dtype=np.float64):
                key = (int(round(t_sn * 1e12)),)
                if key in seen:
                    continue
                seen.add(key)
                age = age_since_formation_myr(t_sn, ft, code_time_to_myr)
                if age < 0:
                    continue
                j = nearest_time_index(tr["time"], t_sn)
                if j is None:
                    continue
                rho = tr["rho_parent"][j]
                if not np.isfinite(rho) or rho <= 0:
                    continue
                rho_plot = float(rho)
                if use_cgs:
                    rho_plot *= CONSTANTS["arepoDensity"]
                ages.append(age)
                log_rho.append(np.log10(rho_plot))
    return np.asarray(ages, dtype=np.float64), np.asarray(log_rho, dtype=np.float64)


def load_snap_range(
    snap_dir,
    snap_first,
    snap_last,
    gas_prefix,
    max_sne,
    max_accretion_events,
    skip_missing=True,
):
    """Read sink_snap_* and matching gas HDF5 from the same directory."""
    snapwise = {}
    for isnap in range(int(snap_first), int(snap_last) + 1):
        sink_path = snap_dir / SINK_SNAP_TEMPLATE.format(snap=isnap)
        if not sink_path.is_file():
            if skip_missing:
                continue
            raise FileNotFoundError(sink_path)

        entry = read_sink_snap_binary(
            sink_path,
            max_sne=max_sne,
            max_accretion_events=max_accretion_events,
        )
        gas_path = gas_snap_path(snap_dir, isnap, prefix=gas_prefix)
        if gas_path is None:
            if skip_missing:
                continue
            raise FileNotFoundError(f"no gas HDF5 for snap {isnap} in {snap_dir}")
        attach_parent_fields_from_hdf5(entry, gas_path)
        snapwise[isnap] = entry

    if not snapwise:
        raise RuntimeError("no snapshots loaded; check paths and snap range")
    return snapwise


def plot_hist2d_with_sne(
    ages,
    log_rho,
    sne_ages,
    sne_log_rho,
    out_path,
    age_bins=80,
    rho_bins=80,
    rho_lim=None,
    age_lim=None,
    title=None,
    rho_label=None,
):
    """Draw density–age 2D histogram and overlay SNe markers."""
    if ages.size == 0:
        raise ValueError("no samples for histogram (missing ParentDensity?)")

    if age_lim is None:
        age_lim = (0.0, float(np.nanmax(ages)) * 1.02 + 1e-6)
    if rho_lim is None:
        lo = float(np.nanpercentile(log_rho, 1))
        hi = float(np.nanpercentile(log_rho, 99))
        rho_lim = (lo, hi)

    if rho_label is None:
        rho_label = r"$\log_{10}$ parent $\rho$ [code]"

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=150)
    h = ax.hist2d(
        ages,
        log_rho,
        bins=[age_bins, rho_bins],
        range=[age_lim, rho_lim],
        cmap="inferno",
        norm=colors.LogNorm(),
    )
    if sne_ages.size > 0:
        ax.scatter(
            sne_ages,
            sne_log_rho,
            marker="*",
            s=8,
            c="cyan",
            edgecolors="k",
            linewidths=0.2,
            alpha=0.85,
            zorder=5,
            label="SNe",
        )
        ax.legend(loc="upper right", fontsize=8)

    plt.colorbar(h[3], ax=ax, label="counts")
    ax.set_xlabel("time since formation [Myr]")
    ax.set_ylabel(rho_label)
    if title:
        ax.set_title(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="2D histogram of sink parent density vs age since formation."
    )
    parser.add_argument(
        "--snap-dir",
        type=Path,
        required=True,
        help="directory with sink_snap_<N> and <prefix>_<N>.hdf5",
    )
    parser.add_argument("--gas-prefix", default=GAS_SNAP_PREFIX)
    parser.add_argument("--snap-first", type=int, default=500)
    parser.add_argument("--snap-last", type=int, default=1000)
    parser.add_argument(
        "--code-time-to-myr",
        type=float,
        default=DEFAULT_CODE_TIME_TO_MYR,
        help="multiply (t - FormationTime) in code units to get Myr",
    )
    parser.add_argument("--density-cgs", action="store_true", help="plot rho in g/cm^3")
    parser.add_argument("--max-sne", type=int, default=DEFAULT_MAX_SNE)
    parser.add_argument("--max-accretion-events", type=int, default=DEFAULT_MAX_ACCRETION)
    parser.add_argument("--age-bins", type=int, default=80)
    parser.add_argument("--rho-bins", type=int, default=80)
    parser.add_argument("-o", "--output", type=Path, default=Path("sink_rho_vs_age.png"))
    args = parser.parse_args()

    snapwise = load_snap_range(
        args.snap_dir,
        args.snap_first,
        args.snap_last,
        args.gas_prefix,
        args.max_sne,
        args.max_accretion_events,
    )
    snaps = np.array(sorted(snapwise.keys()), dtype=int)
    print(f"loaded {len(snaps)} snaps ({snaps[0]} .. {snaps[-1]})")

    tracks = build_sink_data_sinkwise(snapwise, snaps=snaps)
    print(f"{len(tracks)} sink tracks")
    n_rho = sum(int(np.isfinite(tr["rho_parent"]).sum()) for tr in tracks.values())
    n_tot = sum(tr["rho_parent"].size for tr in tracks.values())
    print(f"PartType5 parent density on {n_rho:,} / {n_tot:,} track points")

    ages, log_rho = collect_density_age_samples(
        tracks, args.code_time_to_myr, use_cgs=args.density_cgs
    )
    sne_ages, sne_log_rho = collect_sne_overlay(
        tracks, args.code_time_to_myr, use_cgs=args.density_cgs
    )
    print(f"hist samples: {ages.size:,}   SNe markers: {sne_ages.size:,}")

    rho_label = (
        r"$\log_{10}$ parent $\rho$ [g cm$^{-3}$]"
        if args.density_cgs
        else r"$\log_{10}$ parent $\rho$ [code]"
    )
    title = f"snaps {snaps[0]}–{snaps[-1]}  ({len(tracks)} sinks)"
    plot_hist2d_with_sne(
        ages,
        log_rho,
        sne_ages,
        sne_log_rho,
        args.output,
        age_bins=args.age_bins,
        rho_bins=args.rho_bins,
        title=title,
        rho_label=rho_label,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
