#!/usr/bin/env python3
"""2D histogram: parent gas density vs time since sink formation, SNe as stars."""

import argparse
import gc
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

from simviz.utils import (
    CONSTANTS,
    DEFAULT_GAS_SNAP_PREFIX,
    DEFAULT_SINK_MAX_ACCRETION_EVENTS,
    DEFAULT_SINK_MAX_SNE,
    DEFAULT_SINK_SNAP_TEMPLATE,
    append_snap_to_sink_tracks,
    attach_parent_fields_from_hdf5,
    gas_snap_path,
    init_streaming_sink_tracks,
    list_sink_snaps_in_range,
    nearest_snap_index_for_time,
    read_sink_snap_binary,
    sink_snap_dtype,
    validate_sink_gas_snap,
)

SINK_SNAP_TEMPLATE = DEFAULT_SINK_SNAP_TEMPLATE
GAS_SNAP_PREFIX = DEFAULT_GAS_SNAP_PREFIX
DEFAULT_MAX_SNE = DEFAULT_SINK_MAX_SNE
DEFAULT_MAX_ACCRETION = DEFAULT_SINK_MAX_ACCRETION_EVENTS
DEFAULT_CODE_TIME_TO_MYR = 98.7
CHECKPOINT_TEMPLATE = "parent_env_{snap:04d}.npz"


def print_validation_report(report):
    """Print one-snap validation results to stdout."""
    print(f"validate snap {report['isnap']}")
    print(f"  sink: {report.get('sink_path')}")
    print(f"  gas:  {report.get('gas_path')}")
    if "t_sink" in report:
        print(f"  time sink binary: {report['t_sink']}")
        print(f"  time gas header:  {report['t_gas']}  (dt={report['dt_time']})")
    if "n_gas_cells" in report:
        print(f"  gas cells: {report['n_gas_cells']:,}")
    if "n_sinks" in report:
        print(f"  sinks: {report['n_sinks']:,}   finite gas rho: {report['n_finite_rho']:,}")
    for msg in report.get("warnings", []):
        print(f"  warn: {msg}")
    for msg in report.get("errors", []):
        print(f"  ERROR: {msg}")
    for row in report.get("spotcheck", []):
        print(
            f"  id={row['id']}  mass={row['mass']:.4g}  "
            f"rho={row['rho_gas']:.4g}  d={row['d_nearest']:.4g}  "
            f"n_sne={row['n_sne']} ({row['n_sne_valid']} valid)"
        )
    print("ok" if report.get("ok") else "FAILED")


def age_since_formation_myr(snap_time, formation_time, code_time_to_myr):
    """Time since sink creation in Myr (simulation time minus FormationTime)."""
    return (float(snap_time) - float(formation_time)) * float(code_time_to_myr)


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
    """Unique SNe per sink: (age Myr, log10 density) at nearest snap time."""
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
                j = nearest_snap_index_for_time(tr["time"], t_sn)
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


def checkpoint_path(checkpoint_dir, isnap):
    return Path(checkpoint_dir) / CHECKPOINT_TEMPLATE.format(snap=int(isnap))


def save_snap_checkpoint(path, isnap, entry):
    """Write slim per-snap fields for resume without gas HDF5."""
    rec = entry["data"]
    rho = np.asarray(entry["ParentDensity"], dtype=np.float64)
    dist = np.asarray(entry["ParentDistance"], dtype=np.float64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        isnap=np.int32(isnap),
        time_sink=np.float64(entry["time"]),
        sink_id=rec["ID"].astype(np.uint64, copy=False),
        formation_time=rec["FormationTime"].astype(np.float64, copy=False),
        formation_mass=rec["FormationMass"].astype(np.float64, copy=False),
        mass=rec["Mass"].astype(np.float64, copy=False),
        parent_rho=rho,
        parent_dist=dist,
        n_sne=rec["N_sne"].astype(np.uint32, copy=False),
        explosion_time=rec["explosion_time"].astype(np.float64, copy=False),
    )


def load_snap_from_checkpoint(path, max_sne, max_accretion_events):
    """Rebuild a one-snap entry dict from a checkpoint file."""
    with np.load(path, allow_pickle=False) as data:
        n_sink = int(data["sink_id"].shape[0])
        dt = sink_snap_dtype(max_sne=max_sne, max_accretion_events=max_accretion_events)
        rec = np.zeros(n_sink, dtype=dt)
        rec["ID"] = data["sink_id"]
        rec["FormationTime"] = data["formation_time"]
        rec["FormationMass"] = data["formation_mass"]
        rec["Mass"] = data["mass"]
        rec["N_sne"] = data["n_sne"]
        n_sne = min(int(max_sne), int(rec["explosion_time"].shape[1]))
        rec["explosion_time"][:, :n_sne] = data["explosion_time"][:, :n_sne]
        return {
            "time": float(data["time_sink"]),
            "NSinks": n_sink,
            "data": rec,
            "ParentDensity": np.asarray(data["parent_rho"], dtype=np.float64),
            "ParentDistance": np.asarray(data["parent_dist"], dtype=np.float64),
        }


def load_and_attach_one_snap(
    snap_dir,
    isnap,
    gas_prefix,
    max_sne,
    max_accretion_events,
):
    """Read sink binary, attach parent gas density, return entry (transient)."""
    sink_path = Path(snap_dir) / SINK_SNAP_TEMPLATE.format(snap=isnap)
    entry = read_sink_snap_binary(
        sink_path,
        max_sne=max_sne,
        max_accretion_events=max_accretion_events,
    )
    gas_path = gas_snap_path(snap_dir, isnap, prefix=gas_prefix)
    if gas_path is None:
        raise FileNotFoundError(f"no gas HDF5 for snap {isnap} in {snap_dir}")
    attach_parent_fields_from_hdf5(entry, gas_path)
    return entry


def hist_samples_from_snap(entry, code_time_to_myr, use_cgs=False):
    """(age, log_rho) samples from one snap for histogram accumulation."""
    rec = entry["data"]
    rho = np.asarray(entry["ParentDensity"], dtype=np.float64)
    snap_time = float(entry["time"])
    ages = []
    log_rho = []
    for j in range(rec.size):
        sid = int(rec["ID"][j])
        if sid == 0:
            continue
        ft = float(rec["FormationTime"][j])
        if not np.isfinite(ft):
            continue
        rho_j = float(rho[j])
        if not np.isfinite(rho_j) or rho_j <= 0:
            continue
        if not np.isfinite(rec["Mass"][j]):
            continue
        age = age_since_formation_myr(snap_time, ft, code_time_to_myr)
        if age < 0:
            continue
        rho_plot = rho_j
        if use_cgs:
            rho_plot *= CONSTANTS["arepoDensity"]
        ages.append(age)
        log_rho.append(np.log10(rho_plot))
    return np.asarray(ages, dtype=np.float32), np.asarray(log_rho, dtype=np.float32)


def accumulate_histogram(counts, age_edges, rho_edges, ages, log_rho):
    """Add samples into a preallocated 2D count array."""
    if ages.size == 0:
        return counts
    ia = np.digitize(ages, age_edges) - 1
    ir = np.digitize(log_rho, rho_edges) - 1
    ok = (ia >= 0) & (ia < counts.shape[0]) & (ir >= 0) & (ir < counts.shape[1])
    if np.any(ok):
        np.add.at(counts, (ia[ok], ir[ok]), 1)
    return counts


def default_hist_edges(age_bins, rho_bins, age_hi, rho_lo=-3.0, rho_hi=8.0):
    """Histogram edges wide enough for streaming; display limits trimmed later."""
    age_edges = np.linspace(0.0, max(float(age_hi), 1e-6), int(age_bins) + 1)
    rho_edges = np.linspace(float(rho_lo), float(rho_hi), int(rho_bins) + 1)
    return age_edges, rho_edges


def plot_hist2d_counts(
    counts,
    age_edges,
    rho_edges,
    out_path,
    sne_ages=None,
    sne_log_rho=None,
    age_lim=None,
    rho_lim=None,
    title=None,
    rho_label=None,
):
    """Render accumulated 2D counts with optional SNe overlay."""
    if counts.sum() == 0:
        raise ValueError("no samples for histogram (missing gas density at sinks?)")

    age_centers = 0.5 * (age_edges[:-1] + age_edges[1:])
    rho_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])

    if age_lim is None:
        row_w = counts.sum(axis=1)
        if row_w.sum() > 0:
            lo = int(np.argmax(row_w > 0))
            hi = int(len(row_w) - np.argmax(row_w[::-1] > 0))
            age_lim = (age_centers[max(lo - 1, 0)], age_centers[min(hi, len(age_centers) - 1)])
        else:
            age_lim = (age_centers[0], age_centers[-1])

    if rho_lim is None:
        col_w = counts.sum(axis=0)
        pos = col_w > 0
        if np.any(pos):
            vals = rho_centers[pos]
            w = col_w[pos].astype(np.float64)
            cum = np.cumsum(w) / w.sum()
            rho_lim = (
                float(np.interp(0.01, cum, vals)),
                float(np.interp(0.99, cum, vals)),
            )
        else:
            rho_lim = (rho_centers[0], rho_centers[-1])

    ia0 = max(int(np.searchsorted(age_edges, age_lim[0], side="right")) - 1, 0)
    ia1 = min(int(np.searchsorted(age_edges, age_lim[1], side="right")), counts.shape[0])
    ir0 = max(int(np.searchsorted(rho_edges, rho_lim[0], side="right")) - 1, 0)
    ir1 = min(int(np.searchsorted(rho_edges, rho_lim[1], side="right")), counts.shape[1])

    sub = counts[ia0:ia1, ir0:ir1]
    extent = (age_edges[ia0], age_edges[ia1], rho_edges[ir0], rho_edges[ir1])

    if rho_label is None:
        rho_label = r"$\log_{10}$ parent $\rho$ [code]"

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=150)
    mesh = ax.imshow(
        sub.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="inferno",
        norm=colors.LogNorm(vmin=max(sub.min(), 1)),
    )
    sne_ages = np.asarray([] if sne_ages is None else sne_ages, dtype=np.float64)
    sne_log_rho = np.asarray([] if sne_log_rho is None else sne_log_rho, dtype=np.float64)
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

    plt.colorbar(mesh, ax=ax, label="counts")
    ax.set_xlabel("time since formation [Myr]")
    ax.set_ylabel(rho_label)
    if title:
        ax.set_title(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def read_sink_snap_time(sink_path):
    """Read simulation time from a sink_snap binary header."""
    with open(sink_path, "rb") as handle:
        return float(np.fromfile(handle, dtype=np.float64, count=1)[0])


def estimate_age_upper_myr(snap_dir, snaps, code_time_to_myr):
    """Upper age bound from first/last sink binary times in the range."""
    t0 = read_sink_snap_time(Path(snap_dir) / SINK_SNAP_TEMPLATE.format(snap=int(snaps[0])))
    t1 = read_sink_snap_time(Path(snap_dir) / SINK_SNAP_TEMPLATE.format(snap=int(snaps[-1])))
    return max(t1 - t0, t1) * float(code_time_to_myr) * 1.05


def stream_snaps(
    snap_dir,
    snaps,
    gas_prefix,
    max_sne,
    max_accretion_events,
    code_time_to_myr,
    age_bins=80,
    rho_bins=80,
    use_cgs=False,
    checkpoint_dir=None,
    resume=False,
):
    """Process snaps one at a time; return track store and histogram counts."""
    store = init_streaming_sink_tracks(snaps)
    age_hi = estimate_age_upper_myr(snap_dir, snaps, code_time_to_myr)
    age_edges, rho_edges = default_hist_edges(age_bins, rho_bins, age_hi=age_hi)
    counts = np.zeros((age_bins, rho_bins), dtype=np.uint32)
    n_loaded = 0
    n_skipped = 0
    fail_log = None
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        fail_log = checkpoint_dir / "failures.txt"

    for isnap in snaps:
        t0 = time.perf_counter()
        ckpt = None if checkpoint_dir is None else checkpoint_path(checkpoint_dir, isnap)

        try:
            if ckpt is not None and resume and ckpt.is_file():
                entry = load_snap_from_checkpoint(ckpt, max_sne, max_accretion_events)
                source = "checkpoint"
            else:
                entry = load_and_attach_one_snap(
                    snap_dir,
                    isnap,
                    gas_prefix,
                    max_sne,
                    max_accretion_events,
                )
                if ckpt is not None:
                    save_snap_checkpoint(ckpt, isnap, entry)
                source = "hdf5"

            append_snap_to_sink_tracks(store, isnap, entry, max_sne=max_sne)
            ages, log_rho = hist_samples_from_snap(
                entry, code_time_to_myr, use_cgs=use_cgs
            )
            accumulate_histogram(counts, age_edges, rho_edges, ages, log_rho)
            n_loaded += 1
            elapsed = time.perf_counter() - t0
            print(
                f"[{isnap}] {source}  sinks={entry['NSinks']:,}  "
                f"samples={ages.size:,}  tracks={len(store['tracks']):,}  "
                f"{elapsed:.1f}s"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            msg = f"[{isnap}] FAILED after {elapsed:.1f}s: {exc!r}"
            print(msg)
            if fail_log is not None:
                with open(fail_log, "a", encoding="utf-8") as handle:
                    handle.write(msg + "\n")
            if ckpt is not None and resume:
                n_skipped += 1
                continue
            raise
        finally:
            if "entry" in locals():
                del entry
            gc.collect()

    if n_loaded == 0:
        raise RuntimeError("no snapshots loaded; check paths, snap range, and checkpoints")

    print(
        f"streamed {n_loaded} snaps ({snaps[0]} .. {snaps[-1]})  "
        f"skipped={n_skipped}  unique sinks={len(store['tracks']):,}"
    )
    return store, counts, age_edges, rho_edges


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
    parser.add_argument(
        "--validate-snap",
        type=int,
        default=None,
        metavar="N",
        help="check one snap (times, gas rho at sinks) and exit",
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
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="write/read per-snap parent_env_<N>.npz checkpoints here",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip snaps that already have checkpoints (requires --checkpoint-dir)",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("sink_rho_vs_age.png"))
    args = parser.parse_args()

    if args.resume and args.checkpoint_dir is None:
        parser.error("--resume requires --checkpoint-dir")

    if args.validate_snap is not None:
        report = validate_sink_gas_snap(
            args.snap_dir,
            args.validate_snap,
            gas_prefix=args.gas_prefix,
            max_sne=args.max_sne,
            max_accretion_events=args.max_accretion_events,
        )
        print_validation_report(report)
        sys.exit(0 if report["ok"] else 1)

    snaps = np.array(
        list_sink_snaps_in_range(
            args.snap_dir,
            args.snap_first,
            args.snap_last,
            gas_prefix=args.gas_prefix,
        ),
        dtype=int,
    )
    if snaps.size == 0:
        raise RuntimeError("no snapshots found in range; check paths and snap range")

    print(f"found {snaps.size} snaps in range ({snaps[0]} .. {snaps[-1]})")

    store, counts, age_edges, rho_edges = stream_snaps(
        args.snap_dir,
        snaps,
        args.gas_prefix,
        args.max_sne,
        args.max_accretion_events,
        args.code_time_to_myr,
        age_bins=args.age_bins,
        rho_bins=args.rho_bins,
        use_cgs=args.density_cgs,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
    )

    tracks = store["tracks"]
    n_rho = sum(int(np.isfinite(tr["rho_parent"]).sum()) for tr in tracks.values())
    n_tot = sum(tr["rho_parent"].size for tr in tracks.values())
    print(f"gas density at sinks on {n_rho:,} / {n_tot:,} track points")

    sne_ages, sne_log_rho = collect_sne_overlay(
        tracks, args.code_time_to_myr, use_cgs=args.density_cgs
    )
    print(f"hist counts sum: {int(counts.sum()):,}   SNe markers: {sne_ages.size:,}")

    rho_label = (
        r"$\log_{10}$ parent $\rho$ [g cm$^{-3}$]"
        if args.density_cgs
        else r"$\log_{10}$ parent $\rho$ [code]"
    )
    title = f"snaps {snaps[0]}–{snaps[-1]}  ({len(tracks)} sinks)"
    plot_hist2d_counts(
        counts,
        age_edges,
        rho_edges,
        args.output,
        sne_ages=sne_ages,
        sne_log_rho=sne_log_rho,
        title=title,
        rho_label=rho_label,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
