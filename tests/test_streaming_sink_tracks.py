"""Tests for one-snap-at-a-time sink track building and SNe nearest-snap lookup."""

import numpy as np

from scripts.plot_sink_parent_density_age import collect_sne_overlay
from simviz.utils import (
    append_snap_to_sink_tracks,
    build_sink_data_sinkwise,
    init_streaming_sink_tracks,
    nearest_snap_index_for_time,
    sink_snap_dtype,
)


def _snap_entry(snap_time, sid, mass, rho, formation_time, sne_times=None):
    dt = sink_snap_dtype(max_sne=4, max_accretion_events=2)
    rec = np.zeros(1, dtype=dt)
    rec["ID"][0] = sid
    rec["Mass"][0] = mass
    rec["FormationTime"][0] = formation_time
    rec["N_sne"][0] = len(sne_times or [])
    if sne_times:
        rec["explosion_time"][0, : len(sne_times)] = sne_times
    out = {"time": snap_time, "NSinks": 1, "data": rec}
    if rho is not None:
        out["ParentDensity"] = np.array([rho], dtype=np.float64)
        out["ParentDistance"] = np.array([0.0], dtype=np.float64)
    return out


def test_append_snap_matches_build_sink_data_sinkwise():
    snapwise = {
        500: _snap_entry(2.5, 7, 1.0, 10.0, 2.0),
        501: _snap_entry(2.6, 7, 1.1, 20.0, 2.0),
    }
    expected = build_sink_data_sinkwise(snapwise)

    snaps = np.array([500, 501], dtype=int)
    store = init_streaming_sink_tracks(snaps)
    for isnap in snaps:
        append_snap_to_sink_tracks(store, isnap, snapwise[int(isnap)], max_sne=4)

    got = store["tracks"][7]
    assert np.allclose(got["mass"], expected[7]["mass"])
    assert np.allclose(got["rho_parent"], expected[7]["rho_parent"])
    assert np.allclose(got["time"], expected[7]["time"])


def test_nearest_snap_index_for_time_between_snaps():
    times = np.array([2.5, 2.6, np.nan], dtype=np.float64)
    assert nearest_snap_index_for_time(times, 2.58) == 1
    assert nearest_snap_index_for_time(times, 2.52) == 0
    # equidistant: earlier snap wins (numpy argmin tie-break)
    assert nearest_snap_index_for_time(times, 2.55) == 0


def test_sne_overlay_uses_nearest_snap_rho():
    """SNe time between snapshot times should take rho from the closer snap."""
    snapwise = {
        500: _snap_entry(2.5, 9, 1.0, 10.0, 2.0, sne_times=[2.58]),
        501: _snap_entry(2.6, 9, 1.0, 20.0, 2.0, sne_times=[2.58]),
    }
    tracks = build_sink_data_sinkwise(snapwise)
    ages, log_rho = collect_sne_overlay(tracks, code_time_to_myr=100.0)
    assert ages.size == 1
    assert np.isclose(ages[0], 58.0)
    assert np.isclose(log_rho[0], np.log10(20.0))
