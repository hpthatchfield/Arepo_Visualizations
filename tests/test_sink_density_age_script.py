"""Tests for sink parent-density vs age histogram helpers."""

import numpy as np

from scripts.plot_sink_parent_density_age import (
    collect_density_age_samples,
    collect_sne_overlay,
)
from simviz.utils import build_sink_data_sinkwise, sink_snap_dtype


def _one_track_snapwise(mass, rho, formation_time, snap_time, sne_times=None):
    dt = sink_snap_dtype(max_sne=4, max_accretion_events=2)
    rec = np.zeros(1, dtype=dt)
    rec["ID"][0] = 99
    rec["Mass"][0] = mass
    rec["FormationTime"][0] = formation_time
    rec["N_sne"][0] = len(sne_times or [])
    if sne_times:
        rec["explosion_time"][0, : len(sne_times)] = sne_times

    out = {"time": snap_time, "data": rec}
    if rho is not None:
        out["ParentDensity"] = np.array([rho], dtype=np.float64)
    return out


def test_collect_density_age_samples():
    snapwise = {
        500: _one_track_snapwise(1.0, 10.0, 2.0, 2.5),
        501: _one_track_snapwise(1.1, 20.0, 2.0, 2.6),
    }
    tracks = build_sink_data_sinkwise(snapwise)
    ages, log_rho = collect_density_age_samples(tracks, code_time_to_myr=100.0)
    assert ages.size == 2
    assert np.allclose(ages, [50.0, 60.0])
    assert np.allclose(log_rho, [np.log10(10.0), np.log10(20.0)])


def test_collect_sne_overlay_dedup():
    snapwise = {
        500: _one_track_snapwise(1.0, 10.0, 2.0, 2.5, sne_times=[2.3]),
        501: _one_track_snapwise(1.0, 12.0, 2.0, 2.6, sne_times=[2.3]),
    }
    tracks = build_sink_data_sinkwise(snapwise)
    ages, log_rho = collect_sne_overlay(tracks, code_time_to_myr=100.0)
    assert ages.size == 1
    assert np.isclose(ages[0], 30.0)
    assert np.isfinite(log_rho[0])
