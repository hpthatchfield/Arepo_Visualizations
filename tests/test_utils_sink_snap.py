"""Smoke tests for NumPy sink snapshot readers."""

import numpy as np

import h5py

from simviz.utils import (
    DEFAULT_SINK_MAX_ACCRETION_EVENTS,
    DEFAULT_SINK_MAX_SNE,
    attach_parent_fields_from_hdf5,
    build_sink_data_sinkwise,
    read_sink_snap,
    read_sink_snap_binary,
    sink_snap_dtype,
    validate_sink_gas_snap,
    valid_explosion_times,
)


def _write_sink_file(path, time_val, nsinks, records):
    """Write minimal sink_snap binary (header + structured records)."""
    dt = records.dtype
    with open(path, "wb") as fhandle:
        np.array([time_val], dtype=np.float64).tofile(fhandle)
        np.array([nsinks], dtype=np.uint32).tofile(fhandle)
        if nsinks > 0:
            records.tofile(fhandle)


def test_read_sink_snap_binary_one_sink(tmp_path):
    dt = sink_snap_dtype(max_sne=2, max_accretion_events=2)
    rec = np.zeros(1, dtype=dt)
    rec["ID"][0] = 42
    rec["Mass"][0] = 1.25
    rec["Pos"][0] = [1.0, 2.0, 3.0]

    path = tmp_path / "sink_snap_99"
    _write_sink_file(path, 2.5, 1, rec)

    out = read_sink_snap_binary(path, max_sne=2, max_accretion_events=2)
    assert out["time"] == 2.5
    assert out["NSinks"] == 1
    assert out["snap_num"] == 99
    assert out["data"]["ID"][0] == 42
    assert float(out["data"]["Mass"][0]) == 1.25
    assert np.allclose(out["data"]["Pos"][0], [1.0, 2.0, 3.0])


def test_read_sink_snap_binary_zero_sinks(tmp_path):
    dt = sink_snap_dtype(max_sne=2, max_accretion_events=2)
    path = tmp_path / "sink_snap_0"
    _write_sink_file(path, 1.0, 0, np.array([], dtype=dt))

    out = read_sink_snap_binary(path, max_sne=2, max_accretion_events=2)
    assert out["NSinks"] == 0
    assert out["data"].size == 0


def test_read_sink_snap_legacy_myobject(tmp_path):
    dt = sink_snap_dtype(max_sne=2, max_accretion_events=2)
    rec = np.zeros(1, dtype=dt)
    rec["Mass"][0] = 3.0

    path = tmp_path / "sink_snap_5"
    _write_sink_file(path, 0.0, 1, rec)

    obj = read_sink_snap(path, max_sne=2, max_accretion_events=2)
    assert obj.snap_num == 5
    assert len(obj.Mass) == 1
    assert float(obj.Mass[0]) == 3.0
    assert len(obj.time) == 1


def test_valid_explosion_times_drops_padding():
    raw = np.array([2.0, 2.1, 1.0e300, -1.0, 0.0, np.nan], dtype=np.float64)
    times = valid_explosion_times(raw, 6, snap_time=2.5, formation_time=1.0, max_sne=6)
    assert np.allclose(times, [2.0, 2.1])


def test_valid_explosion_times_respects_formation_time():
    raw = np.array([0.5, 1.5, 2.0], dtype=np.float64)
    times = valid_explosion_times(raw, 3, snap_time=3.0, formation_time=1.0, max_sne=3)
    assert np.allclose(times, [1.5, 2.0])


def test_build_sink_data_sinkwise_with_sne_times():
    dt = sink_snap_dtype(max_sne=3, max_accretion_events=2)
    r0 = np.zeros(1, dtype=dt)
    r0["ID"][0] = 7
    r0["Mass"][0] = 1.0
    r0["Pos"][0] = [0.0, 0.0, 1.0]
    r0["FormationTime"][0] = 0.1
    r0["N_sne"][0] = 2
    r0["explosion_time"][0] = [0.2, 0.25, 1e99]

    snapwise = {100: {"time": 1.0, "data": r0}}
    sk = build_sink_data_sinkwise(snapwise)
    assert np.allclose(sk[7]["sne_times"][0], [0.2, 0.25])


def test_attach_parent_fields_from_hdf5(tmp_path):
    dt = sink_snap_dtype(max_sne=2, max_accretion_events=2)
    rec = np.zeros(2, dtype=dt)
    rec["ID"][0] = 100
    rec["ID"][1] = 200
    rec["Pos"][0] = [1.0, 0.0, 0.0]
    rec["Pos"][1] = [2.0, 0.0, 0.0]

    path = tmp_path / "sink_snap_10"
    _write_sink_file(path, 1.0, 2, rec)
    out = read_sink_snap_binary(path, max_sne=2, max_accretion_events=2)

    h5_path = tmp_path / "gas_10.hdf5"
    with h5py.File(h5_path, "w") as fhandle:
        fhandle.create_group("Header")
        fhandle["Header"].attrs["BoxSize"] = 100.0
        fhandle.create_dataset(
            "PartType0/Coordinates",
            data=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
        )
        fhandle.create_dataset("PartType0/Density", data=np.array([7.0, 3.0], dtype=np.float64))

    attach_parent_fields_from_hdf5(out, h5_path)
    assert np.allclose(out["ParentDensity"], [7.0, 3.0])
    assert np.allclose(out["ParentDistance"], [0.0, 0.0])

    tracks = build_sink_data_sinkwise({10: out})
    assert np.allclose(tracks[100]["rho_parent"], [7.0])
    assert np.allclose(tracks[200]["rho_parent"], [3.0])


def test_validate_sink_gas_snap_ok(tmp_path):
    dt = sink_snap_dtype(
        max_sne=DEFAULT_SINK_MAX_SNE,
        max_accretion_events=DEFAULT_SINK_MAX_ACCRETION_EVENTS,
    )
    rec = np.zeros(1, dtype=dt)
    rec["ID"][0] = 100
    rec["Mass"][0] = 5.0
    rec["Pos"][0] = [1.0, 0.0, 0.0]
    rec["FormationTime"][0] = 0.5
    rec["N_sne"][0] = 1
    rec["explosion_time"][0, 0] = 0.9

    sink_path = tmp_path / "sink_snap_10"
    with open(sink_path, "wb") as fhandle:
        np.array([1.0], dtype=np.float64).tofile(fhandle)
        np.array([1], dtype=np.uint32).tofile(fhandle)
        rec.tofile(fhandle)

    h5_path = tmp_path / "phoenix_stinks_1Msun_10.hdf5"
    with h5py.File(h5_path, "w") as fhandle:
        fhandle.create_group("Header")
        fhandle["Header"].attrs["Time"] = 1.0
        fhandle["Header"].attrs["BoxSize"] = 100.0
        fhandle.create_dataset(
            "PartType0/Coordinates",
            data=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        )
        fhandle.create_dataset("PartType0/Density", data=np.array([7.0], dtype=np.float64))

    report = validate_sink_gas_snap(tmp_path, 10)
    assert report["ok"]
    assert report["n_sinks"] == 1
    assert report["n_finite_rho"] == 1
    assert report["spotcheck"][0]["rho_gas"] == 7.0
    assert report["spotcheck"][0]["n_sne_valid"] == 1


def test_validate_sink_gas_snap_time_mismatch(tmp_path):
    dt = sink_snap_dtype(
        max_sne=DEFAULT_SINK_MAX_SNE,
        max_accretion_events=DEFAULT_SINK_MAX_ACCRETION_EVENTS,
    )
    rec = np.zeros(1, dtype=dt)
    rec["ID"][0] = 1
    rec["Pos"][0] = [0.0, 0.0, 0.0]

    sink_path = tmp_path / "sink_snap_3"
    with open(sink_path, "wb") as fhandle:
        np.array([1.0], dtype=np.float64).tofile(fhandle)
        np.array([1], dtype=np.uint32).tofile(fhandle)
        rec.tofile(fhandle)

    h5_path = tmp_path / "phoenix_stinks_1Msun_3.hdf5"
    with h5py.File(h5_path, "w") as fhandle:
        fhandle.create_group("Header")
        fhandle["Header"].attrs["Time"] = 2.0
        fhandle["Header"].attrs["BoxSize"] = 10.0
        fhandle.create_dataset(
            "PartType0/Coordinates",
            data=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        )
        fhandle.create_dataset("PartType0/Density", data=np.array([1.0], dtype=np.float64))

    report = validate_sink_gas_snap(tmp_path, 3)
    assert not report["ok"]
    assert any("time mismatch" in err for err in report["errors"])


def test_build_sink_data_sinkwise_two_snaps():
    dt = sink_snap_dtype(max_sne=2, max_accretion_events=2)
    r0 = np.zeros(1, dtype=dt)
    r0["ID"][0] = 7
    r0["Mass"][0] = 1.0
    r0["Pos"][0] = [0.0, 0.0, 1.0]
    r0["FormationTime"][0] = 0.1
    r0["FormationMass"][0] = 0.5

    r1 = np.zeros(1, dtype=dt)
    r1["ID"][0] = 7
    r1["Mass"][0] = 2.0
    r1["Pos"][0] = [1.0, 0.0, 1.0]
    r1["FormationTime"][0] = 0.1
    r1["FormationMass"][0] = 0.5

    snapwise = {100: {"time": 1.0, "data": r0}, 101: {"time": 1.1, "data": r1}}
    sk = build_sink_data_sinkwise(snapwise)
    assert 7 in sk
    assert np.allclose(sk[7]["mass"], [1.0, 2.0])
    assert sk[7]["first_snap"] == 100
