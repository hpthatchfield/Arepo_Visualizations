"""Snapshot I/O, unit conversions, and physical quantities."""

import gc
import os
import re

import numpy as np
from numpy import float32, float64, uint32, uint64


CONSTANTS = {
    "kb": 1.38e-23,
    "kb_cgs": 1.38e-16,
    "mH": 1.67e-24,
    "mCO": 4.65e-23,
    "mu": 2.8,
    "h": 6.626e-34,
    "h_cgs": 6.62e-27,
    "gamma": 5.0 / 3.0,
    "pc2cm": 3.086e18,
    "msun2g": 2e33,
    "arepoLength": 3.0856e20,
    "arepoMass": 1.991e33,
    "arepoVel": 1.0e5,
}
CONSTANTS["arepoTime"] = CONSTANTS["arepoLength"] / CONSTANTS["arepoVel"]
CONSTANTS["arepoDensity"] = (
    CONSTANTS["arepoMass"] / CONSTANTS["arepoLength"] / CONSTANTS["arepoLength"] / CONSTANTS["arepoLength"]
)
CONSTANTS["arepoEnergy"] = CONSTANTS["arepoMass"] * CONSTANTS["arepoVel"] * CONSTANTS["arepoVel"]
CONSTANTS["arepoColumnDensity"] = CONSTANTS["arepoMass"] / CONSTANTS["arepoLength"] / CONSTANTS["arepoLength"]
CONSTANTS["arepoVolume"] = CONSTANTS["arepoLength"] ** 3
# Gaussian CGS B-field unit: cm^{-1/2} g^{1/2} s^{-1}
CONSTANTS["arepoBfield"] = (
    CONSTANTS["arepoLength"] ** -0.5
    * CONSTANTS["arepoMass"] ** 0.5
    * CONSTANTS["arepoTime"] ** -1
)


def calc_lambda_jeans(rho_cgs, cs_kms):
    """Compute the Jeans length in cm."""
    cs_cgs = cs_kms * 1e5
    grav = 6.67e-8
    return np.sqrt(np.pi / grav / rho_cgs) * cs_cgs


def calc_mass_jeans(rho_cgs, cs_kms):
    """Compute the Jeans mass in solar masses."""
    lambda_jeans = calc_lambda_jeans(rho_cgs, cs_kms)
    return 4.0 / 3.0 * np.pi * rho_cgs * (lambda_jeans / 2e33) * lambda_jeans * lambda_jeans


def calc_sound_speed(T, mu=2.0):
    """Compute adiabatic sound speed in km/s."""
    mp = 1.6726231e-24
    kb = 1.3806485e-16
    cs_cgs = np.sqrt((5.0 / 3.0) * kb * T / (mu * mp))
    return cs_cgs / 1e5


def calc_co_quantities(density_code, masses_code, chem_abundances, xHe=0.1):
    """Compute CO mass (code units) and CO column density (cm^-2) for each gas cell.

    Parameters
    ----------
    density_code : array-like
        Cell densities in Arepo code units (Msun / (100 pc)^3).
    masses_code : array-like
        Cell masses in Arepo code units (Msun).
    chem_abundances : array-like, shape (N, 3)
        Chemical abundances per cell in column order (xH2, xHp, xCO).
    xHe : float, optional
        Helium mass fraction; default 0.1.

    Returns
    -------
    masses_co : ndarray
        CO mass per cell in code units (Msun).
    co_colden_cm : ndarray
        CO column density per cell in cm^-2, approximated as N_CO / r_cell^2.
    """
    C = CONSTANTS
    rho_cgs = np.asarray(density_code) * C["arepoDensity"]
    masses = np.asarray(masses_code)
    xH2, xHp, xCO = np.asarray(chem_abundances).T

    volumes_code = masses / np.asarray(density_code)
    cell_vol_cm = volumes_code * C["arepoVolume"]
    cell_radius_cm = ((3.0 / (4.0 * np.pi)) * cell_vol_cm) ** (1.0 / 3.0)

    n_H_tot = rho_cgs / ((1.0 + 4.0 * xHe) * C["mH"])
    n_CO = xCO * n_H_tot

    co_colden_cm = (n_CO * cell_vol_cm) / cell_radius_cm**2
    masses_co = (n_CO * 28.0 * C["mH"] / C["arepoDensity"]) * volumes_code

    return masses_co, co_colden_cm


def calc_bfield_uG(bfield_code):
    """Convert MagneticField from Arepo code units to micro-Gauss (µG).

    Parameters
    ----------
    bfield_code : array-like, shape (N, 3) or (N,)
        Raw ``MagneticField`` values read from an HDF5 snapshot.

    Returns
    -------
    bfield_uG : ndarray
        Magnetic field in µG; same shape as input.
    """
    return np.asarray(bfield_code) * CONSTANTS["arepoBfield"] * 1e6


def read_snapshot_hdf5(filename):
    """Read an HDF5 snapshot and return gas data and header.

    Parameters
    ----------
    filename : str or Path
        Snapshot file path.

    Returns
    -------
    data : dict
        ``PartType0`` fields.
    header : dict
        Header attribute dictionary.
    """
    header = {}
    data = {}

    import h5py

    with h5py.File(filename, "r") as fhandle:
        for item in fhandle["Header"].attrs:
            header[item] = fhandle["Header"].attrs[item]

        for item in fhandle["PartType0"].keys():
            data[item] = fhandle["PartType0"][item][:]

    gc.collect()
    return data, header


HEADER_NAMES = (
    "num_particles", "mass", "time", "redshift", "flag_sfr", "flag_feedback",
    "num_particles_total", "flag_cooling", "num_files", "boxsize", "omega0",
    "omegaLambda", "hubble0", "flag_stellarage", "flag_metals", "npartTotalHighWord",
    "flag_entropy_instead_u", "flag_doubleprecision", "flag_lpt_ics", "lpt_scalingfactor",
    "flag_tracer_field", "composition_vector_length", "buffer",
)

HEADER_SIZES = (
    (uint32, 6), (float64, 6), (float64, 1), (float64, 1), (uint32, 1), (uint32, 1),
    (uint32, 6), (uint32, 1), (uint32, 1), (float64, 1), (float64, 1), (float64, 1),
    (float64, 1), (uint32, 1), (uint32, 1), (uint32, 6), (uint32, 1), (uint32, 1),
    (uint32, 1), (float32, 1), (uint32, 1), (uint32, 1), (np.uint8, 40),
)


def read_header_binary(file_handle):
    """Read a binary AREPO header block."""
    _ = np.fromfile(file_handle, uint32, 1)[0]
    header = dict(
        (name, np.fromfile(file_handle, dtype=size[0], count=size[1]))
        for name, size in zip(HEADER_NAMES, HEADER_SIZES)
    )
    assert np.fromfile(file_handle, uint32, 1)[0] == 256
    return header


def read_fortran_block(file_handle, dtype=None):
    """Read one Fortran-style unformatted block."""
    data_size = np.fromfile(file_handle, uint32, 1)[0]
    count = int(data_size / np.dtype(dtype).itemsize)
    arr = np.fromfile(file_handle, dtype, count)
    final_block = np.fromfile(file_handle, uint32, 1)[0]
    assert data_size == final_block
    return arr


def read_ids_block(file_handle, count):
    """Read ID block with automatic uint32/uint64 selection."""
    data_size = np.fromfile(file_handle, uint32, 1)[0]
    file_handle.seek(-4, 1)
    count = int(count)
    if data_size / 4 == count:
        dtype = uint32
    elif data_size / 8 == count:
        dtype = uint64
    else:
        raise ValueError("Incorrect number of IDs requested")
    return read_fortran_block(file_handle, dtype)


def read_snapshot_binary(filename):
    """Read a legacy binary AREPO snapshot.

    Returns
    -------
    data_gas, data_sink, header : tuple
        Gas and sink dictionaries plus header dictionary.
    """
    with open(filename, mode="rb") as file_handle:
        data = {}
        data_gas = {}
        data_sink = {}

        header = read_header_binary(file_handle)
        nparts = header["num_particles"]
        total = nparts.sum()
        n_gas = nparts[0]
        n_sink = nparts[5]

        precision = float32
        data["pos"] = read_fortran_block(file_handle, precision).reshape((total, 3))
        data["vel"] = read_fortran_block(file_handle, precision).reshape((total, 3))
        data["id"] = read_ids_block(file_handle, total)
        data["mass"] = read_fortran_block(file_handle, precision)
        data["u_therm"] = read_fortran_block(file_handle, precision)
        data["rho"] = read_fortran_block(file_handle, precision)
        data["chem"] = read_fortran_block(file_handle, precision).reshape((n_gas, 3))
        data["tdust"] = read_fortran_block(file_handle, precision)

    for field in data:
        data_gas[field] = data[field][0:n_gas]

    data_sink["pos"] = data["pos"][n_gas:(n_gas + n_sink)]
    data_sink["vel"] = data["vel"][n_gas:(n_gas + n_sink)]
    data_sink["id"] = data["id"][n_gas:(n_gas + n_sink)]
    data_sink["mass"] = data["mass"][n_gas:(n_gas + n_sink)]
    return data_gas, data_sink, header


def mask_cylinder(x, y, R_max, z=None, z_max=None):
    """Boolean mask selecting particles within a cylinder centred on the origin.

    Parameters
    ----------
    x, y : array-like
        In-plane coordinates (code units, GC-centred).
    R_max : float
        Maximum cylindrical radius (same units as x, y).
    z : array-like, optional
        Vertical coordinates. If given with z_max, also applies a height cut.
    z_max : float, optional
        Maximum |z|. Ignored unless z is also provided.

    Returns
    -------
    mask : ndarray of bool
    """
    R = np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2)
    mask = R <= R_max
    if z is not None and z_max is not None:
        mask &= np.abs(np.asarray(z)) <= z_max
    return mask


def mask_box(x, y, z=None, xmin=None, xmax=None, ymin=None, ymax=None,
             zmin=None, zmax=None):
    """Boolean mask selecting particles within a rectangular box.

    All limit parameters are optional — omit any axis to leave it unconstrained.

    Returns
    -------
    mask : ndarray of bool
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.ones(x.shape, dtype=bool)
    if xmin is not None:
        mask &= x >= xmin
    if xmax is not None:
        mask &= x <= xmax
    if ymin is not None:
        mask &= y >= ymin
    if ymax is not None:
        mask &= y <= ymax
    if z is not None:
        z = np.asarray(z)
        if zmin is not None:
            mask &= z >= zmin
        if zmax is not None:
            mask &= z <= zmax
    return mask


def mask_galactic_window(l_deg, b_deg, l_lim=(-6.0, 6.0), b_lim=(-1.5, 1.5)):
    """Boolean mask selecting particles within a galactic l/b window.

    This is the same cut used by the legacy movie scripts before binning
    (e.g. ``tokeep = abs(l) < 6``) — generalised to also include a b limit.

    Parameters
    ----------
    l_deg, b_deg : array-like
        Galactic coordinates in degrees.
    l_lim : (float, float)
        (l_min, l_max) in degrees.
    b_lim : (float, float)
        (b_min, b_max) in degrees.

    Returns
    -------
    mask : ndarray of bool
    """
    l = np.asarray(l_deg)
    b = np.asarray(b_deg)
    return (l >= l_lim[0]) & (l <= l_lim[1]) & (b >= b_lim[0]) & (b <= b_lim[1])


class MyObject:
    """Simple dictionary-to-attributes container."""

    def __init__(self, values=None):
        if values is not None:
            for key, value in values.items():
                setattr(self, key, value)


def sink_snap_dtype(max_sne=2000, max_accretion_events=50):
    """Structured dtype for binary ``sink_snap_*`` files (with 8-byte alignment).

    Field order and sizes must match the simulation output / legacy ``pycstruct``
    layout. ``max_sne`` and ``max_accretion_events`` must match compile-time
    constants used when the run was written.

    Parameters
    ----------
    max_sne : int, optional
        Length of the ``explosion_time`` array per sink.
    max_accretion_events : int, optional
        Length of ``MassStillToConvert`` and ``AccretionTime`` per sink.

    Returns
    -------
    numpy.dtype
        Structured dtype with ``align=True``.
    """
    return np.dtype(
        [
            ("Pos", np.float64, (3,)),
            ("Vel", np.float64, (3,)),
            ("Accel", np.float64, (3,)),
            ("Mass", np.float64),
            ("FormationMass", np.float64),
            ("FormationTime", np.float64),
            ("ID", np.uint64),
            ("HomeTask", np.uint32),
            ("Index", np.uint32),
            ("FormationOrder", np.uint32),
            ("N_sne", np.uint32),
            ("StellarMass", np.float64),
            ("explosion_time", np.float64, (max_sne,)),
            ("MassStillToConvert", np.float64, (max_accretion_events,)),
            ("AccretionTime", np.float64, (max_accretion_events,)),
        ],
        align=True,
    )


def read_sink_snap_binary(
    filename,
    max_sne=2000,
    max_accretion_events=50,
    check_filesize=True,
):
    """Read a binary ``sink_snap_*`` file using NumPy only (no ``pycstruct``).

    Parameters
    ----------
    filename : str or os.PathLike
        Path to the sink snapshot file.
    max_sne, max_accretion_events : int, optional
        Passed to :func:`sink_snap_dtype`; must match the writer.
    check_filesize : bool, optional
        If True, verify the file is large enough for ``NSinks`` records.

    Returns
    -------
    out : dict
        ``time`` : float
            Simulation time from file header.
        ``NSinks`` : int
            Number of sink records.
        ``snap_num`` : int or None
            Last integer group parsed from the basename, if any.
        ``data`` : ndarray
            Structured array of length ``NSinks``.
        ``dtype`` : numpy.dtype
            The dtype used for reading.
    """
    filename = os.fspath(filename)
    dt = sink_snap_dtype(max_sne=max_sne, max_accretion_events=max_accretion_events)
    matches = re.findall(r"\d+", os.path.basename(filename))
    snap_num = int(matches[-1]) if matches else None

    with open(filename, "rb") as fhandle:
        time_arr = np.fromfile(fhandle, dtype=np.float64, count=1)
        if time_arr.size != 1:
            raise OSError(f"{filename}: could not read time header (float64).")
        time_val = float(time_arr[0])

        nsinks_arr = np.fromfile(fhandle, dtype=np.uint32, count=1)
        if nsinks_arr.size != 1:
            raise OSError(f"{filename}: could not read NSinks header (uint32).")
        nsinks = int(nsinks_arr[0])

        if check_filesize:
            pos = fhandle.tell()
            fhandle.seek(0, os.SEEK_END)
            file_size = fhandle.tell()
            fhandle.seek(pos, os.SEEK_SET)
            expected = pos + nsinks * dt.itemsize
            if file_size < expected:
                raise OSError(
                    f"{filename}: file too small for NSinks={nsinks} with this dtype.\n"
                    f"file_size={file_size} bytes, expected_at_least={expected} bytes.\n"
                    "Check max_sne / max_accretion_events against the simulation."
                )

        if nsinks == 0:
            rec = np.array([], dtype=dt)
        else:
            rec = np.fromfile(fhandle, dtype=dt, count=nsinks)
            if rec.size != nsinks:
                raise OSError(f"{filename}: expected {nsinks} records, got {rec.size}.")

    return {
        "time": time_val,
        "NSinks": nsinks,
        "snap_num": snap_num,
        "data": rec,
        "dtype": dt,
    }


def _sink_structured_to_legacy_lists(rec, time_header, nsinks_header):
    """Build movie-script dict (list per field) from structured sink records."""
    legacy = {}
    if rec.size == 0:
        for name in rec.dtype.names:
            legacy[name] = []
    else:
        for name in rec.dtype.names:
            col = rec[name]
            if col.ndim == 1:
                legacy[name] = [np.asarray(col[i]) for i in range(rec.shape[0])]
            else:
                legacy[name] = [np.asarray(col[i]) for i in range(rec.shape[0])]

    legacy["time"] = [time_header]
    legacy["NSinks"] = [nsinks_header]
    return legacy


def read_sink_snap(filename, max_sne=2000, max_accretion_events=50, check_filesize=True):
    """Read binary sink snapshot used by movie scripts.

    Uses :func:`read_sink_snap_binary` (NumPy structured arrays). Returns a
    :class:`MyObject` whose attributes match the legacy list-based layout
    (each field is a list over sinks, plus ``time`` and ``NSinks`` as
    single-element lists, and integer ``snap_num``).

    For structured-array access use :func:`read_sink_snap_binary` instead.

    Parameters
    ----------
    filename : str or os.PathLike
        Sink snapshot path.
    max_sne, max_accretion_events : int, optional
        Layout dimensions; must match the simulation output.
    check_filesize : bool, optional
        Passed through to :func:`read_sink_snap_binary`.

    Returns
    -------
    MyObject
        Attribute bag compatible with older movie scripts.
    """
    out = read_sink_snap_binary(
        filename,
        max_sne=max_sne,
        max_accretion_events=max_accretion_events,
        check_filesize=check_filesize,
    )
    time_hdr = np.array([out["time"]], dtype=np.float64)
    nsinks_hdr = np.array([out["NSinks"]], dtype=np.uint32)
    legacy = _sink_structured_to_legacy_lists(out["data"], time_hdr, nsinks_hdr)

    matches = re.findall(r"\d+", os.path.basename(filename))
    snap_int = int(matches[0]) if matches else -1

    legacy["snap_num"] = snap_int
    return MyObject(legacy)


def build_sink_data_sinkwise(sink_data_snapwise, snaps=None, require_ids_nonzero=True):
    """Reorganize snap-wise sink dicts into per-sink-ID time series.

    ``sink_data_snapwise`` maps snapshot index ``isnap`` to a dict like the return
    value of :func:`read_sink_snap_binary`, optionally with extra keys per snap
    (e.g. ``ParentDensity``, ``ParentDistance``) attached after reading gas.

    Parameters
    ----------
    sink_data_snapwise : dict
        ``{isnap: {"time": float, "data": structured_array, ...}, ...}``.
    snaps : array-like of int or None, optional
        Snapshot indices to include; default is sorted keys of ``sink_data_snapwise``.
    require_ids_nonzero : bool, optional
        If True, ignore sink records with ``ID == 0``.

    Returns
    -------
    dict
        ``sink_id ->`` dict with shared ``time`` and ``snaps`` arrays plus per-sink
        series ``pos``, ``mass``, ``rho_parent``, ``d_parent`` (nan if missing) and
        scalars ``formationTime``, ``formationMass``, ``first_snap``, ``first_index``.
    """
    if snaps is None:
        snaps = np.array(sorted(sink_data_snapwise.keys()), dtype=int)
    else:
        snaps = np.asarray(snaps, dtype=int)
    n = len(snaps)

    snap_t = np.full(n, np.nan, dtype=float)

    all_ids = set()
    for i, isnap in enumerate(snaps):
        rec = sink_data_snapwise[isnap]["data"]
        snap_t[i] = float(sink_data_snapwise[isnap]["time"])

        ids = np.asarray(rec["ID"], dtype=np.uint64)
        if require_ids_nonzero:
            ids = ids[ids != 0]
        all_ids.update(ids.tolist())

    sink_data_sinkwise = {}
    for sid in all_ids:
        sid = int(sid)
        sink_data_sinkwise[sid] = {
            "time": snap_t,
            "snaps": snaps,
            "pos": np.full((n, 3), np.nan, dtype=float),
            "mass": np.full(n, np.nan, dtype=float),
            "rho_parent": np.full(n, np.nan, dtype=float),
            "d_parent": np.full(n, np.nan, dtype=float),
            "formationTime": np.nan,
            "formationMass": np.nan,
            "first_snap": None,
            "first_index": None,
        }

    for i, isnap in enumerate(snaps):
        rec = sink_data_snapwise[isnap]["data"]
        ids = np.asarray(rec["ID"], dtype=np.uint64)

        rho = sink_data_snapwise[isnap].get("ParentDensity", None)
        dist = sink_data_snapwise[isnap].get("ParentDistance", None)

        if require_ids_nonzero:
            id_to_row = {int(sid): j for j, sid in enumerate(ids) if sid != 0}
        else:
            id_to_row = {int(sid): j for j, sid in enumerate(ids)}

        for sid, j in id_to_row.items():
            tr = sink_data_sinkwise.get(sid)
            if tr is None:
                continue

            tr["pos"][i, :] = np.asarray(rec["Pos"][j]).reshape(3)
            tr["mass"][i] = float(rec["Mass"][j])

            if rho is not None:
                tr["rho_parent"][i] = float(rho[j])
            if dist is not None:
                tr["d_parent"][i] = float(dist[j])

            if tr["first_snap"] is None:
                tr["first_snap"] = int(isnap)
                tr["first_index"] = int(i)
                tr["formationTime"] = float(np.asarray(rec["FormationTime"][j]).squeeze())
                tr["formationMass"] = float(np.asarray(rec["FormationMass"][j]).squeeze())

    return sink_data_sinkwise
