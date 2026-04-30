"""Snapshot I/O, unit conversions, and physical quantities."""

import gc
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


def read_sink_snap(filename, max_sne=2000, max_accretion_events=50):
    """Read binary sink snapshot used by movie scripts."""
    snap_num = re.findall(r"\d+", str(filename))

    import pycstruct

    struct = pycstruct.StructDef(alignment=8)
    struct.add("float64", "Pos", shape=3)
    struct.add("float64", "Vel", shape=3)
    struct.add("float64", "Accel", shape=3)
    struct.add("float64", "Mass")
    struct.add("float64", "FormationMass")
    struct.add("float64", "FormationTime")
    struct.add("uint64", "ID")
    struct.add("uint32", "HomeTask")
    struct.add("uint32", "Index")
    struct.add("uint32", "FormationOrder")
    struct.add("uint32", "N_sne")
    struct.add("float64", "StellarMass")
    struct.add("float64", "explosion_time", shape=max_sne)
    struct.add("float64", "MassStillToConvert", shape=max_accretion_events)
    struct.add("float64", "AccretionTime", shape=max_accretion_events)

    with open(filename, "rb") as file_handle:
        time = np.fromfile(file_handle, np.double, 1)
        nsinks = np.fromfile(file_handle, np.uint32, 1)

        inbytes = file_handle.read(struct.size())
        data = struct.deserialize(inbytes)

        for item in data:
            data[item] = [data[item]]

        for _ in range(nsinks[0] - 1):
            inbytes = file_handle.read(struct.size())
            row = struct.deserialize(inbytes)
            for item in data:
                data[item] += [row[item]]

    data["time"] = [time]
    data["NSinks"] = [nsinks]
    data["snap_num"] = int(snap_num[0]) if snap_num else -1
    return MyObject(data)
