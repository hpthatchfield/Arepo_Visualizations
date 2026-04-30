# simviz

Utilities for plotting AREPO snapshots: B-field maps (Stokes parameters, LIC texture), gas projections (surface density, perspective fly-through), Galactic `l–b–v` histograms, and more.

## Install

```bash
cd simviz   # directory that contains pyproject.toml
pip install -e .
```

Dependencies: NumPy, Matplotlib, h5py, SciPy (see `requirements.txt`)

## Examples

Notebooks in `examples/` are meant to run top to bottom after you give paths for your own snapshots:

| Notebook | Whats in it |
|----------|----------------|
| `bfield_demo.ipynb` | XY B-field orientation from gas cells |
| `bfield_planck_lic_demo.ipynb` | Three-panel + Planck-style `l–b` view |
| `lbv_demo.ipynb` | CO-weighted `l–b` and `l–v` maps |
| `surface_density_flythrough_demo.ipynb` | Camera path + PNG frames |

## Package layout

| Module | Role |
|--------|------|
| `simviz.field_plots` | 2D maps, LIC, three-panel B-field plots |
| `simviz.projections` | Bar frame, Galactic coords, camera geometry |
| `simviz.utils` | Snapshot i/o, unit transforms, masking helpers |

