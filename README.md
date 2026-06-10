# simviz

Utilities for plotting AREPO snapshots: B-field maps (Stokes parameters, LIC texture), gas projections (surface density, perspective fly-through), Galactic `l–b–v` histograms, and more.

## Install

```bash
cd Arepo_Visualizations   # repository root (contains pyproject.toml)
pip install -e .
```

Dependencies: NumPy, Matplotlib, h5py, SciPy (see `requirements.txt`)

Portable paths (repo root, `sample_snaps/`, etc.) live in `simviz.paths`. CMZ simulation data outside the repo defaults to `~/Research/Archive/Old_Code/arepo_CMZ/TS_2020`; override with `SIMVIZ_CMZ_DATA_DIR` if yours lives elsewhere.

## Examples

Notebooks in `examples/` are meant to run top to bottom after you give paths for your own snapshots:

| Notebook | Whats in it |
|----------|----------------|
| `bfield_demo.ipynb` | XY B-field orientation from gas cells |
| `bfield_planck_lic_demo.ipynb` | Three-panel + Planck-style `l–b` view |
| `lbv_demo.ipynb` | CO-weighted `l–b` and `l–v` maps |
| `surface_density_flythrough_demo.ipynb` | Camera path + PNG frames |

## Scripts

Long-running jobs live in `scripts/`. Run from the repo root with the package installed (`pip install -e .`).

### Fly-through movie (`render_flythrough_movie.py`)

Renders a PNG sequence from gas snapshots using a perspective surface-density projection. Camera paths:

| `--path` | Description |
|----------|-------------|
| `orbit` (default) | Tilted circular orbit with radial drift |
| `cinematic` | Keyframed zoom-in, orbit from above, dip to edge-on, exit below disk |
| `edge-orbit` | Edge-on (mock from-the-sun view) → 30° above plane → one orbit → back to edge-on |
| `zoom-observe` | Far-out galaxy view → zoom to CMZ → partial orbit → edge-on observational end |

Progress is printed to stdout with immediate flush (startup banner, per-snapshot load times, rolling frame progress with ETA). Projections default to **density-weighted column integration** (`--projection-method column`, same weighting idea as ``project_column_density_xy``); use `--projection-method surface` for the legacy 2D splat + masked_fill path.

On a cluster, use `tmux` and `python -u` so SSH drops do not lose output:

```bash
tmux new -s flythrough
python -u scripts/render_flythrough_movie.py \
  --snap-dir /path/to/snapshots/ \
  --snap-prefix phoenix_stinks_1Msun \
  --first-snap-number 820 \
  --last-snap-number 999 \
  --path edge-orbit \
  --n-frames 180 \
  --frames-per-snap 1 \
  --progress-every 10 \
  -o flythrough_frames \
  2>&1 | tee flythrough_render.log
```

This renders **180 frames** (one snapshot per frame, snaps 820–999), giving a **~7.5 s** movie at 24 fps with a smooth, looping camera path.

Encode frames with ffmpeg:

```bash
ffmpeg -y -framerate 24 -i flythrough_frames/frame_%04d.png \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
  -c:v libx264 -pix_fmt yuv420p flythrough.mp4
```

### Single-frame preview (`preview_flythrough_frame.py`)

Renders one frame with the same parameters as the movie script — useful for checking camera position or color scale before a long run.

## Demo output preview

Save rendered demo figures/GIFs to `example_output/` and they will render directly here.
Suggested filenames below match the current notebooks:

![B-field map demo](example_output/bfield_demo.png)
![Planck LIC demo](example_output/bfield_planck_lic_demo.png)
![LBV demo](example_output/lbv_demo.png)
![Surface density flythrough](example_output/surface_density_flythrough.gif)

## Package layout

| Module | Role |
|--------|------|
| `simviz.field_plots` | 2D maps, LIC, three-panel B-field plots |
| `simviz.projections` | Bar frame, Galactic coords, camera geometry |
| `simviz.utils` | Snapshot i/o, unit transforms, masking helpers |

