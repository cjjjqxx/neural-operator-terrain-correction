# Neural-operator terrain correction of global heat flow

Code for the manuscript "Global terrain correction of continental heat flow with a neural-operator surrogate of three-dimensional heat-conduction simulations" (Junji Chen et al., submitted to JGR: Solid Earth).
Licence: MIT. Data products: Zenodo https://doi.org/10.5281/zenodo.23311169.

## What is here

| Group | Scripts |
|---|---|
| Finite-element solver and tiles | `fem3d.py` (scikit-fem steady conduction with linear tetrahedra, six per terrain-following hexahedral cell), `tile_reader.py` (57 km tiles from SRTM15+), `step2_parallel.py` (batch generation of the training set) |
| Neural operator | `step3_train_large.py` (FNO definition, D4 eight-orientation averaging, training) |
| Applying the correction | `correct_heatflow.py` (correct your own sites), `fetch_dem.py` (prepare the elevation grid), `test_correct_heatflow.py` (regression test against the released results) |
| Global correction and products | `rebuild_correction.py`, `global_B_field_05deg.py`, `krige_global_ok.py`, `site_reliability.py`, `error_model.py`, `error_budget.py` |
| Validation and tests | accuracy (`validate_independent*.py`, `holdout_d4_r2.py`, `linear_spectral_baseline.py`), numerical convergence (`analytic_benchmark*.py`, `mesh_*`, `vertical_mesh_test.py`, `footprint_*`), uncertainty calibration (`mc_*`, `dropout_calibration_test.py`), observational tests (`paired_*`, `c15_*`, `davies_*`, `magnitude_scaling_test.py`), valley siting bias (`valley_*`, `random_land_baseline.py`, `province_artifact_test.py`) |
| Figures | `figures/v2/*.py` with the shared style in `figures/nature_style_v2.py` |

Comments in the code are partly in Chinese.

## Requirements

Python 3.11 with the packages in `requirements.txt` (versions used for the paper). A CUDA GPU speeds up the operator
but is not required for `correct_heatflow.py`.

## Input data (not included)

Third-party data, obtained from their original sources:

- IHFC Global Heat Flow Database, Release 2024 (V. 2026.03): https://doi.org/10.5880/fidgeo.2024.014.
  Set `IHFC_GHFDB` to the path of `IHFC_2024_GHFDB_v.2026.03.txt`, or place it in `data/`.
- SRTM15+ global topography (Tozer et al., 2019): https://doi.org/10.1029/2019EA000658. The exact grid used in the
  paper (ocean set to 0 m) is included in the Zenodo data record as `SRTM15_450m.tif`; place it at
  `data/SRTM15_450m.tif` or set `SRTM15_DEM` to its path. `fetch_dem.py` can instead download SRTM15+ from its source
  and check a raster against the requirements.
- Only for the external tests: the 2° equal-area heat-flow table of Davies (2013, https://doi.org/10.1002/ggge.20271,
  supporting information), saved as `ggge20271-sup-0001-data_table1_eq_area_global_heat_flow.csv`; and the 0.5° kriged
  field of Petrunin et al. (2026, https://doi.org/10.5880/GFZ.KHAG.2026.004), placed at
  `ghfdb2024_official/IHFC_2024_GHFDB_05x05_WGS1984_Kriging.csv`.

From the Zenodo data record of this paper: place `operator_weights.pt` as `fno3d_large_model.pt` and
`fem_training_set.npz` as `global_fem_dataset.npz` in this directory, put `SRTM15_450m.tif` in `data/`, and unzip
`intermediate_products.zip` here.

## Quick start: correct your own sites

`python test_correct_heatflow.py` first checks the installation: it corrects 200 sites of the released dataset
(`data/sites_terrain_corrected.csv`, or set `HEATFLOW_SITES`) and compares the result with the released values.

```bash
python correct_heatflow.py --sites my_sites.csv --dem "$SRTM15_DEM" -o corrected.csv
```

The input needs longitude, latitude and measured heat flow (and, if available, the borehole interval depth).

## Reproducing the paper

With the inputs above in place, `run_all.sh` reruns the global correction of all sites, the kriged fields, the
statistical and observational tests, the reliability grades, `numbers.json` (every number quoted in the paper,
written by `emit_numbers.py`) and the main figures, in dependency order; logs go to `logs/`.

The intermediate products come from runs that take hours of finite-element or Monte Carlo computation. Each can be
regenerated with the script listed for it in `MANIFEST.csv` inside `intermediate_products.zip`; regenerating the
training set (`step2_parallel.py`) and retraining the operator (`step3_train_large.py`) are likewise optional.
