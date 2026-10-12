
import os
import sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, rasterio
from scipy.ndimage import zoom
from fem3d import Fem3D

DEM_PATH = os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif'))
NX = 40  # production resolution

SITES = [
    ('Andes-1', -71.5483, -39.3500),
    ('Andes-2', -70.0483, -33.8317),
    ('Himalaya-1', 103.8619, 31.2325),
    ('Himalaya-2', 97.9133, 39.4467),
    ('Alps-1', 6.8833, 45.8500),
    ('Alps-2', 10.1100, 46.7633),
    ('EAfricaRift-1', 32.0500, -2.8833),
    ('EAfricaRift-2', 31.2743, 1.6883),
    ('NAmCordillera-1', -123.1600, 49.6117),
    ('NAmCordillera-2', -121.6500, 47.6833),
]

def extract_window(ds, lon, lat, half_km, res_px=128):
    # extract a raw window of given half-width (km), resample to res_px x res_px
    try:
        r, c = ds.index(lon, lat)
    except Exception:
        return None, None
    # approx: dataset resolution
    t = ds.transform
    px_m = abs(t.a) * 111320 * np.cos(np.radians(lat))  # rough m/pixel at this lat for lon-degree pixel size... 
    # Instead use dataset's actual pixel size in meters directly if in projected CRS; here DEM is 450m/px lonlat SRTM15+ tif
    cell_m = 450.0
    half_px = int(half_km * 1000 / cell_m)
    try:
        w = ((r - half_px, r + half_px), (c - half_px, c + half_px))
        raw = ds.read(1, window=w).astype(float)
    except Exception:
        return None, None
    raw[raw < -1000] = np.nan
    if raw.size < 100 or np.isnan(raw).mean() > 0.5:
        return None, None
    mask = ~np.isfinite(raw)
    if mask.any(): raw[mask] = np.nanmean(raw)
    dem = zoom(raw, (res_px/raw.shape[0], res_px/raw.shape[1]), order=1)
    actual_cell_m = (2*half_km*1000) / res_px
    return dem.astype(np.float32), actual_cell_m

def center_B(dem, cell_m, depth=None):
    f = Fem3D(nx=NX, ny=NX, nz=10)
    f.run(dem, cell_m, depth=depth, K=2.5)
    c = f.B_grid.shape[0]//2
    return float(f.B_grid[c, c]), float(dem.max()-dem.min())

print('=== FOOTPRINT CONVERGENCE (expanded to 10 real sites, nx=40) ===', flush=True)
ds = rasterio.open(DEM_PATH)
footprint_results = {}
t0=time.time()
for name, lon, lat in SITES:
    row = {}
    for half_km in [10, 20, 40, 28.5, 50]:  # 28.5 half-width = 57km full footprint (production)
        dem, cell_m = extract_window(ds, lon, lat, half_km)
        if dem is None:
            row[half_km] = None; continue
        B, relief = center_B(dem, cell_m)
        row[half_km] = (B, relief)
    footprint_results[name] = row
    print(f'[{time.time()-t0:.0f}s] {name} done', flush=True)

print('\nSite            10km(B,relief)      20km               40km               57km(prod)         100km')
for name, row in footprint_results.items():
    def fmt(k):
        v = row.get(k)
        return f'{v[0]:.3f}(r{v[1]:.0f}m)' if v else 'N/A'
    print(f'{name:16s} {fmt(10):18s} {fmt(20):18s} {fmt(40):18s} {fmt(28.5):18s} {fmt(50):18s}')

print('\n=== DEPTH CONVERGENCE (expanded to 10 real sites, footprint=57km, nx=40, nz scaled) ===', flush=True)
depth_results = {}
for name, lon, lat in SITES:
    dem, cell_m = extract_window(ds, lon, lat, 28.5)
    if dem is None:
        depth_results[name] = None; continue
    row = {}
    for depth, nz in [(3000,10),(9000,14),(25000,20)]:
        f = Fem3D(nx=NX, ny=NX, nz=nz)
        f.run(dem, cell_m, depth=depth, K=2.5)
        c = f.B_grid.shape[0]//2
        row[depth] = float(f.B_grid[c,c])
    depth_results[name] = row
    print(f'[{time.time()-t0:.0f}s] {name} depth-series done', flush=True)

print('\nSite             3000m      9000m      25000m')
for name, row in depth_results.items():
    if row is None:
        print(f'{name:16s} N/A'); continue
    print(f'{name:16s} {row[3000]:.4f}    {row[9000]:.4f}    {row[25000]:.4f}')

ds.close()
np.savez('footprint_depth_expand_results.npz', footprint=footprint_results, depth=depth_results)
print('\nDONE', flush=True)
