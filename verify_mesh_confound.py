import os
import sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, rasterio
from scipy.ndimage import zoom
from fem3d import Fem3D

DEM_PATH = os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif'))

def extract_window(ds, lon, lat, half_km, res_px=128):
    try:
        r, c = ds.index(lon, lat)
    except Exception:
        return None, None
    cell_m = 450.0
    half_px = int(half_km * 1000 / cell_m)
    if half_px < 4: half_px = 4
    # 统一走 tile_reader:窗口按真实公里数取,经向像元数按 cos(纬度) 放大,
    # 因此各档窗口都是正方形,窗口收敛曲线才不会混入纬度畸变。
    from tile_reader import read_tile
    tile_km = 2.0 * half_km
    dem = read_tile(ds, lon, lat, tile_km=tile_km, res=res_px)
    if dem is None:
        return None, None
    return dem.astype(np.float32), tile_km * 1000 / res_px

def center_B(dem, cell_m, nx):
    f = Fem3D(nx=nx, ny=nx, nz=10)
    f.run(dem, cell_m, depth=None, K=2.5)
    c = f.B_grid.shape[0]//2
    return float(f.B_grid[c, c])

# 载入已有300点结果, 每个起伏档抽10个点复用坐标(保证可比)
results = np.load('footprint_stability_300_results.npy', allow_pickle=True).item()
rng = np.random.default_rng(7)
bins_order = ['near_flat', 'moderate', 'high', 'extreme']
sample = []
for rc in bins_order:
    keys = [k for k in results if k[0] == rc]
    chosen = rng.choice(len(keys), size=min(10, len(keys)), replace=False)
    for i in chosen:
        rc2, lon, lat = keys[i]
        b57_nx40 = results[keys[i]].get(57)
        b100_nx40 = results[keys[i]].get(100)
        sample.append((rc, lon, lat, b57_nx40, b100_nx40))

print(f'抽样{len(sample)}个点做网格分辨率混淆验证', flush=True)

ds = rasterio.open(DEM_PATH)
NX_MATCHED = 70  # 100km/70 ≈ 1.43 km/node, 与57km/40 ≈ 1.425 km/node 匹配

out = []
t0 = time.time()
for i, (rc, lon, lat, b57_nx40, b100_nx40) in enumerate(sample):
    dem, cell_m = extract_window(ds, lon, lat, 50.0)  # 100km full width
    if dem is None:
        continue
    b100_nx70 = center_B(dem, cell_m, NX_MATCHED)
    out.append((rc, lon, lat, b57_nx40, b100_nx40, b100_nx70))
    print(f'[{time.time()-t0:.0f}s] {i+1}/{len(sample)} {rc}: B57(nx40)={b57_nx40:.4f} '
          f'B100(nx40)={b100_nx40:.4f} B100(nx70)={b100_nx70:.4f}', flush=True)

ds.close()
np.save('verify_mesh_confound_results.npy', out, allow_pickle=True)

print('\n=== 汇总: |B100(nx40)-B57| vs |B100(nx70)-B57| ===', flush=True)
for rc in bins_order:
    rows = [r for r in out if r[0] == rc]
    if not rows: continue
    d_nx40 = np.array([abs(r[4]-r[3]) for r in rows])
    d_nx70 = np.array([abs(r[5]-r[3]) for r in rows])
    print(f'{rc}: n={len(rows)}  原nx40误差中位数={np.median(d_nx40):.4f}  '
          f'匹配分辨率nx70误差中位数={np.median(d_nx70):.4f}  '
          f'误差缩小比例={100*(1-np.median(d_nx70)/max(np.median(d_nx40),1e-9)):.0f}%')
print('\nDONE', flush=True)
