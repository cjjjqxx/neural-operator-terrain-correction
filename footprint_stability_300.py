import os
import sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, rasterio
from scipy.ndimage import zoom
from fem3d import Fem3D

DEM_PATH = os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif'))
NX = 40  # production resolution

# ---- 1. 按|B-1|分层随机抽样300个真实测点 ----
d = np.load('rebuild_points.npz')
lon_all, lat_all, B_all = d['lon'], d['lat'], d['B']
absB1 = np.abs(B_all - 1)

rng = np.random.default_rng(42)
bins = [('near_flat', 0, 0.02), ('moderate', 0.02, 0.1), ('high', 0.1, 0.2), ('extreme', 0.2, 10)]
N_PER_BIN = 75

sites = []
for name, lo, hi in bins:
    idx = np.where((absB1 >= lo) & (absB1 < hi))[0]
    chosen = rng.choice(idx, size=min(N_PER_BIN, len(idx)), replace=False)
    for i in chosen:
        sites.append((name, float(lon_all[i]), float(lat_all[i]), float(B_all[i])))

print(f'总抽样测点数: {len(sites)}', flush=True)
for name, lo, hi in bins:
    print(f'  {name}: {sum(1 for s in sites if s[0]==name)}')

# ---- 2. 多窗口FEM ----
FULL_WIDTHS_KM = [5, 10, 20, 40, 57, 100]
HALF_KM = {5: 2.5, 10: 5, 20: 10, 40: 20, 57: 28.5, 100: 50}

def extract_window(ds, lon, lat, half_km, res_px=128):
    try:
        r, c = ds.index(lon, lat)
    except Exception:
        return None, None
    cell_m = 450.0
    half_px = int(half_km * 1000 / cell_m)
    if half_px < 4:
        half_px = 4
    # 统一走 tile_reader:窗口按真实公里数取,经向像元数按 cos(纬度) 放大,
    # 因此各档窗口都是正方形,窗口收敛曲线才不会混入纬度畸变。
    from tile_reader import read_tile
    tile_km = 2.0 * half_km
    dem = read_tile(ds, lon, lat, tile_km=tile_km, res=res_px)
    if dem is None:
        return None, None
    return dem.astype(np.float32), tile_km * 1000 / res_px

def center_B(dem, cell_m):
    f = Fem3D(nx=NX, ny=NX, nz=10)
    f.run(dem, cell_m, depth=None, K=2.5)
    c = f.B_grid.shape[0]//2
    return float(f.B_grid[c, c])

ds = rasterio.open(DEM_PATH)
results = {}  # (relief_class, lon, lat) -> {width_km: B or None}
t0 = time.time()
n_done = 0
for relief_class, lon, lat, B_prod in sites:
    row = {}
    for w_km in FULL_WIDTHS_KM:
        dem, cell_m = extract_window(ds, lon, lat, HALF_KM[w_km])
        if dem is None:
            row[w_km] = None
            continue
        try:
            row[w_km] = center_B(dem, cell_m)
        except Exception as e:
            row[w_km] = None
    results[(relief_class, lon, lat)] = row
    n_done += 1
    if n_done % 10 == 0:
        elapsed = time.time() - t0
        eta = elapsed / n_done * (len(sites) - n_done)
        print(f'[{elapsed:.0f}s] {n_done}/{len(sites)} done, ETA {eta:.0f}s', flush=True)

ds.close()
np.save('footprint_stability_300_results.npy', results, allow_pickle=True)
print(f'\n总耗时: {time.time()-t0:.0f}s', flush=True)

# ---- 3. 统计汇总 ----
print('\n=== 相邻窗口宽度之间的相对变化统计 ===', flush=True)
pairs = [(5,10), (10,20), (20,40), (40,57), (57,100)]
for relief_class, _, _ in [(b[0], None, None) for b in bins]:
    print(f'\n--- {relief_class} ---')
    site_rows = [row for (rc, lo, la), row in results.items() if rc == relief_class]
    for w1, w2 in pairs:
        rel_changes = []
        sign_flips = 0
        n_valid = 0
        for row in site_rows:
            b1, b2 = row.get(w1), row.get(w2)
            if b1 is None or b2 is None: continue
            n_valid += 1
            rel_changes.append(abs(b2-b1)/abs(b1-1) if abs(b1-1)>1e-6 else 0)
            if (b1-1)*(b2-1) < 0:
                sign_flips += 1
        if n_valid == 0:
            print(f'  {w1}->{w2}km: 无有效数据'); continue
        rel_changes = np.array(rel_changes)
        big_change = (rel_changes > 0.10).mean()*100
        print(f'  {w1}->{w2}km: n={n_valid}, 相对变化中位数={np.median(rel_changes)*100:.1f}%, '
              f'>10%变化占比={big_change:.1f}%, 符号翻转={sign_flips}/{n_valid}')

print('\nDONE', flush=True)
