
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L2 验证: 全球 IHFC 52,000 测点 — 校正前后地形相关性检验

逻辑:
  校正前: q_measured 与局部地形起伏应该明显负相关 (山低/谷高)
  校正后: q_corrected = q/B 与地形起伏的相关性应消失
  统计: 六大洲各算一条回归线, 斜率和p值校正前后对比
"""
import sys, os; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy import stats
from scipy.ndimage import zoom
import rasterio
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from fem3d import Fem3D
from step3_train_large import FNO3D
import os

# ── IHFC 数据库路径 ──────────────────────────────────────────────────
# 不写死本地路径:优先取环境变量 IHFC_GHFDB,否则找本脚本同级的 data/。
# 数据本身从 GFZ Data Services 公开获取(doi:10.5880/fidgeo.2024.014)。
def _ihfc_path():
    p = os.environ.get('IHFC_GHFDB')
    if p and os.path.exists(p):
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, '..', 'IHFC_2024_full', 'IHFC_2024_GHFDB_v.2026.03.txt'),
                 os.path.join(here, 'data', 'IHFC_2024_GHFDB_v.2026.03.txt'),
                 os.path.join(here, 'IHFC_2024_GHFDB_v.2026.03.txt')):
        if os.path.exists(cand):
            return cand
    raise SystemExit(
        'IHFC database not found.\n'
        '  Set the environment variable IHFC_GHFDB to the file path, or place\n'
        '  IHFC_2024_GHFDB_v.2026.03.txt under ./data/ next to this script.\n'
        '  The dataset is openly available from GFZ Data Services, '
        'doi:10.5880/fidgeo.2024.014')



RES, FN, CELL_M, TILE_KM = 128, 40, 464, 57
DEM_PATH = os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif'))

# ── 加载FNO模型 ──
ck = torch.load('fno3d_large_model.pt', map_location='cuda', weights_only=False)
MODEL = FNO3D(modes=24, width=96, H_out=FN+1, W_out=FN+1, n_layers=6).cuda().eval()
MODEL.load_state_dict(ck['state'])
X_MEAN, X_STD = ck['X_mean'], ck['X_std']
Y_MEAN, Y_STD = ck['Y_mean'], ck['Y_std']

# ── 六大洲边界 ──
CONTINENTS = {
    'Asia':      (60, 150, 10, 55),
    'Europe':    (-10, 60, 35, 72),
    'N. America': (-170, -50, 15, 72),
    'S. America': (-80, -35, -55, 10),
    'Africa':    (-20, 55, -35, 38),
    'Oceania':   (110, 155, -45, -10),
}
CONT_NAMES = list(CONTINENTS.keys())

def taper_dem(dem):
    zm = dem.mean(); h, w = dem.shape
    k = max(2, int(h * 0.12))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, k)))
    wy, wx = np.ones(h), np.ones(w)
    wy[:k] = ramp; wy[-k:] = ramp[::-1]; wx[:k] = ramp; wx[-k:] = ramp[::-1]
    return zm + (dem - zm) * np.outer(wy, wx)

def process_point(dem_ds, lon, lat):
    """从DEM切57km窗口→重采样→taper→返回处理好的128×128数组
    同时返回原始窗口的起伏。如果失败返回 (None, None)."""
    half_px = int(TILE_KM * 500 / CELL_M)
    try:
        r, c = dem_ds.index(lon, lat)
        w = ((r - half_px, r + half_px), (c - half_px, c + half_px))
        raw = dem_ds.read(1, window=w).astype(float)
    except Exception:
        return None, None

    raw[raw < -1000] = np.nan
    if raw.size < 100: return None, None
    if np.isnan(raw).mean() > 0.5: return None, None

    raw_relief = np.nanmax(raw) - np.nanmin(raw)
    mask = ~np.isfinite(raw)
    if mask.any(): raw[mask] = np.nanmean(raw)
    dem = zoom(raw, (RES / raw.shape[0], RES / raw.shape[1]), order=1)
    dem = taper_dem(dem)
    return dem.astype(np.float32), raw_relief

def fno_predict_batch(dems):
    """批量FNO推理, 返回每个瓦片中心3×3的B均值"""
    N = len(dems)
    x_batch = (dems.reshape(N, 1, 128, 128).astype(np.float32) - X_MEAN) / X_STD
    with torch.no_grad():
        pred = MODEL(torch.tensor(x_batch, dtype=torch.float32).cuda())
    B_grids = pred.cpu().numpy() * Y_STD + Y_MEAN  # (N, 1, FN+1, FN+1)
    c = FN // 2
    B_vals = B_grids[:, 0, c-2:c+3, c-2:c+3].mean(axis=(1, 2))
    return B_vals

# ── 读取IHFC数据 ──
print('加载 IHFC 数据...')
with open(_ihfc_path(),
          'r', encoding='latin-1') as f:
    lines = f.readlines()

records = []
for l in lines[10:]:
    d = l.strip().split('\t')
    if len(d) < 50: continue
    try:
        lon = float(d[3]) if d[3].strip() and d[3].strip() != '-' else None
        lat = float(d[4]) if d[4].strip() and d[4].strip() != '-' else None
        q = float(d[13]) if d[13].strip() and d[13].strip() != '-' else None  # C1
    except: continue
    if None in (lon, lat, q): continue
    if not (-180 <= lon <= 180 and -90 <= lat <= 90): continue
    if q <= 0 or q > 500: continue
    # 质量过滤: C37 >= 3 (或不存在C37的数据默认为合格)
    quality = d[49].strip() if len(d) > 49 else ''
    try:
        qs = int(quality) if quality != '-' and quality != '' else 3
    except: qs = 3
    if qs < 3: continue
    records.append((lon, lat, q))

print(f'IHFC 合格测点: {len(records)} (质量>=3, q in (0,500])')

# 统计抽样: 每洲最多1500点, 六大洲均匀覆盖
np.random.seed(42)
sampled = []
for cname, (lomin, lomax, lamin, lamax) in CONTINENTS.items():
    in_cont = [(lon, lat, q) for (lon, lat, q) in records
               if lomin <= lon <= lomax and lamin <= lat <= lamax]
    n_take = min(1500, len(in_cont))
    if n_take > 0:
        idxs = np.random.choice(len(in_cont), n_take, replace=False)
        sampled.extend([in_cont[i] for i in idxs])
records = sampled
np.random.shuffle(records)
print(f'抽样后: {len(records)} 点 (每洲上限1500)')

# ── 逐点处理 ──
dem_ds = rasterio.open(DEM_PATH)
results = []
batch_size = 64
batch_dems, batch_idxs = [], []
N_total = len(records)

for idx, (lon, lat, q) in enumerate(records):
    dem, relief = process_point(dem_ds, lon, lat)
    if dem is not None and relief is not None:
        batch_dems.append(dem)
        batch_idxs.append((idx, q, relief))
    if len(batch_dems) >= batch_size or (idx == N_total - 1 and batch_dems):
        B_vals = fno_predict_batch(np.array(batch_dems))
        for bi, (bidx, q0, dr) in enumerate(batch_idxs):
            lon0, lat0, _ = records[bidx]
            B = float(B_vals[bi])
            if 0.5 < B < 2.0:
                results.append((lon0, lat0, dr, q0, B, q0 / B))
        batch_dems, batch_idxs = [], []

    if (idx + 1) % 1000 == 0:
        print(f'  {idx+1}/{N_total}, 有效: {len(results)}')

dem_ds.close()
print(f'最终有效: {len(results)} 条 (lon, lat, relief, q, B, q_corr)')

# ── 统计分析 ──
data = np.array([(r[2], r[3], r[4], r[5], r[0], r[1])
                 for r in results], dtype=np.float64)
relief = data[:, 0]; q_meas = data[:, 1]; B_val = data[:, 2]; q_corr = data[:, 3]
lon_arr = data[:, 4]; lat_arr = data[:, 5]

print(f'\n=== 全局统计 ===')
print(f'B: [{B_val.min():.3f}, {B_val.max():.3f}], mean={B_val.mean():.3f}')
print(f'q_meas: [{q_meas.min():.0f}, {q_meas.max():.0f}], mean={q_meas.mean():.1f}')
print(f'q_corr: [{q_corr.min():.0f}, {q_corr.max():.0f}], mean={q_corr.mean():.1f}')

# 校正前回归
slope_before, int_before, r_before, p_before, _ = stats.linregress(relief, q_meas)
slope_after, int_after, r_after, p_after, _ = stats.linregress(relief, q_corr)
print(f'\n校正前: slope={slope_before:.4f}, r={r_before:.3f}, p={p_before:.2e}')
print(f'校正后: slope={slope_after:.4f}, r={r_after:.3f}, p={p_after:.2e}')

# ── 分六大洲 ──
print(f'\n=== 分洲统计 ===')
continent_stats = {}
for cname, (lomin, lomax, lamin, lamax) in CONTINENTS.items():
    mask = (lon_arr >= lomin) & (lon_arr <= lomax) & (lat_arr >= lamin) & (lat_arr <= lamax)
    if mask.sum() < 50:
        continent_stats[cname] = None
        print(f'{cname}: {mask.sum()} 点 (太少, 跳过)')
        continue
    rl = relief[mask]; qm = q_meas[mask]; qc = q_corr[mask]
    sb, _, rb, pb, _ = stats.linregress(rl, qm)
    sa, _, ra, pa, _ = stats.linregress(rl, qc)
    continent_stats[cname] = {
        'n': int(mask.sum()), 'slope_before': sb, 'slope_after': sa,
        'p_before': pb, 'p_after': pa, 'relief': rl, 'q_meas': qm, 'q_corr': qc
    }
    print(f'{cname}: {mask.sum()} 点, '
          f'校正前 slope={sb:.4f} (p={pb:.1e}), '
          f'校正后 slope={sa:.4f} (p={pa:.1e})')

# ── 保存结果 ──
np.savez('l2_validation_results.npz',
         relief=relief, q_meas=q_meas, B=B_val, q_corr=q_corr,
         lon=lon_arr, lat=lat_arr,
         slope_before=slope_before, slope_after=slope_after,
         p_before=p_before, p_after=p_after,
         r_before=r_before, r_after=r_after)
print('\n结果已保存: l2_validation_results.npz')
