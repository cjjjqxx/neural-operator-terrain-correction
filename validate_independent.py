#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立验证: 500个全新瓦片 (seed=123) FEM vs FNO"""
import sys, os, time, argparse; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import rasterio
from tile_reader import read_tile, CELL_M
import torch
from fem3d import Fem3D
from step3_train_large import FNO3D, compute_r2

# 有限元单元尺寸取 tile_reader.CELL_M:瓦片已重采样为 57 km / 128 像元(约 445 m),
# 与训练集 step2_parallel.py 同一口径。
RES, FN = 128, 40  # FN须与训练数据集FN一致
SEED_VAL = 123
DEM_DEFAULT = os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif'))
CONTINENTS = {
    'Asia': (60,150,10,55), 'Europe': (-10,60,35,72),
    'N_America': (-170,-50,15,72), 'S_America': (-80,-35,-55,10),
    'Africa': (-20,55,-35,38), 'Oceania': (110,155,-45,-10),
}

def process_one_tile(args):
    lon, lat, idx, dem_path = args
    try:
        with rasterio.open(dem_path) as ds:
            dem = read_tile(ds, lon, lat, reject_ocean=True)
        if dem is None: return None
        if dem.max() - dem.min() < 25: return None
        relief = dem.max() - dem.min()
        if relief < 15: return None
        f = Fem3D(nx=FN, ny=FN, nz=10)
        f.build(dem, CELL_M, depth=max(3000, relief*1.5))
        f.solve_heat()
        Bv = f.compute_B()
        if Bv.min() < 0.01 or Bv.max() > 10 or not np.all(np.isfinite(Bv)):
            return None
        return {'dem': dem.astype(np.float32), 'B_fem': Bv.astype(np.float32),
                'relief': relief, 'lon': lon, 'lat': lat, 'idx': idx}
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dem', default=DEM_DEFAULT, help='SRTM15+ 高程栅格(EPSG:4326)')
    a = ap.parse_args()
    print('加载FNO模型...')
    ck = torch.load('fno3d_large_model.pt', map_location='cuda', weights_only=False)
    model = FNO3D(modes=24, width=96, H_out=FN+1, W_out=FN+1, n_layers=6).cuda().eval()
    model.load_state_dict(ck['state'])
    X_mean, X_std = ck['X_mean'], ck['X_std']
    Y_mean, Y_std = ck['Y_mean'], ck['Y_std']

    # ── 生成候选 → 并行FEM ──
    print('生成500个独立验证瓦片 (seed=123, FEM并行)...')
    rng = np.random.RandomState(SEED_VAL)
    continents_list = list(CONTINENTS.items())
    results = []
    tasks_submitted = 0
    t0 = time.time()

    n_workers = max(1, os.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        while len(results) < 500:
            cands = []
            for _ in range(100):
                ci = rng.randint(0, len(continents_list))
                nm, (lomin, lomax, lamin, lamax) = continents_list[ci]
                cands.append((rng.uniform(lomin, lomax), rng.uniform(lamin, lamax)))
            chunk_tasks = [(lon, lat, tasks_submitted + i, a.dem)
                           for i, (lon, lat) in enumerate(cands)]
            futures = [pool.submit(process_one_tile, t) for t in chunk_tasks]
            # 按候选序号而非完成先后收集:最后一批只取前若干个,
            # 按完成顺序截断会让同一种子每次得到不同的 500 块。
            chunk = [f.result() for f in as_completed(futures)]
            results.extend(sorted((r for r in chunk if r is not None), key=lambda r: r['idx']))
            tasks_submitted += len(cands)
            pct = len(results)/max(1, tasks_submitted)*100
            sys.stdout.write(f'\r  {len(results)}/500, 已试 {tasks_submitted}, 通过率 {pct:.1f}%')
            sys.stdout.flush()

    print()
    elapsed = time.time() - t0
    print(f'FEM完成: {len(results[:500])} 瓦片, 耗时 {elapsed/60:.1f}min')

    results = results[:500]

    # 地形分布
    reliefs = np.array([r['relief'] for r in results])
    bins = [0,50,100,200,500,10000]
    names = ['<50m','50-100m','100-200m','200-500m','>500m']
    for tl, th, nm in zip(bins[:-1], bins[1:], names):
        print(f'  {nm}: {int(((reliefs>=tl)&(reliefs<th)).sum())}张')

    # ── FNO批量推理 ──
    print('\nFNO批量推理...')
    dems = np.stack([r['dem'] for r in results])
    B_fem_grids = np.stack([r['B_fem'] for r in results]).reshape(500, FN+1, FN+1)

    x_batch = (dems.reshape(-1, 1, 128, 128).astype(np.float32) - X_mean) / X_std
    with torch.no_grad():
        pred = model(torch.tensor(x_batch, dtype=torch.float32).cuda())
    B_fno_grids = pred.cpu().numpy()[:, 0, :, :] * Y_std + Y_mean

    # ── 逐像素统计 ──
    B_fem_flat = B_fem_grids.reshape(500, -1)
    B_fno_flat = B_fno_grids.reshape(500, -1)

    r2_total = compute_r2(B_fno_flat.reshape(1,1,500,-1),
                          B_fem_flat.reshape(1,1,500,-1))
    print(f'\n=== 独立验证结果 (seed=123, 500瓦片) ===')
    print(f'整体逐像素 R²: {r2_total:.4f}')

    # 每瓦片R²
    per_tile_r2 = []
    for i in range(500):
        r2 = compute_r2(B_fno_flat[i:i+1].reshape(1,1,1,-1),
                        B_fem_flat[i:i+1].reshape(1,1,1,-1))
        per_tile_r2.append(r2)
    per_tile_r2 = np.array(per_tile_r2)
    print(f'每瓦片R²: median={np.median(per_tile_r2):.4f}, mean={per_tile_r2.mean():.4f}')
    print(f'  >0.95: {(per_tile_r2>0.95).sum()}, >0.90: {(per_tile_r2>0.90).sum()}, '
          f'0.80-0.90: {((per_tile_r2>=0.8)&(per_tile_r2<0.9)).sum()}, '
          f'<0.60: {(per_tile_r2<0.6).sum()}')

    # 分地形
    for tl, th, nm in zip(bins[:-1], bins[1:], names):
        mask = (reliefs>=tl) & (reliefs<th)
        if mask.sum() > 0:
            r2c = compute_r2(B_fno_flat[mask].reshape(1,1,-1),
                             B_fem_flat[mask].reshape(1,1,-1))
            mae = np.abs(B_fno_flat[mask] - B_fem_flat[mask]).mean()
            print(f'  {nm}: {mask.sum()}张, R²={r2c:.4f}, MAE={mae:.3f}')

    # 中心点B值
    from scipy import stats
    c = FN // 2
    B_fc = B_fem_grids[:, c, c]
    B_fnc = B_fno_grids[:, c, c]
    slope, intercept, r_val, p_val, _ = stats.linregress(B_fc, B_fnc)
    print(f'\n中心点B值: B_fem=[{B_fc.min():.3f},{B_fc.max():.3f}], '
          f'B_fno=[{B_fnc.min():.3f},{B_fnc.max():.3f}], r={r_val:.4f}')

    np.savez('validation_independent_500.npz',
             dems=dems, B_fem=B_fem_grids, B_fno=B_fno_grids,
             per_tile_r2=per_tile_r2, r2_total=r2_total,
             reliefs=reliefs, B_fem_center=B_fc, B_fno_center=B_fnc)
    print('\n已保存: validation_independent_500.npz')

if __name__ == '__main__':
    main()
