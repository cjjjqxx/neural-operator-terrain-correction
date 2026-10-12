# -*- coding: utf-8 -*-
"""随机陆地位置上 B > 1 的期望占比 —— "谷地过采样"结论的正确零假设。


现成的训练瓦片(global_fem_dataset.npz)带有 raw_relief < 25 m 的剔除准则,因此
是"起伏 > 25 m 的随机陆地",与含 79.8% 近平坦测点的应用集口径不可比。本脚本按
完全相同的采样域重采,但不施加任何起伏过滤,给出无偏参照。
"""
import sys, os, json, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
FN = 40      # 有限元水平单元数,与训练集一致
REGIONS = {'Asia': (60, 180, 5, 78), 'Europe': (-10, 60, 35, 82),
           'N_America': (-170, -10, 15, 84), 'S_America': (-80, -35, -55, 10),
           'Africa': (-20, 55, -35, 38), 'Oceania': (95, 180, -50, 5)}



def one(args):
    lon, lat, dem_path = args
    try:
        import rasterio
        from fem3d import Fem3D
        from tile_reader import read_tile, CELL_M
        with rasterio.open(dem_path) as ds:
            dem = read_tile(ds, lon, lat, reject_ocean=True)
        if dem is None:
            return None
        relief = float(dem.max() - dem.min())
        f = Fem3D(nx=FN, ny=FN, nz=10)
        f.build(dem, CELL_M, depth=max(3000, relief * 1.5))
        f.solve_heat()
        f.compute_B()
        G = f.B_grid                      # compute_B 返回展平数组,二维在 B_grid
        if not np.all(np.isfinite(G)):
            return None
        c0 = G.shape[0] // 2
        return float(G[c0, c0]), relief
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--dem', default=None)
    ap.add_argument('--procs', type=int, default=10)
    a = ap.parse_args()

    dem = a.dem
    if dem is None:
        import glob
        cand = glob.glob(os.path.join(HERE, '..', '**', '*.tif'), recursive=True)
        cand = [c for c in cand if 'srtm' in c.lower() or 'topo' in c.lower() or '15' in c]
        dem = cand[0] if cand else None
    if not dem or not os.path.exists(dem):
        print(f'✗ 未找到 DEM,请用 --dem 指定。候选搜索结果为空。'); sys.exit(1)
    print(f'DEM: {dem}\n目标样本 {a.n},进程 {a.procs}')

    # Natural Earth 陆地多边形:保证候选点确实落在陆地上
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import Point
    from shapely.prepared import prep
    from shapely.ops import unary_union
    LAND = prep(unary_union(list(shpreader.Reader(shpreader.natural_earth(
        resolution='110m', category='physical', name='land')).geometries())))

    rng = np.random.RandomState(20260815)
    names = list(REGIONS)
    got, tried = [], 0
    while len(got) < a.n and tried < a.n * 30:
        batch = []
        while len(batch) < a.procs * 20:
            lomin, lomax, lamin, lamax = REGIONS[names[rng.randint(len(names))]]
            lo, la = rng.uniform(lomin, lomax), rng.uniform(lamin, lamax)
            if LAND.contains(Point(lo, la)):
                batch.append((lo, la, dem))
        tried += len(batch)
        with Pool(a.procs) as p:
            for r in p.map(one, batch):
                if r is not None:
                    got.append(r)
        print(f'  已求解 {len(got):,} / {a.n}  (尝试 {tried:,})', flush=True)

    got = got[:a.n]
    B = np.array([g[0] for g in got]); rel = np.array([g[1] for g in got])
    frac = 100 * (B > 1).mean()
    print(f'\n随机陆地(无起伏过滤) n={len(B):,}')
    print(f'  B>1 占比 = {frac:.2f}%   均值 B = {B.mean():.5f}')
    print(f'  起伏中位 {np.median(rel):.0f} m,其中 <25 m 者占 {100*(rel<25).mean():.1f}%')
    for lo, hi, nm in [(0, 25, '<25 m'), (25, 100, '25-100'), (100, 500, '100-500'), (500, 1e9, '>500')]:
        m = (rel >= lo) & (rel < hi)
        if m.sum() > 5:
            print(f'    {nm:<10} n={m.sum():>5}  B>1 占比 {100*(B[m]>1).mean():.1f}%')
    json.dump(dict(n=len(B), frac_B_gt1=frac, mean_B=float(B.mean()),
                   B=B.tolist(), relief=rel.tolist()),
              open(os.path.join(HERE, 'random_land_baseline.json'), 'w', encoding='utf-8'), indent=2)
    print('\n已保存 random_land_baseline.json')


if __name__ == '__main__':
    main()
