# -*- coding: utf-8 -*-
"""垂向网格(nz)对"地表 B"的影响 —— 检验深度平均是否重复扣减。


与此同时,f(D) 是用 nz=40 的加密垂向网格刻画的,其归一化基准 B_surf 对应的深度
区间要浅得多。若如此,B_production x f(D) 就把同一段浅部深度扣了两次:
按 f 曲线插值 f(300 m) ~ 0.91,再乘 f(360 m) = 0.88,有效保留比例约 0.80 而非 0.88。
深度平均整体对中位校正的影响是 0.615% -> 0.511%(因子 0.83),与这一差距同量级。

检验
----
固定 nx=40(生产水平分辨率),只变 nz。若 nz=10 的 |B-1| 系统性小于 nz=80,
则证实"地表 B"已含区间平均;由收敛值可反推 nz=10 实际对应的等效深度。
"""
import sys, os, json, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
NX = 40
NZ_LADDER = [10, 20, 40, 80]
BANDS = [(0.05, 0.10, 7), (0.10, 0.20, 7), (0.20, 9.0, 6)]   # 只取有意义的档位


def one(args):
    lon, lat, dem_path = args
    try:
        import rasterio
        from fem3d import Fem3D
        from mesh_resolution_test import load_tile
        from tile_reader import CELL_M
        with rasterio.open(dem_path) as ds:
            dem = load_tile(ds, lon, lat)
        if dem is None:
            return None
        relief = float(dem.max() - dem.min())
        depth = max(3000, relief * 1.5)
        out = {'lon': lon, 'lat': lat, 'relief': relief, 'depth': depth, 'B': {}, 'dz_top': {}}
        for nz in NZ_LADDER:
            f = Fem3D(nx=NX, ny=NX, nz=nz)
            f.build(dem, CELL_M, depth=depth)
            f.solve_heat(); f.compute_B()
            G = f.B_grid
            out['B'][nz] = float(G[G.shape[0] // 2, G.shape[1] // 2])
            # 中心列顶层单元的厚度 = 有限差分实际跨越的深度
            ix = iy = NX // 2
            node = lambda iz: iz * (NX + 1) * (NX + 1) + iy * (NX + 1) + ix
            out['dz_top'][nz] = float(f.pts[2, node(nz)] - f.pts[2, node(nz - 1)])
        return out
    except Exception as e:
        return {'error': repr(e)[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dem', required=True)
    ap.add_argument('--procs', type=int, default=6)
    a = ap.parse_args()

    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    lon, lat, B0 = d['lon'], d['lat'], d['B_surface']
    ab = np.abs(B0 - 1.0)
    rng = np.random.default_rng(20260815)
    jobs, meta = [], []
    for lo, hi, k in BANDS:
        idx = np.where((ab >= lo) & (ab < hi))[0]
        for i in rng.choice(idx, min(k, len(idx)), replace=False):
            jobs.append((float(lon[i]), float(lat[i]), a.dem))
            meta.append(float(ab[i]))
    print(f'测点 {len(jobs)} 个(|B-1| >= 0.05),nz 阶梯 {NZ_LADDER},nx 固定 {NX}', flush=True)

    with Pool(a.procs) as p:
        res = p.map(one, jobs)
    rows = [r for r in res if r and 'B' in r]
    print(f'成功 {len(rows)} / {len(jobs)}\n')

    print(f'{"nz":>5}{"顶层单元厚(中位)":>18}{"|B-1| 中位":>13}{"相对 nz=80":>13}')
    ref = np.median([abs(r['B'][NZ_LADDER[-1]] - 1) for r in rows])
    summ = {}
    for nz in NZ_LADDER:
        v = np.median([abs(r['B'][nz] - 1) for r in rows])
        dz = np.median([r['dz_top'][nz] for r in rows])
        summ[str(nz)] = dict(median_absB1=float(v), median_dz_top=float(dz),
                             rel_to_finest=float(100 * (v - ref) / ref))
        print(f'{nz:>5}{dz:>17.0f}m{v:>13.4f}{100*(v-ref)/ref:>+12.1f}%')

    # 用 f(D) 曲线反推:nz=10 的"地表 B"相当于多深的区间平均
    _F_D = np.array([0., 100., 146., 250., 390., 500., 1000., 1450., 2000.])
    _F_V = np.array([1.000, 0.987, 0.970, 0.929, 0.873, 0.833, 0.677, 0.574, 0.476])
    v10 = summ['10']['median_absB1']
    ratio = v10 / ref
    eq = float(np.interp(ratio, _F_V[::-1], _F_D[::-1])) if ratio < 1 else 0.0
    print(f'\nnz=10 相对最细网格的保留比例 = {ratio:.4f}')
    print(f'按 f(D) 曲线反推,等效于已做了 [0, {eq:.0f} m] 的区间平均')
    print(f'(生产顶层单元厚中位 {summ["10"]["median_dz_top"]:.0f} m 可作对照)')
    if eq > 50:
        print(f'→ 存在重复扣减:再乘 f(360 m)=0.88 会把同一段浅部扣两次')
    else:
        print(f'→ 未见明显重复扣减,深度平均可照原样施加')

    json.dump(dict(nx=NX, nz_ladder=NZ_LADDER, summary=summ,
                   equivalent_depth_m=eq, ratio_nz10_to_finest=ratio,
                   rows=[{'absB1': m, 'relief': r['relief'],
                          'B': {str(k): v for k, v in r['B'].items()}}
                         for m, r in zip(meta, res) if r and 'B' in r]),
              open(os.path.join(HERE, 'vertical_mesh_test.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('\n已保存 vertical_mesh_test.json')


if __name__ == '__main__':
    main()
