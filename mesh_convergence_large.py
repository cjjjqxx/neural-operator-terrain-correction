# -*- coding: utf-8 -*-
"""网格收敛的扩大试验:每档 30 个测点,检验 mesh_resolution_test(每档 5–10 个)的结论是否偶然。

口径与 mesh_resolution_test.py 相同:窗口固定 57 km,垂向 n_z = 10,域深 max(3000, 1.5·起伏),
只变水平单元数。另加一种网格:n_x = 40 但求解前先把 128×128 高程按单元尺度做块平均(抗混叠),
检验个别测点加密后大幅跳变是否来自点采样混叠。
抽样:按 8 朝向平均算子的 |B_s − 1| 分五档,每档随机 30 个(种子固定);结果按生产配置有限元的 |B − 1| 报告。
每个解记录中心像元 B 与墙钟耗时。
用法:python mesh_convergence_large.py --probe   (单测点试解各网格,看耗时与内存)
      python mesh_convergence_large.py --procs N
输出 mesh_convergence_large.json。
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import uniform_filter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEM_PATH = os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif'))
VARIANTS = [('40', 40, False), ('40aa', 40, True), ('64', 64, False), ('96', 96, False), ('128', 128, False)]
BANDS = [(0.0, 0.02, '<0.02'), (0.02, 0.05, '0.02-0.05'), (0.05, 0.10, '0.05-0.10'),
         (0.10, 0.20, '0.10-0.20'), (0.20, 9.0, '>0.20')]
N_PER, SEED = 30, 20260930


def solve(args):
    sid, lon, lat, tag, nx, aa = args
    import rasterio
    from fem3d import Fem3D
    from tile_reader import read_tile, CELL_M
    with rasterio.open(DEM_PATH) as ds:
        dem = read_tile(ds, lon, lat)
    if dem is None:
        return sid, tag, np.nan, np.nan
    dem = dem.astype(float)
    relief = float(dem.max() - dem.min())
    if aa:                                         # 块平均到单元尺度(128/40 ≈ 3 像元)
        dem = uniform_filter(dem, size=int(round(dem.shape[0] / nx)), mode='nearest')
    t0 = time.time()
    f = Fem3D(nx=nx, ny=nx, nz=10)
    f.build(dem, CELL_M, depth=max(3000.0, 1.5 * relief))
    f.solve_heat()
    f.compute_B()
    return sid, tag, float(f.B_grid[nx // 2, nx // 2]), time.time() - t0


def boot_ci(x, rng, B=2000):
    x = np.asarray(x); m = [np.median(rng.choice(x, len(x))) for _ in range(B)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--probe', action='store_true')
    a = ap.parse_args()
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    lon, lat, Bs = d['lon'], d['lat'], d['B_surface']
    if a.probe:
        i = int(np.argmin(np.abs(np.abs(Bs - 1) - 0.15)))
        for tag, nx, aa in VARIANTS:
            _, _, b, t = solve((i, float(lon[i]), float(lat[i]), tag, nx, aa))
            print('  %-5s B = %.4f  用时 %.1f s' % (tag, b, t), flush=True)
        return
    rng = np.random.default_rng(SEED)
    ab = np.abs(Bs - 1)
    sites = []
    for lo, hi, lab in BANDS:
        cand = np.where((ab >= lo) & (ab < hi))[0]
        sites += [(int(i), lab) for i in rng.choice(cand, N_PER, replace=False)]
    tasks = [(i, float(lon[i]), float(lat[i]), tag, nx, aa) for i, _ in sites for tag, nx, aa in VARIANTS]
    tasks.sort(key=lambda t: -t[4])                  # 先跑最大的网格,内存峰值集中在前段、便于观察
    res = {}
    t0 = time.time()
    with Pool(a.procs) as p:
        for k, (sid, tag, b, t) in enumerate(p.imap_unordered(solve, tasks), 1):
            res.setdefault(sid, {})[tag] = (b, t)
            if k % 50 == 0:
                print('  %d / %d  %.1f min' % (k, len(tasks), (time.time() - t0) / 60), flush=True)
    out = dict(n_per_band=N_PER, seed=SEED, sites=[], bands={}, timing_s={})
    for i, lab in sites:
        r = res.get(i, {})
        out['sites'].append(dict(idx=i, band_sampled=lab, lon=float(lon[i]), lat=float(lat[i]),
                                 B={k: v[0] for k, v in r.items()}, t={k: v[1] for k, v in r.items()}))
    for tag, _, _ in VARIANTS:
        ts = [s['t'][tag] for s in out['sites'] if tag in s['t'] and np.isfinite(s['t'][tag])]
        out['timing_s'][tag] = dict(median=float(np.median(ts)), max=float(np.max(ts)))
    # 按生产配置有限元 |B−1| 分档报告
    ok = [s for s in out['sites'] if all(np.isfinite(s['B'].get(t, np.nan)) for t, _, _ in VARIANTS)]
    a40 = np.array([abs(s['B']['40'] - 1) for s in ok])
    for lo, hi, lab in BANDS:
        grp = [s for s, v in zip(ok, a40) if lo <= v < hi]
        if len(grp) < 5:
            continue
        g = lambda t: np.array([abs(s['B'][t] - 1) for s in grp])
        a_40, a_aa, a_96, a_128 = g('40'), g('40aa'), g('96'), g('128')
        den = np.median(a_40)
        rel = (a_128 - a_40) / den                     # 与误差模型同一口径:变化量 ÷ 档内中位
        row = dict(n=len(grp),
                   bias_40_to_128_pct=100 * float(np.median(rel)), bias_ci=[100 * v for v in boot_ci(rel, rng)],
                   scatter_40_to_128_pct=100 * float(1.4826 * np.median(np.abs(rel - np.median(rel)))),
                   step_96_to_128_pct=100 * float(np.median((a_128 - a_96) / np.median(a_96))),
                   bias_40aa_to_128_pct=100 * float(np.median((a_128 - a_aa) / np.median(a_aa))),
                   scatter_40aa_to_128_pct=100 * float(1.4826 * np.median(np.abs(
                       (a_128 - a_aa) / np.median(a_aa) - np.median((a_128 - a_aa) / np.median(a_aa))))),
                   sign_flip_40=int(np.sum(np.sign(np.array([s['B']['40'] - 1 for s in grp])) !=
                                         np.sign(np.array([s['B']['128'] - 1 for s in grp])))),
                   sign_flip_40aa=int(np.sum(np.sign(np.array([s['B']['40aa'] - 1 for s in grp])) !=
                                           np.sign(np.array([s['B']['128'] - 1 for s in grp])))))
        out['bands'][lab] = row
        print('%-10s n=%2d  40→128 偏差 %+.1f%% [%+.1f, %+.1f] 离散 %.1f%% | 96→128 %+.1f%% | '
              '抗混叠40→128 %+.1f%% 离散 %.1f%% | 变号 %d→%d'
              % (lab, row['n'], row['bias_40_to_128_pct'], *row['bias_ci'], row['scatter_40_to_128_pct'],
                 row['step_96_to_128_pct'], row['bias_40aa_to_128_pct'], row['scatter_40aa_to_128_pct'],
                 row['sign_flip_40'], row['sign_flip_40aa']), flush=True)
    print('耗时中位(秒):' + '  '.join('%s %.1f' % (k, v['median']) for k, v in out['timing_s'].items()))
    json.dump(out, open(os.path.join(HERE, 'mesh_convergence_large.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('已保存 mesh_convergence_large.json,用时 %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
