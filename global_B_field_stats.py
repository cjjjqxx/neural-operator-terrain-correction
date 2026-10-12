# -*- coding: utf-8 -*-
"""从全球陆地 B 场派生正文与图 5 共用的统计量。

为什么单独一个脚本
------------------
global_B_field_05deg.py 跑一次要 2.8 小时(绝大部分是从约 7 GB 高程数据里
逐格点取窗口),不适合为了补几个统计量重跑。它落下的 .npz 里已经有 B 与起伏,
派生量在这里算,落成独立产物供 emit_numbers 读走。

"""
import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))

g = np.load(os.path.join(HERE, 'global_B_field_05deg.npz'))
B, R = g['B'], g['relief']
ok = np.isfinite(B)
dev = np.abs(B - 1.0)
big = ok & (dev > 0.10)

with open(os.path.join(HERE, 'fem_speed_benchmark.json'), encoding='utf-8') as fh:
    FEM_MS = json.load(fh)['fem_total_ms']
with open(os.path.join(HERE, 'fno_speed_benchmark.json'), encoding='utf-8') as fh:
    FNO_MS = json.load(fh)['fno_batch_ms']

N_SITES = 36392                     # 交付汇编的陆地测点数(见 rebuild_points.npz)
DEM_PATH = os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif'))
N_05 = int(ok.sum())
N_01 = N_05 * 25                    # 同一陆地掩膜细分到 0.1°:每格分成 5x5


def _grid_variability():
    """0.5° 格子内部的 B 变异 —— 数据产品能不能当查表用,取决于这个。

    发布一张 0.5° 的 B 场,读者自然会想拿它给自己的测点查值。这样做对不对,
    要看同一个格子内部 B 变化多大:若格内变异与校正量本身同量级,查表得到的
    就不是这个测点的校正,而是这一格的平均地形。

    用交付汇编中同格多测点的实测 B 直接量:这是"格内不同位置"的真实样本,
    不是对场再做一次插值。注意测点偏向可达之处,山区格子的真实变异只会更大,
    因此这里给出的是下界。
    """
    import collections
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    key = (np.floor((d['lat'] + 90) / 0.5).astype(int) * 720
           + np.floor((d['lon'] + 180) / 0.5).astype(int))
    g = collections.defaultdict(list)
    for k, b in zip(key, d['B_surface']):
        g[k].append(b)
    rng = np.array([max(v) - min(v) for v in g.values() if len(v) >= 2])
    return dict(
        # 与格内极差同口径的对照量:两者都用地表 B,不能一个用 B_s 一个用深度平均后的 B
        cellvar_ref_absdev=round(float(np.median(np.abs(d['B_surface'] - 1.0))), 4),
        cellvar_n_cells=int(len(rng)),
        cellvar_median=round(float(np.median(rng)), 4),
        cellvar_p90=round(float(np.percentile(rng, 90)), 4),
        cellvar_frac_gt2pct=round(100 * float((rng > 0.02).mean()), 1),
        cellvar_frac_gt10pct=round(100 * float((rng > 0.10).mean()), 1),
    )


def _grid_lookup():
    """格点场直接当查表用的误差(2026-10-05,复查指出"格内极差 vs 单点偏离"是两类量,
    两个以上的值取极差本就接近单点偏离的两倍,不能直接说明格心值没有代表性)。

    这里直接检验:以测点所在 0.5° 格的格心值代替测点自身的 B_s,误差多大、正负号是否一致。
    同时给出含多测点的格子中同时有 B_s > 1 与 < 1 测点的比例(图 4b 所示现象的全库占比)。
    """
    import collections
    from scipy.stats import spearmanr
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    lats = g['lats']
    assert lats[0] < lats[-1]                       # 存档按南→北,行号 i 对应格心 −89.75 + 0.5 i
    i = np.clip(np.floor((d['lat'] + 90) / 0.5).astype(int), 0, B.shape[0] - 1)
    j = np.clip(np.floor((d['lon'] + 180) / 0.5).astype(int), 0, B.shape[1] - 1)
    bg = B[i, j]
    m = np.isfinite(bg)
    bs, bg = d['B_surface'][m], bg[m]
    cells = collections.defaultdict(list)
    for k, b in zip(i * 720 + j, d['B_surface']):
        cells[k].append(b)
    multi = [v for v in cells.values() if len(v) >= 2]
    mixed = np.mean([(max(v) > 1) and (min(v) < 1) for v in multi])
    return dict(
        lookup_n=int(m.sum()),
        lookup_err_median=round(float(np.median(np.abs(bs - bg))), 4),
        lookup_sign_agree_pct=round(100 * float(np.mean((bs > 1) == (bg > 1))), 1),
        lookup_absdev_rho=round(float(spearmanr(np.abs(bs - 1), np.abs(bg - 1))[0]), 2),
        cell_mixed_sign_pct=round(100 * float(mixed), 1),
    )


res = dict(
    n_cells_05=N_05,
    n_cells_01=N_01,
    # 强校正格点几乎全在高起伏地带 —— 正文用它说明校正没有散布到平坦区
    n_dev_gt10pct=int(big.sum()),
    dev_gt10_relief_gt500_pct=round(100 * float((R[big] > 500).mean()), 1),
    dev_gt10_relief_median_m=int(round(float(np.median(R[big])))),
    land_relief_median_m=int(round(float(np.median(R[ok])))),
    B_area_mean=round(float(np.average(B[ok], weights=np.broadcast_to(
        np.cos(np.radians(g['lats']))[:, None], B.shape)[ok])), 4),
    # 图 5b 的六个数:纯计算,不含高程窗口读取(那部分对两条路线是共同的)
    fem_h_sites=round(N_SITES * FEM_MS / 1000 / 3600, 1),
    fno_min_sites=round(N_SITES * FNO_MS / 1000 / 60, 1),
    fem_h_05=round(N_05 * FEM_MS / 1000 / 3600, 1),
    fno_min_05=round(N_05 * FNO_MS / 1000 / 60, 1),
    fem_d_01=round(N_01 * FEM_MS / 1000 / 86400, 1),
    fno_h_01=round(N_01 * FNO_MS / 1000 / 3600, 1),
    # 图注要说明 2.8 小时墙钟里绝大部分是读高程,体积本身也得有产物
    dem_gb=round(os.path.getsize(DEM_PATH) / 1024 ** 3, 1),
    **_grid_variability(),
    **_grid_lookup(),
)
json.dump(res, open(os.path.join(HERE, 'global_B_field_stats.json'), 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1)

print('强校正格点(|B−1| > 10%%)共 %s 个' % format(res['n_dev_gt10pct'], ','))
print('  其中起伏 > 500 m 的占 %.1f%%,起伏中位 %s m(全部陆地格点中位 %s m)'
      % (res['dev_gt10_relief_gt500_pct'],
         format(res['dev_gt10_relief_median_m'], ','),
         format(res['land_relief_median_m'], ',')))
print()
print('纯计算耗时(不含高程读取)')
print('  %s 测点        FEM %5.1f h   算子 %5.1f min'
      % (format(N_SITES, ','), res['fem_h_sites'], res['fno_min_sites']))
print('  %s 格(0.5°)    FEM %5.1f h   算子 %5.1f min'
      % (format(N_05, ','), res['fem_h_05'], res['fno_min_05']))
print('  %s 格(0.1°)  FEM %5.1f d   算子 %5.1f h'
      % (format(N_01, ','), res['fem_d_01'], res['fno_h_01']))
print()
print('0.5° 格内 B 变异(%s 个含多测点的格子)' % format(res['cellvar_n_cells'], ','))
print('  极差中位 %.4f  P90 %.4f;超过 0.02 的占 %.1f%%,超过 0.10 的占 %.1f%%'
      % (res['cellvar_median'], res['cellvar_p90'],
         res['cellvar_frac_gt2pct'], res['cellvar_frac_gt10pct']))
print('已保存 global_B_field_stats.json')
