# -*- coding: utf-8 -*-
"""把本文的地形校正施加到 IHFC 官方克里金场上,做一次独立于本文插值流程的对照。


两个问题用同一份材料就能回答:IHFC 在 GHFDB 2024 发布时同时公开了官方的
0.5° 克里金场(Petrunin et al., 2026, doi:10.5880/GFZ.KHAG.2026.004),
其处理流程与本文一致 —— 0.5° 网格、格内取中位数、普通克里金。

  ① 那个 67–68 是**全球算术平均**;把官方场按 cos(纬度) 面积加权到陆地上,
     官方自己的值是 61.0,与本文的 61.7 相差不到一个单位。六个单位的落差
     是口径之差,不是流程之差。
  ② 把本文的逐格点相对校正施加到官方场上,得到的变化与本文自身场上的变化
     几乎相同 —— 校正效应对底层场的选择不敏感。

口径
----
两套网格的坐标定义不同:官方是网格线(−180, −179.5, …, 180;721×361),
本文是格心(−179.75, −179.25, …;720×360)。故对官方场做双线性插值取到本文格心,
再在本文的陆地掩膜上比较。这样两者用的是同一批格点、同一套权重。
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
OFF = os.path.join(HERE, 'ghfdb2024_official',
                   'IHFC_2024_GHFDB_05x05_WGS1984_Kriging.csv')

K = pd.read_csv(OFF)
kg = K.pivot(index='Lat', columns='Lon', values='Q')
interp = RegularGridInterpolator((kg.index.values, kg.columns.values), kg.values,
                                 bounds_error=False, fill_value=np.nan)

d = np.load(os.path.join(HERE, 'kriged_heatflow_05deg_global.npz'))
gm, gc, glat, glon = d['krige_qm'], d['krige_qc'], d['lats'], d['lons']
LA, LO = np.meshgrid(glat, glon, indexing='ij')
off = interp(np.stack([LA.ravel(), LO.ravel()], axis=-1)).reshape(gm.shape)

land = np.isfinite(gm) & np.isfinite(gc) & np.isfinite(off)
w = np.cos(np.radians(glat))[:, None] * np.ones_like(gm)


def A(x):
    return float(np.average(x[land], weights=w[land]))


# 全球(含海洋)的算术平均 —— 文献里 68 那个数就是这么来的,列出来供正文对照口径
glob_arith = float(K['Q'].mean())
glob_area = float(np.average(K['Q'].values, weights=np.cos(np.radians(K['Lat'].values))))

rel = np.where(gm != 0, (gc - gm) / gm, 0.0)
off_cor = off * (1.0 + rel)

res = dict(
    n_land_cells=int(land.sum()),
    official_global_arithmetic=round(glob_arith, 2),
    official_global_area_weighted=round(glob_area, 2),
    official_land_area_weighted=round(A(off), 2),
    ours_land_area_weighted=round(A(gm), 2),
    official_minus_ours=round(A(gm) - A(off), 2),
    official_corrected=round(A(off_cor), 3),
    official_delta=round(A(off_cor) - A(off), 3),
    official_delta_pct=round(100 * (A(off_cor) - A(off)) / A(off), 3),
    ours_delta=round(A(gc) - A(gm), 3),
    ours_delta_pct=round(100 * (A(gc) - A(gm)) / A(gm), 3),
)
json.dump(res, open(os.path.join(HERE, 'official_grid_benchmark.json'), 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1)

print('共有陆地格点 %d' % res['n_land_cells'])
print()
print('官方场(GHFDB 2024, Petrunin et al. 2026)')
print('  全球算术平均       %.2f   ← 文献常引的 68 是这个口径' % res['official_global_arithmetic'])
print('  全球面积加权       %.2f' % res['official_global_area_weighted'])
print('  陆地面积加权       %.2f' % res['official_land_area_weighted'])
print('本文场')
print('  陆地面积加权       %.2f   (与官方相差 %+.2f)'
      % (res['ours_land_area_weighted'], res['official_minus_ours']))
print()
print('施加本文的地形校正:')
print('  官方场   %.2f → %.3f   %+.4f mW/m² (%+.3f%%)'
      % (res['official_land_area_weighted'], res['official_corrected'],
         res['official_delta'], res['official_delta_pct']))
print('  本文场   %.2f → %.3f   %+.4f mW/m² (%+.3f%%)'
      % (res['ours_land_area_weighted'], A(gc), res['ours_delta'], res['ours_delta_pct']))
print()
print('已保存 official_grid_benchmark.json')
