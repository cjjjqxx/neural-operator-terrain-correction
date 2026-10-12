# -*- coding: utf-8 -*-
"""
Fig. 8 | 校正的空间格局与地形起伏的对照。


为什么不并进图 7:四行地图共 272 mm,超 Nature 双栏图上限(210 mm)62 mm;
缩到能塞下的尺寸后,安第斯那种窄条带就看不清了。拆成两张各两行,
每张 138 mm,两条信息各自完整。

  a  逐测点的相对校正 Δq/q_obs(不是插值场),对数—线性混合色标
  b  0.5° 网格内的地形起伏(高程 max - min),口径与训练瓦片一致
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
import matplotlib.cm as mcm
from cmocean import cm as _cmo
from cmcrameri import cm as _cmc
import cartopy.crs as ccrs
from nature_style_v2 import (set_style, COL, CMAP_B, W2, RECT_AR,
                             panel_fig, map_ax, cbar, save)

set_style()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
OUT = os.path.join(HERE, '..', 'figs_v2')
os.makedirs(OUT, exist_ok=True)

# ── 数据 ────────────────────────────────────────────────────────────
d = np.load(os.path.join(ROOT, 'kriged_heatflow_05deg_global.npz'))
lons, lats = d['lons'], d['lats']
LON, LAT = np.meshgrid(lons, lats)

import cartopy.feature as cfeature
from rasterio.features import rasterize
from rasterio.transform import from_origin
land = rasterize([(g, 1) for g in cfeature.LAND.geometries()],
                 out_shape=d['krige_qm'].shape,
                 transform=from_origin(-180, 90, 0.5, 0.5),
                 fill=0, all_touched=True).astype(bool)[::-1]
land &= np.isfinite(d['krige_qm'])

# a 用逐测点值,不用克里金场。测点只覆盖 12.2% 的陆地格点(附图 S5),
# 克里金把它们外推到整片陆地 —— 撒哈拉腹地、亚马逊、西伯利亚北部那些平滑
# 圆斑底下一个测点都没有,画出来的是变差函数在无信息区回归全局均值的结果。
# 更要紧的是它会拆掉这张图自己的论证:b 是逐格点算出的真实起伏,
# a 若有一多半是外推,两行之间的"对应"在无观测区就是假的。
pts = np.load(os.path.join(ROOT, 'rebuild_points.npz'))
p_lon, p_lat = pts['lon'], pts['lat']
p_rel = 100.0 * (pts['q_corrected'] - pts['q_measured']) / pts['q_measured']

rf = np.load(os.path.join(ROOT, 'global_relief_05deg.npz'))
assert np.allclose(rf['lons'], lons) and np.allclose(rf['lats'], lats), \
    '起伏网格与克里金网格不一致,不能按位置对齐'
relief = np.where(land, rf['relief'], np.nan)

DQ_NORM = SymLogNorm(linthresh=1.0, linscale=1.0, vmin=-30.0, vmax=30.0, base=10)
# NaN 透明,让底图(海洋白、无观测陆地浅灰)露出来;填成灰会把底图整片盖住
cd = CMAP_B.copy()
cd.set_bad(alpha=0.0)
CMAP_R = LinearSegmentedColormap.from_list(
    'relief', _cmc.grayC_r(np.linspace(0.02, 0.95, 256)), N=256)
cr = CMAP_R.copy()
cr.set_bad(alpha=0.0)
# 起伏跨三个数量级(平原几十米、喜马拉雅七千米),线性色标下平原全糊成一片。
R_TICKS = [0, 100, 250, 500, 1000, 2000, 4000, 7500]
from matplotlib.colors import BoundaryNorm
R_NORM = BoundaryNorm(R_TICKS, len(R_TICKS), extend='neither')
CMAP_R_D = LinearSegmentedColormap.from_list(
    'relief_d', _cmc.grayC_r(np.linspace(0.02, 0.95, len(R_TICKS))), N=len(R_TICKS))
CMAP_R_D.set_bad(alpha=0.0)

FIG_H = 69.0
PROJ = ccrs.PlateCarree(central_longitude=0)


def map_box(x, top, w):
    h = (w * W2) / RECT_AR / (FIG_H / 25.4)
    return [x, top - h, w, h]


fig = plt.figure(figsize=(W2, FIG_H / 25.4))
ax_b = fig.add_axes(map_box(0.090, 0.935, 0.800), projection=PROJ)
map_ax(ax_b, labels=True, graticule=False)
im_b = ax_b.pcolormesh(LON, LAT, relief, transform=ccrs.PlateCarree(),
                       cmap=CMAP_R_D, norm=R_NORM, shading='nearest',
                       rasterized=True, zorder=2)
ax_b.set_title('Elevation range (max − min) within each grid cell',
               fontsize=7.2, pad=3, color=COL['ink'])
cbar(fig, im_b, [0.908, 0.140, 0.013, 0.660], 'Relief (m)',
     ticks=R_TICKS, orientation='vertical')

save(fig, os.path.join(OUT, 'ExtendedDataFig8'))
_r = relief[np.isfinite(relief)]
print('Fig. 8 done.  陆地起伏中位 %.0f m,p95 %.0f m,最大 %.0f m'
      % (np.median(_r), np.percentile(_r, 95), _r.max()))
