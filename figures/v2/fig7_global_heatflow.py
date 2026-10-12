# -*- coding: utf-8 -*-
"""
Fig. 7 | Global terrain-corrected heat flow.

核心结论:用完全相同的克里金流程处理原始与校正后点值,两个网格化场之差因此只能
归因于地形校正本身;该差值场在空间上连贯,集中在已知高起伏造山带与裂谷系统。

"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import cartopy.crs as ccrs
from nature_style_v2 import (set_style, COL, CMAP_Q, CMAP_B, W2, RECT_AR,
                             panel_fig, map_ax, cbar, save)

set_style(base=7)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figs_v2')
ROOT = os.path.join(HERE, '..', '..')
os.makedirs(OUT, exist_ok=True)

_KG = os.path.join(ROOT, 'kriged_heatflow_05deg_global.npz')
if not os.path.exists(_KG):
    raise SystemExit('缺少 kriged_heatflow_05deg_global.npz,先跑 krige_global_ok.py')
d = np.load(_KG)
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
qm = np.where(land, d['krige_qm'], np.nan)
qc = np.where(land, d['krige_qc'], np.nan)
# (c) 用相对变化而非绝对值:绝对值口径下中位 |dq| 仅为色标的 1/43,
# 大部分陆地无从分辨;相对口径下为 1/14,且 |dq/q| > 1% 的格点占 19.3%
# (绝对口径下 > 1 mW m-2 的仅 11.1%)。
dq_abs = np.where(land, d['krige_dq'], np.nan)
dq = 100.0 * dq_abs / np.where(np.isfinite(qm) & (qm > 0), qm, np.nan)

# 色标由 20-150 收到 30-100。实测热流的 p5-p95 是 43.2-81.4 mW m-2,
# 在 20-150 的线性色标上只占 18%-47%,九成格点挤在三分之一色标里,
# a 与 b 因此都是一片相近的橙。收窄后同一区间铺开到 19%-73%,
# 仍覆盖 98.3% 的格点,两端由 extend 箭头标示。
Q_MIN, Q_MAX = 30.0, 100.0
# 相对校正的分布是"中心极度堆积 + 长尾":50% 的格点落在 ±0.5% 以内,
# 只有 5.1% 超过 ±3%,极值到 -29% / +60%。线性色标在这种分布上没有好的
# 取法 —— 实测四种方案下各分位数在色标上的位置(0 到 1):
#     线性 ±3   p5 0.13  p25 0.41  p50 0.49  p75 0.51  p95 0.78  p99 1.19(溢出)
#     线性 ±15  p5 0.43  p25 0.48  p50 0.50  p75 0.50  p95 0.56  p99 0.64
#     SymLog    p5 0.20  p25 0.38  p50 0.48  p75 0.52  p95 0.78  p99 0.86
# 放宽到 ±15 反而最差,九成以上的格点挤成一点白。SymLog 在 |Δq/q| < 1% 内
# 线性展开、之外按对数压缩,小偏离与强异常这才能同时读出来。
DQ_LIM = 30.0
DQ_NORM = SymLogNorm(linthresh=1.0, linscale=1.0,
                     vmin=-DQ_LIM, vmax=DQ_LIM, base=10)
# NaN 必须透明,不能填成灰色。pcolormesh 铺的是覆盖全球的网格,海洋与无观测的
# 陆地都是 NaN;若用 set_bad 填一个灰,这一层会整片盖在底图之上 —— 无论把
# COL['ocean'] 调成什么颜色都看不见,整幅图始终是灰底。透明之后才露出
# 底图本身:海洋白、无观测陆地浅灰,数据区的颜色才浮得出来。
cq = CMAP_Q.copy(); cq.set_bad(alpha=0.0)
cd = CMAP_B.copy(); cd.set_bad(alpha=0.0)
PROJ = ccrs.PlateCarree(central_longitude=0)

# 每个地图框的高宽比必须等于 Robinson 全球投影自身的比例,否则地图会被
# 高度限死、在框内横向居中留白,而按绝对坐标摆放的色标就会落到相邻面板上。
# 裁掉南极后地图变扁,画布跟着收:176 mm 时 a/b 与色标之间、c 与底部之间
# 各空出 21 mm 以上。150 mm 下两张地图占到版面的 68%。
FIG_H = 138.0                       # mm(两行通栏,矩形裁到 57S-87N)
# 不裁纬度:Robinson 的边界是弧形,按矩形裁掉南北之后边框变成
# "两侧弧形 + 上下平直"的多边形,像被削掉了角。南大洋的空白宁可留着。
MAP_AR = RECT_AR                    # 等经纬 58S-84N 的宽/高 = 2.535


def map_box(x, top, w):
    """给定左端、上沿与宽度(均为画布比例),返回符合投影比例的 axes 矩形。"""
    h = (w * W2) / MAP_AR / (FIG_H / 25.4)        # W2 已是英寸
    return [x, top - h, w, h]


fig = plt.figure(figsize=(W2, FIG_H / 25.4))

# 一列三行,同宽同左右边界:读者上下扫一眼就能比同一处经度上的差别。
# a/b 并排时 b 的纬度标签会挤在两图之间、贴着 a 的右边框(那条缝只有 6.4 mm),
# 竖排之后每行左侧都有完整的标签空间。
ax_a = fig.add_axes(map_box(0.090, 0.945, 0.800), projection=PROJ)
ax_b = fig.add_axes(map_box(0.090, 0.470, 0.800), projection=PROJ)

for ax in (ax_a, ax_b):
    map_ax(ax, labels=True, graticule=False)
# 两行的经度完全相同,只在下面一行标
ax_a.set_xticklabels([])

im_a = ax_a.pcolormesh(LON, LAT, qm, transform=ccrs.PlateCarree(), cmap=cq,
                       vmin=Q_MIN, vmax=Q_MAX, shading='nearest',
                       rasterized=True, zorder=2)
ax_b.pcolormesh(LON, LAT, qc, transform=ccrs.PlateCarree(), cmap=cq,
                vmin=Q_MIN, vmax=Q_MAX, shading='nearest',
                rasterized=True, zorder=2)

ax_a.set_title('Measured heat flow', fontsize=7.2, pad=3, color=COL['ink'])
ax_b.set_title('Terrain-corrected heat flow', fontsize=7.2, pad=3, color=COL['ink'])

# 竖排在右侧,跨 a、b 两行的高度居中 —— 位置本身说明它同时管这两张图。
_cba = cbar(fig, im_a, [0.908, 0.330, 0.013, 0.400],
            'Heat flow (mW m$^{-2}$)', ticks=list(range(30, 101, 10)),
            extend='both', orientation='vertical')

panel_fig(fig, 'a', 0.022, 0.962)
panel_fig(fig, 'b', 0.022, 0.487)

save(fig, os.path.join(OUT, 'ExtendedDataFig7'))
print('Fig. 7 done.')
