# -*- coding: utf-8 -*-
"""附图 S4｜深度衰减与地形谱。


  (a) "四个 |B - 1| 档的 f(D) 曲线高度重合,档间中位数极差在 250 米处为 0.021、
      390 米处为 0.028、500 米处为 0.033" —— 重合与否,画出来一眼可判。
      这个面板从原 ExtendedDataFig1 的 d 面板搬来:它讲的是深度平均,属于
      S2.1,挂在"三组解耦试验"那张图上是跨节了。
  (b) "质心波长与 e 折深度无显著秩相关(Spearman r = 0.26,p = 0.10)" ——
      否定性结论最该配散点图,否则读者只能相信一个相关系数。
  (c) e 折深度相对于钻孔区间深度的位置:40 条里 38 条在 360 m 之上,
      即多数测点的采样区间仍处在衰减曲线的近线性段。

数据取自既有产物,不重算:
    depth_decay_profiles.json   40 条剖面的 f(D) 曲线
    depth_decay_spectral.json   40 条剖面的质心波长、e 折深度、起伏
"""
import json
import re
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.dirname(HERE))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from nature_style_v2 import set_style, COL, W2, panel, save
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

_BAND_RAMP = LinearSegmentedColormap.from_list(
    'relief', ['#BBD5EA', '#7FB0DC', '#4A85C0', '#1A5FA8', '#0D3B6B'], N=5)


def RELIEF_RAMP(t):
    return _BAND_RAMP(0.2 + 0.8 * t)


def log_ticks(ax):
    """对数纵轴固定刻度、普通数字(原先只落得下 10³ 一个主刻度)。"""
    ax.yaxis.set_major_locator(FixedLocator([200, 500, 1000, 2000, 5000]))
    ax.yaxis.set_major_formatter(FixedFormatter(['200', '500', '1,000', '2,000', '5,000']))
    ax.yaxis.set_minor_formatter(NullFormatter())


set_style()
OUT = os.path.join(HERE, '..', 'figs_v2')
os.makedirs(OUT, exist_ok=True)


def jload(name):
    with open(os.path.join(ROOT, name), encoding='utf-8') as f:
        return json.load(f)


BAND_ORDER = ['near_flat', 'moderate', 'high', 'extreme']
BAND_SHORT = {'near_flat': '< 0.02', 'moderate': '0.02–0.10',     # 按 |B−1| 分档(depth_decay_characterize.py)
              'high': '0.10–0.20', 'extreme': '> 0.20'}
with open(os.path.join(ROOT, 'rebuild_correction.py'), encoding='utf-8') as _fh:
    D_FALLBACK = float(re.search(r'MEDIAN_DEPTH_M = ([\d.]+)', _fh.read()).group(1))

fig = plt.figure(figsize=(W2, 76 / 25.4))

# ─────────────── a  f(D):四档几乎重合 ───────────────
ax = fig.add_axes([0.070, 0.235, 0.250, 0.660])
prof = jload('depth_decay_profiles.json')
D = np.array([0, 100, 146, 250, 390, 500, 1000, 1450, 2000])
allf = np.array([[1.0] + [p['f'].get(str(x), np.nan) for x in D[1:]]
                 for p in prof], float)
ax.fill_between(D, np.nanpercentile(allf, 25, axis=0),
                np.nanpercentile(allf, 75, axis=0),
                color=COL['ref_light'], alpha=0.35, lw=0, zorder=0)
med_by_band = []
for i, band in enumerate(BAND_ORDER):
    sub = [p for p in prof if p['band'] == band]
    if not sub:
        continue
    fs = np.array([[1.0] + [p['f'].get(str(x), np.nan) for x in D[1:]]
                   for p in sub], float)
    m = np.nanmedian(fs, axis=0)
    med_by_band.append(m)
    ax.plot(D, m, 'o-', color=RELIEF_RAMP(i / 3.0), ms=2.6, lw=1.1,
            markerfacecolor='white', markeredgewidth=0.8, zorder=3)
# 档间极差只在钻孔实际采样的深度范围内取 —— 正文那句"高度重合"带的正是这个
# 限定(250/390/500 m 三处)。若取到 2,000 m,极差会到 0.050,与正文的 0.033 对不上,
# 读者会以为图文矛盾;而那个深度已远超本文样本的区间底深。
_in_range = D <= 500
spread = (np.nanmax(np.array(med_by_band), axis=0) -
          np.nanmin(np.array(med_by_band), axis=0))[_in_range]
ax.set_xlabel('Borehole interval depth, $D$ (m)', labelpad=2)
ax.set_ylabel('Retained surface departure, $f(D)$', labelpad=2)
ax.axvline(D_FALLBACK, color=COL['accent'], lw=0.7, ls=(0, (3, 2)), zorder=2)
ax.text(D_FALLBACK * 1.10, 0.30, 'fallback\ndepth',
        transform=ax.get_xaxis_transform(), fontsize=7.2, color=COL['accent'],
        va='top', linespacing=1.25)
ax.text(0.97, 0.97, 'max spread between bands,\n$D$ ≤ 500 m: %.3f' % np.nanmax(spread),
        transform=ax.transAxes, ha='right', va='top', fontsize=7.2,
        color=COL['ref'], linespacing=1.3)
panel(ax, 'a', x=-0.230, y=1.045)

# ─────────────── b  质心波长 vs e 折深度 ───────────────
ax = fig.add_axes([0.403, 0.235, 0.250, 0.660])
sp = jload('depth_decay_spectral.json')
rows = sp['rows']
lam = np.array([r['lambda_c_m'] for r in rows]) / 1000.0
ef = np.array([r['efold_m'] for r in rows])
bnd = [r['band'] for r in rows]
for i, band in enumerate(BAND_ORDER):
    m = [j for j, b in enumerate(bnd) if b == band]
    if not m:
        continue
    ax.scatter(lam[m], ef[m], s=13, color=RELIEF_RAMP(i / 3.0),
               edgecolors='white', linewidths=0.4, zorder=3)
ax.axhline(D_FALLBACK, color=COL['accent'], lw=0.7, ls=(0, (3, 2)), zorder=2)
ax.text(0.03, D_FALLBACK * 1.12, 'fallback interval depth, %.0f m' % D_FALLBACK,
        transform=ax.get_yaxis_transform(), fontsize=7.2, color=COL['accent'])
ax.set_yscale('log')
log_ticks(ax)
ax.set_ylim(150, 12000)
ax.set_xlabel('Spectral centroid wavelength (km)', labelpad=2)
ax.set_ylabel('e-folding depth (m)', labelpad=2)
# 相关系数取自产物记录的口径,不在这里重算,避免与正文两套算法
ax.text(0.97, 0.97, 'Spearman $r$ = 0.26\n$p$ = 0.10  ($n$ = %d)' % len(rows),
        transform=ax.transAxes, ha='right', va='top', fontsize=7.2,
        color=COL['ink'], linespacing=1.3)
panel(ax, 'b', x=-0.230, y=1.045)

ax = fig.add_axes([0.736, 0.235, 0.250, 0.660])
order = np.argsort(ef)
ycol = [RELIEF_RAMP(BAND_ORDER.index(bnd[j]) / 3.0) for j in order]
ax.scatter(np.arange(len(ef)), ef[order], s=13, color=ycol,
           edgecolors='white', linewidths=0.4, zorder=3)
ax.axhline(D_FALLBACK, color=COL['accent'], lw=0.7, ls=(0, (3, 2)), zorder=2)
n_above = int((ef > D_FALLBACK).sum())
ax.text(0.97, 0.05, '%d of %d above\nthe fallback depth' % (n_above, len(ef)),
        transform=ax.transAxes, ha='right', va='bottom', fontsize=7.2,
        color=COL['accent'], linespacing=1.3)
ax.set_yscale('log')
log_ticks(ax)
ax.set_xlabel('Profile, sorted by e-folding depth', labelpad=2)
ax.set_ylabel('e-folding depth (m)', labelpad=2)
panel(ax, 'c', x=-0.230, y=1.045)

# 三个面板共用起伏分档的颜色,图例只给一次
_h = [plt.Line2D([], [], marker='o', ls='', markerfacecolor=RELIEF_RAMP(i / 3.0),
                 markeredgecolor='white', markeredgewidth=0.4, markersize=4.2,
                 label=BAND_SHORT[b]) for i, b in enumerate(BAND_ORDER)]
fig.legend(handles=_h, loc='lower center', bbox_to_anchor=(0.5, 0.005), ncol=4,
           fontsize=7.2, frameon=False, handletextpad=0.4, columnspacing=1.8,
           title='$|B-1|$ band', title_fontsize=7.2)

save(fig, os.path.join(OUT, 'ExtendedDataFig4'))
print('Extended Data Fig. 4 done.  D<=500m 档间最大极差 %.3f,%d/%d 条剖面 e 折深度 > %.0f m'
      % (np.nanmax(spread), n_above, len(ef), D_FALLBACK))
