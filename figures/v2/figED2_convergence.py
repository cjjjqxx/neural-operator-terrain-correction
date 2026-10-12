# -*- coding: utf-8 -*-
"""Extended Data Fig. 1 | Numerical convergence and scale sensitivity.


本版四个面板全部从 json 读取,与正文 2.1 所述的三组解耦试验一一对应:
  a  网格(短波端截断)   <- mesh_convergence_large.json(150 测点,按生产配置有限元 |B−1| 分档)
  b  窗口(长波端截断)   <- footprint_narrow_test.json + footprint_fixed_resolution_test.json
  c  垂向层数(深度截断) <- vertical_mesh_test.json
  d  传递因子随深度衰减 <- depth_decay_profiles.json
数据一改,图随之改;依赖关系也因此对审计可见。
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter
from nature_style_v2 import set_style, COL, W2, panel, save

set_style(base=7)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figs_v2')
ROOT = os.path.join(HERE, '..', '..')
os.makedirs(OUT, exist_ok=True)

RAMP = LinearSegmentedColormap.from_list(
    'relief', ['#BBD5EA', '#7FB0DC', '#4A85C0', '#1A5FA8', '#0D3B6B'], N=5)


def jload(name):
    with open(os.path.join(ROOT, name), encoding='utf-8') as fh:
        return json.load(fh)


def band_label(b):
    """json 的档位键形如 '0.05-0.1';最高档上界是哨兵值 9.0,写成开区间更好懂。"""
    lo, hi = b.split('-')
    return '> %.2f' % float(lo) if float(hi) > 1 else '%.2f–%.2f' % (float(lo), float(hi))


def adopted(ax, x, yfrac, va):
    """'adopted' 标在采用值竖线右侧 3 pt 处。原先用 axes 坐标放置,a/b/c 三处都被红色虚线穿过。"""
    ax.annotate('adopted', xy=(x, yfrac), xycoords=('data', 'axes fraction'),
                xytext=(3, 0), textcoords='offset points', ha='left', va=va,
                fontsize=7.2, color=COL['accent'])


BANDS = ['0.05-0.1', '0.1-0.2', '0.2-9.0']
BCOL = {b: RAMP((i + 2) / 4.0) for i, b in enumerate(BANDS)}

fig = plt.figure(figsize=(W2, 78 / 25.4))

# ─────────────── a  网格分辨率(短波端截断) ───────────────
ax = fig.add_axes([0.080, 0.300, 0.240, 0.600])
mesh = jload('mesh_convergence_large.json')
nxs = mesh['nx_ladder']
for b in BANDS:
    if b not in mesh['summary']:
        continue
    s = mesh['summary'][b]
    m = [s['median_absB1'][str(n)] for n in nxs]
    ax.plot(nxs, [100 * (v - m[0]) / m[0] for v in m], 'o-', color=BCOL[b],
            ms=3.0, lw=1.1, label=band_label(b),
            markerfacecolor='white', markeredgewidth=0.9, zorder=3)
ax.axhline(0, color=COL['ref'], lw=0.5, ls=':', zorder=1)
ax.axvline(nxs[0], color=COL['accent'], lw=0.7, ls=(0, (3, 2)), zorder=2)
adopted(ax, nxs[0], 0.97, 'top')
ax.set_xlabel('Elements per tile edge, $n_x$\n(element 1,425 to 445 m)',
              labelpad=2, linespacing=1.25)
ax.set_ylabel('Change in median $|B - 1|$\nfrom adopted $n_x$ = 40 (%)',
              labelpad=2, linespacing=1.25)
ax.set_xticks(nxs)
_lg_handles, _lg_labels = ax.get_legend_handles_labels()
panel(ax, 'a', x=-0.230 * 0.250 / 0.240 - 0.010 / 0.240, y=1.045)

# ─────────────── b  瓦片窗口(长波端截断) ───────────────
ax = fig.add_axes([0.403, 0.300, 0.250, 0.600])
nb = jload('footprint_narrow_test.json')
wb = jload('footprint_fixed_resolution_test.json')
WN, WW = [20, 28, 40, 57], [80, 114, 160]
for b in BANDS:
    if b not in nb['summary'] or b not in wb['summary']:
        continue
    mn, mw = nb['summary'][b]['median_absB1'], wb['summary'][b]['median_absB1']
    # 57 km 是两组共有的锚点:同一批测点、同一配置,数值必须相等,否则拼接无意义
    assert abs(mn['57'] - mw['57']) < 1e-9, (b, mn['57'], mw['57'])
    ws = WN + WW
    m = [mn[str(x)] for x in WN] + [mw[str(x)] for x in WW]
    ax.plot(ws, [100 * (v - m[0]) / m[0] for v in m], 'o-', color=BCOL[b],
            ms=3.0, lw=1.1,
            label='%s (n=%d)' % (band_label(b), nb['summary'][b]['n']),
            markerfacecolor='white', markeredgewidth=0.9, zorder=3)
ax.axhline(0, color=COL['ref'], lw=0.5, ls=':', zorder=1)
ax.axvline(57, color=COL['accent'], lw=0.7, ls=(0, (3, 2)), zorder=2)
adopted(ax, 57, 0.97, 'top')
ax.set_xscale('log')
ax.set_xticks(WN + WW)
ax.set_xticklabels([str(x) for x in WN + WW], fontsize=7.2)
ax.xaxis.set_minor_formatter(NullFormatter())
ax.tick_params(axis='x', which='minor', length=0)
ax.set_xlabel('Tile footprint (km)\n(element size fixed at 1,425 m)', labelpad=2,
              linespacing=1.25)
ax.set_ylabel('Change in median $|B - 1|$\nfrom 20 km (%)', labelpad=2,
              linespacing=1.25)
panel(ax, 'b', x=-0.230, y=1.045)

# ─────────────── c  垂向层数(深度截断) ───────────────
ax = fig.add_axes([0.736, 0.300, 0.250, 0.600])
vm = jload('vertical_mesh_test.json')
nzs = vm['nz_ladder']
med = [vm['summary'][str(n)]['median_absB1'] for n in nzs]
rel = [vm['summary'][str(n)]['rel_to_finest'] for n in nzs]
dz = [vm['summary'][str(n)]['median_dz_top'] for n in nzs]
ax.plot(nzs, med, 'o-', color=COL['hero'], ms=3.6, lw=1.3,
        markerfacecolor='white', markeredgewidth=1.0, zorder=3)
_last = len(nzs) - 1
for i, (n, m, r) in enumerate(zip(nzs, med, rel)):
    ax.annotate(('%+.1f%%' % r).replace('-', '−'), (n, m), textcoords='offset points',
                xytext=(14 if i == 0 else (3 if i == _last else 4), -12),
                ha='right' if i == _last else 'left',
                fontsize=7.2, color=COL['ref'])
ax.axvline(nzs[0], color=COL['accent'], lw=0.7, ls=(0, (3, 2)), zorder=2)
adopted(ax, nzs[0], 0.06, 'bottom')
ax.set_xscale('log')
ax.set_xticks(nzs)
ax.set_xticklabels([str(n) for n in nzs], fontsize=7.2)
ax.xaxis.set_minor_formatter(NullFormatter())
ax.tick_params(axis='x', which='minor', length=0)
ax.set_xlabel('Vertical layers, $n_z$\n(top layer %.0f to %.0f m)'
              % (dz[0], dz[-1]), labelpad=2, linespacing=1.25)
ax.set_ylabel('Median $|B - 1|$', labelpad=2)
ax.set_ylim(min(med) * 0.90, max(med) * 1.06)
ax.text(0.085, 0.95, 'change vs. $n_z$ = %d' % nzs[-1],
        transform=ax.transAxes, fontsize=7.2, color=COL['ref'], va='top')
panel(ax, 'c', x=-0.230, y=1.045)

# a 与 b 是同一套 |B - 1| 分档,横排后两份图例会把窄面板占满,收到底部共享。
fig.legend(_lg_handles, _lg_labels, loc='lower center',
           bbox_to_anchor=(0.5, 0.005), ncol=3, fontsize=7.2,
           title='$|B - 1|$ band', title_fontsize=7.2, frameon=False,
           handlelength=1.1, handletextpad=0.4, columnspacing=1.6)


save(fig, os.path.join(OUT, 'ExtendedDataFig2'))
