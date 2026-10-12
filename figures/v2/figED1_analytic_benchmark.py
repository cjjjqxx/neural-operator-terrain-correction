# -*- coding: utf-8 -*-
"""附图 S1｜解析基准。


(a) 残差随垂向层数的变化,四个振幅各一条。四条线几乎重合 —— 这正是正文
    "四个振幅下残差几乎不变,说明它是离散误差,而非线性化解本身 O((ak)^2) 的
    截断"那句话的图形形式:若是线性化截断,残差会随振幅系统性张开。
(b) 残差随"计算域深度 / e 折深度"的变化。横轴是覆盖倍数而不是波长本身 ——
    决定误差的是域深够不够盖住扰动,不是波长的绝对值。

数据全部取自既有产物,不重算:
    analytic_benchmark.json             4 个振幅 x 4 种网格配置的 B 与相对误差
    analytic_benchmark_wavelength.json  4 个波长 x 2 个振幅的域深覆盖与残差
"""
import json
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

from nature_style_v2 import set_style, COL, W2, panel_fig, save, FS_TICK, FS_LABEL
from matplotlib.colors import LinearSegmentedColormap

set_style()
OUT = os.path.join(HERE, '..', 'figs_v2')
os.makedirs(OUT, exist_ok=True)


def jload(name):
    with open(os.path.join(ROOT, name), encoding='utf-8') as f:
        return json.load(f)


# 同一色相的四档深浅:重点是四条线重合,颜色只用来区分振幅,不表示类别差异
AMP_RAMP = LinearSegmentedColormap.from_list(
    'amp', ['#9CC3E4', '#5B93C9', '#2C6BAA', '#0A3057'], N=256)

fig = plt.figure(figsize=(W2, 74 / 25.4))
AX_B, AX_H = 0.145, 0.705
LETTER_Y = AX_B + AX_H + 0.105

# ─────────────── a  残差 vs 垂向层数(四个振幅) ───────────────
ax = fig.add_axes([0.082, AX_B, 0.400, AX_H])
ab = jload('analytic_benchmark.json')
cases = ab['cases']
# 只取水平离散固定为 nx=40 的三档,单独把 nx80 拿出来做水平加密的对照
NZ = [10, 40, 80]
KEYS = ['nx40_nz10', 'nx40_nz40', 'nx40_nz80']
for i, c in enumerate(cases):
    y = [abs(c['fem'][k]['rel_err_pct']) for k in KEYS]
    ax.plot(NZ, y, 'o-', color=AMP_RAMP(i / max(len(cases) - 1, 1)),
            ms=3.2, lw=1.1, markerfacecolor='white', markeredgewidth=0.9,
            label='%.0f m' % c['a_m'], zorder=3)
# 水平加密对照:nx 由 40 到 80(在 nz=40 处),四个振幅的点几乎落在一起
y80 = [abs(c['fem']['nx80_nz40']['rel_err_pct']) for c in cases]
ax.plot([40] * len(y80), y80, '+', color=COL['ink'], ms=5.5, mew=1.0, zorder=4)
ax.annotate('$n_x$ = 80\n(same $n_z$)', xy=(40, float(np.mean(y80))),
            xytext=(52, 15.5), fontsize=7.2, color=COL['ink'],
            # shrinkB:引线止于 + 号外约 2 mm,不再与其下的折线相交
            arrowprops=dict(arrowstyle='-', lw=0.5, color=COL['ink'], shrinkB=5))
ax.set_xscale('log')
ax.set_xticks(NZ)
ax.set_xticklabels([str(n) for n in NZ], fontsize=7.2)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.tick_params(axis='x', which='minor', length=0)
ax.set_xlabel('Vertical layers, $n_z$', labelpad=2)
ax.set_ylabel('|Residual| vs. analytical surface value (%)', labelpad=2)
ax.set_ylim(0, 27)
ax.legend(fontsize=7.2, loc='upper right', title='Amplitude $a$',
          title_fontsize=7.2, handlelength=1.1, handletextpad=0.4,
          labelspacing=0.22, borderpad=0.25)
ax.text(0.035, 0.045, 'four amplitudes nearly coincide:\n'
        'the residual is discretisation,\nnot $O((ak)^2)$ truncation',
        transform=ax.transAxes, fontsize=7.2, color=COL['ref'], va='bottom',
        linespacing=1.3)
panel_fig(fig, 'a', 0.082 - 0.135 * 0.352, LETTER_Y)

# ─────────────── b  残差 vs 域深/e折深度 ───────────────
ax = fig.add_axes([0.598, AX_B, 0.352, AX_H])
wl = jload('analytic_benchmark_wavelength.json')
rows = wl['rows']
cov = np.array([r['domain_over_efold'] for r in rows])
err = np.array([abs(r['err_vs_D0_pct']) for r in rows])
lam = np.array([r['lambda_km'] for r in rows])
amp = np.array([r['a_m'] for r in rows])
for a_val, mk in ((100.0, 'o'), (300.0, 's')):
    m = amp == a_val
    if not m.any():
        continue
    ax.plot(cov[m], err[m], mk + '-', color=COL['hero'], ms=3.4, lw=1.1,
            markerfacecolor='white', markeredgewidth=0.9,
            label='$a$ = %.0f m' % a_val, zorder=3)
# 波长标在顶部副坐标轴上(点太密,标在数据区会压线);在设定 xlim 之后同步
_WL = sorted(set(lam), reverse=True)
# 生产配置在真实地形上的中位覆盖倍数(见 depth_decay_spectral.json)
sp = jload('depth_decay_spectral.json')
ax.axvline(sp['domain_over_efold_median'], color=COL['accent'], lw=0.7,
           ls=(0, (3, 2)), zorder=2)
# 标在参考线左侧:线本身已经贴近右边界,文字放右边会被画布切掉
ax.text(sp['domain_over_efold_median'] * 0.95, 0.95,
        'median over the\n40 real profiles', transform=ax.get_xaxis_transform(),
        fontsize=7.2, color=COL['accent'], va='top', ha='right', linespacing=1.3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(cov.min() * 0.82, max(cov.max(), sp['domain_over_efold_median']) * 1.18)
import sys as _sys; _sys.path.insert(0, ROOT)
from error_model import fe_bias as _fe
ax.axhline(100 * _fe(), color=COL['ref'], lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
ax.text(0.60, 100 * _fe() * 1.04, 'error-budget value %.1f%%' % (100 * _fe()),
        fontsize=FS_TICK, color=COL['ref'], va='bottom', ha='left')
ax.set_ylim(4, 60)
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 40]))
ax.yaxis.set_major_formatter(FixedFormatter(['5', '10', '20', '40']))
ax.yaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_locator(FixedLocator([0.7, 1, 2, 3]))
ax.xaxis.set_major_formatter(FixedFormatter(['0.7', '1', '2', '3']))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlabel('Domain depth / e-folding depth', labelpad=2)
_top = ax.twiny()
_top.set_xscale('log')
_top.set_xlim(ax.get_xlim())
_top.set_xticks([float(cov[lam == L].mean()) for L in _WL])
_top.set_xticklabels(['%.1f' % L for L in _WL], fontsize=FS_TICK)
_top.minorticks_off()
_top.set_xlabel('Wavelength (km)', labelpad=2, fontsize=FS_LABEL)
ax.set_ylabel('|Residual| vs. analytical average over $D_0$ (%)', labelpad=2)
ax.legend(fontsize=7.2, loc='lower left', handlelength=1.1,
          handletextpad=0.4, labelspacing=0.22, borderpad=0.25)
panel_fig(fig, 'b', 0.598 - 0.135 * 0.352, LETTER_Y)

save(fig, os.path.join(OUT, 'ExtendedDataFig1'))
print('Extended Data Fig. 1 done.  nz 阶梯 %s,域深覆盖 %.2f-%.2f'
      % (NZ, cov.min(), cov.max()))
