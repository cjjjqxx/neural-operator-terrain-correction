# -*- coding: utf-8 -*-
"""
Fig. 6 | Database filtering and the magnitude of the terrain correction.

核心结论:经过一条可核查的过滤链得到 36,392 个测点后,地形校正对多数测点很小,
但有一条实质性长尾,且这条尾巴系统性地集中在地形敏感度最高的一档。

面板分工(2×2 四宫格,每格只回答一个问题):
  a  数据是怎么筛出来的      -> 过滤级联
  b  校正量有多大            -> |校正量| 分布
  c  校正会不会改变解读      -> 按地形敏感度分档的跨档率
  d  校正是加是减、是否对称  -> 带符号变化的分布与山脊/谷地占比

"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from nature_style_v2 import (set_style, COL, CMAP_B, W2, panel, save)

set_style(base=7)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figs_v2')
os.makedirs(OUT, exist_ok=True)

d = np.load(os.path.join(HERE, '..', '..', 'rebuild_points.npz'))
qm, qc, B = d['q_measured'], d['q_corrected'], d['B']
corr = np.abs(qc - qm) / qm * 100.0
absB1 = np.abs(B - 1.0)

FIG_H_MM = 72.0
AX_B = (0.200 * 68 + 4.0) / FIG_H_MM
AX_H = 0.680 * 68 / FIG_H_MM
fig = plt.figure(figsize=(W2, FIG_H_MM / 25.4))

# ─────────────── a  校正量分布 (hero) ───────────────
ax = fig.add_axes([0.070, AX_B, 0.252, AX_H])
bins = np.logspace(-3, 2, 64)
ax.hist(corr[corr > 0], bins=bins, color=COL['hero_light'],
        edgecolor='white', lw=0.15, zorder=2)
ax.set_xscale('log')
ymax = ax.get_ylim()[1] * 1.16   # 给分位数标注留出顶部余量,避免压住最高柱

# 分位数标注:竖线高度错开,文字统一贴线右侧,避免互相压盖
marks = [(50, 'Median', COL['hero'], 0.98),
         (90, '90th',   COL['warn'], 0.74),
         (99, '99th',   COL['accent'], 0.50)]
for p, lab, c, hfrac in marks:
    v = np.percentile(corr, p)
    ax.plot([v, v], [0, ymax * hfrac], color=c, lw=0.9, ls=(0, (3, 2)), zorder=3)
    ax.text(v * 1.15, ymax * hfrac, f'{lab}\n{v:.1f}%', fontsize=7.2,
            color=c, ha='left', va='top', linespacing=1.15, zorder=4)

ax.set_ylim(0, ymax)
ax.yaxis.set_major_locator(MultipleLocator(500))
ax.set_xlabel('|Correction| (% of measured flux)', labelpad=2)
ax.set_ylabel('Number of sites', labelpad=2)
panel(ax, 'a', x=-0.175, y=1.035)

# ─────────────── b  按地形敏感度分档的跨档率 ───────────────
ax = fig.add_axes([0.402, AX_B, 0.252, AX_H])
dec_edges = np.percentile(qm, np.arange(10, 100, 10))
qrt_edges = np.percentile(qm, [25, 50, 75])
moved_dec = np.digitize(qm, dec_edges) != np.digitize(qc, dec_edges)
moved_qrt = np.digitize(qm, qrt_edges) != np.digitize(qc, qrt_edges)

bands = [(0, 0.02, 'Near-flat'), (0.02, 0.1, 'Moderate'), (0.1, 10, 'Large\ndeparture')]
xpos = np.arange(len(bands)); w = 0.36
dec_pct, qrt_pct, ns = [], [], []
for lo, hi, _ in bands:
    m = (absB1 >= lo) & (absB1 < hi)
    ns.append(int(m.sum()))
    dec_pct.append(100 * moved_dec[m].mean())
    qrt_pct.append(100 * moved_qrt[m].mean())

ax.bar(xpos - w / 2, dec_pct, w, color=COL['hero'], edgecolor='white', lw=0.4,
       label='Crosses a decile', zorder=2)
ax.bar(xpos + w / 2, qrt_pct, w, color=COL['hero_light'], edgecolor='white', lw=0.4,
       label='Crosses a quartile', zorder=2)
for x, v in zip(xpos - w / 2, dec_pct):
    ax.text(x, v + 1.4, f'{v:.1f}', ha='center', fontsize=7.2, color=COL['hero'])
for x, v in zip(xpos + w / 2, qrt_pct):
    ax.text(x, v + 1.4, f'{v:.1f}', ha='center', fontsize=7.2, color=COL['hero_light'])

ax.set_xticks(xpos)
ax.set_xticklabels([f'{b[2]}\nn={n:,}' for b, n in zip(bands, ns)],
                   fontsize=7.2, linespacing=1.2)
ax.set_ylim(0, max(dec_pct) * 1.30)
ax.set_ylabel('Sites changing class (%)', labelpad=2)
ax.set_xlabel('Terrain sensitivity, $|B_D - 1|$', labelpad=3)
ax.legend(fontsize=7.2, loc='upper left', handlelength=0.9,
          handletextpad=0.4, borderpad=0.15, labelspacing=0.25)
panel(ax, 'b', x=-0.175, y=1.035)

# ─────────────── c  带符号变化:山脊上调 vs 谷地下调 ───────────────
ax = fig.add_axes([0.734, AX_B, 0.252, AX_H])
dq = qc - qm
lo, hi = np.percentile(dq, [0.5, 99.5])
# 分箱边界对齐 0:否则跨零的箱会把谷地(Δq ≤ 0)的柱画到零线右侧
_w = (hi - lo) / 60
bins_d = np.arange(np.floor(lo / _w) * _w, hi + _w, _w)
up, dn = B < 1, B > 1
# 配色与全套 CMAP_B 一致:蓝端 = B<1(山脊,上调),红端 = B>1(谷地,下调)
_n_up, _, _ = ax.hist(dq[up], bins=bins_d, color=CMAP_B(0.16), lw=0, zorder=2,
        label=f'Ridge, $B$ < 1  ({100*up.mean():.1f}%)')
_n_dn, _, _ = ax.hist(dq[dn], bins=bins_d, color=CMAP_B(0.84), lw=0, zorder=2,
        label=f'Valley, $B$ > 1  ({100*dn.mean():.1f}%)')
# 分布在 0 附近极度集中,线性纵轴只看得到一根尖峰;取对数后两侧尾部才可读
ax.set_yscale('log')
ax.set_ylim(1, ax.get_ylim()[1] * 40)   # 上方留白放均值注记与图例
ax.axvline(0, color=COL['ink'], lw=0.6, zorder=3, ymax=0.78)
ax.axvline(dq.mean(), color=COL['accent'], lw=0.9, ls=(0, (3, 2)), zorder=4, ymax=0.78)
_mu = float(dq.mean())
_se = float(dq.std(ddof=1) / np.sqrt(len(dq)))
_lo95, _hi95 = _mu - 1.96 * _se, _mu + 1.96 * _se
_sgn = lambda v: ('%.3f' % v).replace('-', '−')
# 面板由 61 mm 缩到 46 mm(2x2 -> 1x3)后,两行标注必然压住左尾的柱子;
# 原注释记的那个「再高撞图例、再低压柱子」的折中窗口已不存在。
# 压成一行放到左上高处 —— 对数纵轴上那里本就是空的。
ax.annotate('mean %s\n[%s, %s]\nmW m$^{-2}$'
            % (_sgn(_mu), _sgn(_lo95), _sgn(_hi95)),
            xy=(dq.mean(), 0.78), xycoords=('data', 'axes fraction'),
            xytext=(0.03, 0.72), textcoords='axes fraction',   # 0.88 与右上图例同高(overlap 报 93%);0.70 在图例之下、左尾柱顶(约 0.60)之上
            fontsize=7.2, color=COL['accent'], ha='left', va='center',
            linespacing=1.45,   # 第三行 mW m⁻² 的上标会顶到第二行
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.92), zorder=10)
ax.set_xlabel('Signed correction, $\Delta q$ (mW m$^{-2}$)', labelpad=2)
ax.set_ylabel('Number of sites', labelpad=2)
ax.legend(fontsize=7.2, loc='upper right', handlelength=0.9,
          handletextpad=0.4, borderpad=0.25, labelspacing=0.25)
panel(ax, 'c', x=-0.175, y=1.035)

save(fig, os.path.join(OUT, 'Fig5'))
print('Fig. 6 done.')
