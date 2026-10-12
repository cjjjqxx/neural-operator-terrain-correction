# -*- coding: utf-8 -*-
"""
Fig. 6 | 校正量随地形起伏增大,且地形效应越强、校正越显著地超出方法误差。



输出 Fig6 与 terrain_control_stats.json。json 的原有字段一个不改;新增:
  relief_bins[i].frac_lt2 / frac_2_5 / frac_5_10 / frac_ge10  —— 单位为百分比(0–100),与 gt5_pct 同单位;
  frac_5_10 + frac_ge10 == gt5_pct,frac_ge10 == gt10_pct。
分档边界与 gt5_pct / gt10_pct 的严格不等号一致:<2 即 ≤2,2–5 即 (2, 5],5–10 即 (5, 10],≥10 即 >10;
实测数据在 2、5、10 处没有恰好相等的值(脚本里断言),所以标签上的 < / ≥ 与计算口径没有差别。
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from nature_style_v2 import set_style, COL, W2, panel_fig, save, FS_TICK

set_style(base=7)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figs_v2')
ROOT = os.path.join(HERE, '..', '..')
os.makedirs(OUT, exist_ok=True)

UP, DOWN = '#2F6DB0', '#C2553A'          # 山脊一侧(B_D<1,上调,蓝)/ 谷地一侧(B_D>1,下调,红);沿用原图 6a 配色

# ════════════════════ 计算:原 fig6_terrain_control.py 的统计部分,原样迁入 ════════════════════
d = np.load(os.path.join(ROOT, 'rebuild_points.npz'))
t = np.load(os.path.join(ROOT, 'site_terrain.npz'))
assert np.allclose(d['lon'], t['lon']) and np.allclose(d['lat'], t['lat'])
dq = 100.0 * (d['q_corrected'] / d['q_measured'] - 1.0)
ab = np.abs(dq)
rel = t['relief_m']
up = d['B'] < 1                           # d['B'] 即 B_D = q_obs / q_cor(逐点相等,已核)

m = (rel > 10) & (ab > 1e-4)
x, y = np.log10(rel[m]), np.log10(ab[m])
q = np.quantile(x, np.linspace(0, 1, 21))
bx, by, b25, b75 = [], [], [], []
for lo, hi in zip(q[:-1], q[1:]):
    k = (x >= lo) & (x <= hi)
    bx.append(np.median(x[k])); by.append(np.median(y[k]))
    b25.append(np.percentile(y[k], 25)); b75.append(np.percentile(y[k], 75))
bx, by, b25, b75 = map(np.array, (bx, by, b25, b75))
fit = stats.linregress(bx, by)
pt = stats.linregress(x, y)
rho = stats.spearmanr(rel, ab).correlation
S = dict(power_exponent_binned=float(fit.slope), r2_binned=float(fit.rvalue ** 2),
         power_exponent_site=float(pt.slope), r2_site=float(pt.rvalue ** 2), spearman=float(rho))
S['power_median_1000m_pct'] = float(10 ** (fit.intercept + 3 * fit.slope))
S['power_fold_per_decade'] = float(10 ** fit.slope)
S['power_relief_range_m'] = [float(10 ** bx.min()), float(10 ** bx.max())]
# 最高起伏档实测中位高于关系式(残差 +0.16 dex);正文写出,免得与紧接的「超过 2,000 米为 6.0%」对不上
S['power_top_bin_median_pct'] = float(10 ** by[-1])
S['power_top_bin_fit_pct'] = float(10 ** (fit.intercept + fit.slope * bx[-1]))
# 脊/谷分开的逐点 R²(回答「分开算会不会更好」;正文只引用其量级)
for nm, kk in (('ridge', up[m]), ('valley', ~up[m])):
    S['r2_site_' + nm] = float(stats.linregress(x[kk], y[kk]).rvalue ** 2)

EDGES = [0, 100, 250, 500, 1000, 2000, 1e9]
LAB = ['<100', '100–\n250', '250–\n500', '500–\n1,000', '1,000–\n2,000', '>2,000']
rows = []
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    k = (rel >= lo) & (rel < hi)
    rows.append(dict(lo=lo, hi=hi, n=int(k.sum()), med=float(np.median(ab[k])),
                     gt5_pct=float(100 * np.mean(ab[k] > 5)), gt10_pct=float(100 * np.mean(ab[k] > 10))))
S['relief_bins'] = rows

for thr in (2, 5, 10):
    assert not np.any(ab == thr), '校正量恰好等于 %d%% 的测点存在,分档口径需重新约定' % thr
CLS = [('frac_lt2', '< 2%'), ('frac_2_5', '2–5%'), ('frac_5_10', '5–10%'), ('frac_ge10', '≥ 10%')]
print('[fig6b] 各起伏档四类占比(%):  <2 | 2–5 | 5–10 | ≥10  ||  5–10+≥10  vs  gt5_pct')
for r in rows:
    k = (rel >= r['lo']) & (rel < r['hi'])
    a_ = ab[k]
    r['frac_lt2'] = float(100 * np.mean(a_ <= 2))
    r['frac_2_5'] = float(100 * np.mean((a_ > 2) & (a_ <= 5)))
    r['frac_5_10'] = float(100 * np.mean((a_ > 5) & (a_ <= 10)))
    r['frac_ge10'] = float(100 * np.mean(a_ > 10))
    s5 = r['frac_5_10'] + r['frac_ge10']
    assert abs(s5 - r['gt5_pct']) < 1e-9, (r['lo'], s5, r['gt5_pct'])
    assert abs(r['frac_ge10'] - r['gt10_pct']) < 1e-9
    assert abs(sum(r[c] for c, _ in CLS) - 100.0) < 1e-9
    hi_s = '∞' if r['hi'] > 1e8 else '%g' % r['hi']
    print('  %5g–%-5s n=%5d  %6.2f | %6.2f | %6.2f | %6.2f  ||  %6.3f  vs  %6.3f'
          % (r['lo'], hi_s, r['n'], r['frac_lt2'], r['frac_2_5'], r['frac_5_10'], r['frac_ge10'],
             s5, r['gt5_pct']))

# ════════════════════ 计算:原 fig_error_budget.py(c、d)════════════════════
z = np.load(os.path.join(ROOT, 'site_reliability.npz'))
qm, qc, sq, snr, grade = z['q_measured'], z['q_corrected'], z['sigma_q'], z['snr'], z['grade']
dqe = np.abs(qc - qm)
bd = np.abs(qm / qc - 1)          # |B_D − 1|,与表 1 同一分档变量
EDGES_B = [0, 0.02, 0.05, 0.10, 0.20, np.inf]
KEYS_B = ['<0.02', '0.02-0.05', '0.05-0.10', '0.10-0.20', '>0.20']
LAB_B = ['<0.02', '0.02–\n0.05', '0.05–\n0.10', '0.10–\n0.20', '>0.20']
xb = np.arange(len(KEYS_B))
est = []
for lo, hi in zip(EDGES_B[:-1], EDGES_B[1:]):
    mm_ = (bd >= lo) & (bd < hi)
    est.append((np.percentile(dqe[mm_], [25, 50, 75]), np.percentile(sq[mm_], [25, 50, 75]),
                np.median(snr[mm_]), int(mm_.sum())))

# ════════════════════ 版面(单位 mm,183 × 145)════════════════════
FIG_W, FIG_H = 183.0, 212.0
fig = plt.figure(figsize=(W2, FIG_H / 25.4))


def add_ax(x0, top, w, h):
    """以 mm 给出左缘、上缘(距画布顶)、宽、高。"""
    return fig.add_axes([x0 / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H])


XL, WL = 14.0, 72.0               # 左列(a、c)
XR, WR = 106.0, 74.0              # 右列(b、d)
TOP1, H1 = 15.5, 50.5             # 第一行:上方留给 a 的三行图例
TOP2, H2 = 87.0, 44.5             # 第二行:上方留给 d 的单行图例
ax_a = add_ax(XL, TOP1, WL, H1)
ax_b = add_ax(XR, TOP1, WR, H1)
ax_c = add_ax(XL, TOP2, WL, H2)
ax_d = add_ax(XR, TOP2, WR, H2)
TOP3, H3 = 155.0, 44.5
ax_e = add_ax(XL, TOP3, WL, H3)
ax_f = add_ax(XR, TOP3, WR, H3)

# 面板字母:8 pt 加粗,同一行齐平,横向与各列纵轴标题的左缘对齐
LET_Y1 = 1 - (1.2 + 2.9) / FIG_H
LET_Y2 = 1 - (TOP2 - 7.2 + 2.9) / FIG_H
panel_fig(fig, 'a', 1.2 / FIG_W, LET_Y1)
panel_fig(fig, 'b', (XR - 13.0) / FIG_W, LET_Y1)
panel_fig(fig, 'c', 1.2 / FIG_W, LET_Y2)
panel_fig(fig, 'd', (XR - 13.0) / FIG_W, LET_Y2)
LET_Y3 = 1 - (TOP3 - 7.2 + 2.9) / FIG_H
panel_fig(fig, 'e', 1.2 / FIG_W, LET_Y3)
panel_fig(fig, 'f', (XR - 13.0) / FIG_W, LET_Y3)

# ─────────────── a  校正量对瓦片起伏 ───────────────
ax = ax_a
XLIM = (np.log10(15), np.log10(6500))
YLIM = (-3.2, 2.55)               # 上限由 2.1 提到 2.55:给左上角统计注记留出不压密度格的空间
GREY = LinearSegmentedColormap.from_list('grey_density', ['#F1F1F1', '#A9A9A9'], N=256)   # 浅灰:底图只作背景,让中位线浮出来
# 六边形在显示坐标里近似正六边形:ny = nx · (H/W) / √3
NX = 58
NY = int(round(NX * (H1 / WL) / np.sqrt(3)))
hb = ax.hexbin(x, y, gridsize=(NX, NY), extent=(*XLIM, *YLIM), bins='log', mincnt=1,
               cmap=GREY, linewidths=0.15, edgecolors='face', zorder=1)
ax.fill_between(bx, b25, b75, color='#000000', alpha=0.12, lw=0, zorder=2)
for yy_ in (b25, b75):
    ax.plot(bx, yy_, color=COL['ink'], lw=0.45, alpha=0.5, zorder=2)
# 山脊 / 谷地各自的分档中位数:同一组 20 个起伏档(边界 q 由全部测点给出),横坐标取各自档内中位
side = {}
for nm, kk in (('ridge', up[m]), ('valley', ~up[m])):
    sx_, sy_ = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        k = (x >= lo) & (x <= hi) & kk
        sx_.append(np.median(x[k])); sy_.append(np.median(y[k]))
    side[nm] = (np.array(sx_), np.array(sy_))
# 脊/谷中位与总体中位只差 ±0.1 个数量级左右(约 1 mm),彩线画在黑线之上,否则会被黑线盖住
ax.plot(bx, by, color=COL['ink'], lw=1.3, zorder=3)
ax.plot(*side['ridge'], color=UP, lw=0.9, zorder=4)
ax.plot(*side['valley'], color=DOWN, lw=0.9, zorder=4)
xf = np.array([bx.min() - 0.15, bx.max() + 0.15])
ax.plot(xf, fit.intercept + fit.slope * xf, color=COL['ink'], lw=0.9, ls=(0, (4, 2)), zorder=5)
ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
xt = [30, 100, 300, 1000, 3000]
ax.set_xticks(np.log10(xt)); ax.set_xticklabels(['30', '100', '300', '1,000', '3,000'])
yt = [0.001, 0.01, 0.1, 1, 10, 100]
ax.set_yticks(np.log10(yt)); ax.set_yticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])
ax.set_xlabel('Tile relief (m)', labelpad=2)
ax.set_ylabel(r'Correction, $|\Delta q\,/\,q_{\rm obs}|$ (%)', labelpad=2)
txt = ('Power-law exponent: %.2f' % fit.slope + '\n'
       + 'Binned medians: R² = %.2f' % fit.rvalue ** 2 + '\n'
       + 'All sites: Spearman ρ = %.2f' % rho)
ax.text(0.025, 0.975, txt, transform=ax.transAxes, va='top', ha='left', fontsize=FS_TICK,
        linespacing=1.45, color=COL['ink'])
# 图例放在坐标区上方(两列),不压任何密度格
h_den = tuple(Line2D([], [], ls='none', marker='h', ms=5.2, mec='none', mfc=GREY(v))
              for v in (0.05, 0.5, 1.0))
h_med = (Patch(facecolor='#000000', alpha=0.13, lw=0),
         Line2D([], [], color=COL['ink'], lw=1.3))
h_fit = Line2D([], [], color=COL['ink'], lw=0.9, ls=(0, (4, 2)))
h_up = Line2D([], [], color=UP, lw=0.9)
h_dn = Line2D([], [], color=DOWN, lw=0.9)
ax.legend([h_den, h_med, h_fit, h_up, h_dn],
          ['Sites (log density)', 'All-site median, IQR', 'Power-law fit to medians',
           'Ridge-side median ($B_D$ < 1)', 'Valley-side median ($B_D$ > 1)'],
          # 密度:三个由浅到深的六边形并排;中位线:灰带与黑线叠在一起(ndivide=1)
          handler_map={h_den: HandlerTuple(ndivide=None, pad=0.0), h_med: HandlerTuple(ndivide=1)},
          loc='lower left', bbox_to_anchor=(-0.01, 1.0), ncol=2, frameon=False,
          fontsize=FS_TICK, handlelength=2.0, handletextpad=0.5, columnspacing=1.2,
          labelspacing=0.3, borderaxespad=0.25, borderpad=0.2)

# ─────────────── b  各起伏档的校正量构成 ───────────────
ax = ax_b
SEQ = ['#E3F1DF', '#A9D3A0', '#529E5F', '#1D5B33']      # 浅 -> 深:<2、2–5、5–10、≥10(绿色系,不与 a 的蓝红、c/d 的蓝灰撞色)
xr = np.arange(len(rows))
bottom = np.zeros(len(rows))
handles = {}
for (key, lab), col in reversed(list(zip(CLS, SEQ))):   # 深色在下:≥10 -> 5–10 -> 2–5 -> <2
    v = np.array([r[key] for r in rows])
    handles[key] = ax.bar(xr, v, 0.68, bottom=bottom, color=col, edgecolor='white', lw=0.4,
                          label=lab, zorder=2)
    bottom += v
# 柱内数字 = >5% 的占比(5–10% 与 ≥10% 之和),写在这条分界之上;
# 2–5% 段比一行字还薄时(前两档)改写在该段之上,免得数字骑在段间白线上。
gt5 = np.array([r['frac_5_10'] + r['frac_ge10'] for r in rows])
for k_, v in enumerate(gt5):
    s_ = '0' if v == 0 else ('%.1f' % v if v < 10 else '%.0f' % v)
    f25 = rows[k_]['frac_2_5']
    yv = v + 1.2 if f25 >= 7 else v + f25 + 1.0
    ax.text(xr[k_], yv, s_, ha='center', va='bottom', fontsize=FS_TICK, color=COL['ink'], zorder=4)
ax.set_ylim(0, 100)
ax.set_xlim(-0.5, len(rows) - 0.5)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_xticks(xr); ax.set_xticklabels(LAB, fontsize=FS_TICK, linespacing=1.1)
ax.set_xlabel('Tile relief (m)', labelpad=2)
ax.set_ylabel('Sites (%)', labelpad=2)
leg_b = ax.legend([handles[c] for c, _ in CLS], [lab for _, lab in CLS],
                  loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=4, frameon=False,
                  fontsize=FS_TICK, handlelength=0.9, handletextpad=0.35, columnspacing=0.8,
                  borderaxespad=0.25, borderpad=0.2)

# ─────────────── c  校正量与随机误差(原图 7a,内容不变)───────────────
ax = ax_c
w = 0.17
for k, (c3, s3, r_, n_) in enumerate(est):
    ax.plot([xb[k] - w] * 2, [c3[0], c3[2]], color=COL['hero'], lw=2.4, solid_capstyle='butt', zorder=3)
    ax.plot(xb[k] - w, c3[1], 'o', color=COL['hero'], ms=4.6, zorder=4)
    ax.plot([xb[k] + w] * 2, [s3[0], s3[2]], color=COL['ref'], lw=2.4, solid_capstyle='butt', zorder=3)
    ax.plot(xb[k] + w, s3[1], 'o', color='white', mec=COL['ref'], mew=1.1, ms=4.6, zorder=4)
    ax.text(xb[k], max(c3[2], s3[2]) * 1.45, '×%.1f' % r_, ha='center', va='bottom', fontsize=FS_TICK,
            color=COL['ink'])
ax.set_yscale('log')
ax.set_ylim(0.03, 80)
ax.set_yticks([0.1, 1, 10]); ax.set_yticklabels(['0.1', '1', '10'])
ax.set_xlim(-0.5, len(KEYS_B) - 0.5)
ax.set_xticks(xb); ax.set_xticklabels(LAB_B, fontsize=FS_TICK, linespacing=1.1)
ax.set_xlabel('$|B_D - 1|$', labelpad=2)
ax.set_ylabel('Per-site value (mW m$^{-2}$)', labelpad=2)
ax.plot([], [], 'o', color=COL['hero'], ms=4.6, label='Correction $|\\Delta q|$')
ax.plot([], [], 'o', color='white', mec=COL['ref'], mew=1.1, ms=4.6, label='Random error $\\sigma_q$')
ax.legend(loc='upper left', frameon=False, fontsize=FS_TICK, handletextpad=0.3, borderaxespad=0.2)
ax.text(0.98, 0.03, '×: median SNR', transform=ax.transAxes, ha='right', va='bottom', fontsize=FS_TICK,
        color=COL['ref'])

# ─────────────── d  分级构成(原图 7b,内容不变)───────────────
ax = ax_d
GC = [('A', 'A: SNR ≥ 2', COL['hero']), ('B', 'B: 1–2', COL['hero_light']), ('C', 'C: < 1 or extrapolated', COL['ref_light'])]
bottom = np.zeros(len(KEYS_B))
fr = {}
for g, lab, col in GC:
    f = np.array([100 * (grade[(bd >= lo) & (bd < hi)] == g).mean() for lo, hi in zip(EDGES_B[:-1], EDGES_B[1:])])
    fr[g] = f
    ax.bar(xb, f, 0.66, bottom=bottom, color=col, edgecolor='white', lw=0.4, label=lab, zorder=2)
    bottom += f
for k in range(len(KEYS_B)):
    ax.text(xb[k], fr['A'][k] / 2, '%.0f' % fr['A'][k], ha='center', va='center', fontsize=FS_TICK,
            color='white')
ax.set_ylim(0, 100)
ax.set_xlim(-0.5, len(KEYS_B) - 0.5)
ax.set_xticks(xb); ax.set_xticklabels(LAB_B, fontsize=FS_TICK, linespacing=1.1)
ax.set_xlabel('$|B_D - 1|$', labelpad=2)
ax.set_ylabel('Sites (%)', labelpad=2)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=FS_TICK,
          handlelength=0.9, handletextpad=0.3, columnspacing=0.6, borderaxespad=0.25, borderpad=0.2)

# b 的图例前缀:「|Δq/q_obs|:」紧贴图例左侧、与图例同一基线(按渲染后的图例位置摆放)
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_lb = leg_b.get_window_extent(_r).transformed(ax_b.transAxes.inverted())
ax_b.text(_lb.x0 - 0.01, 0.5 * (_lb.y0 + _lb.y1), r'Correction $|\Delta q/q_{\rm obs}|$:',
          transform=ax_b.transAxes, ha='right', va='center', fontsize=FS_TICK, color=COL['ink'])
# 柱内数字的含义写在图例上一行(与 a 的图例第二行同高的空白处),右对齐
ax_b.text(_lb.x1, _lb.y1 + 0.005, 'Numbers: % of sites with correction > 5%',
          transform=ax_b.transAxes, ha='right', va='bottom', fontsize=FS_TICK, color=COL['ink'])

# ─────────────── e  算子误差的校准:两个外部检验集上误差落在 ±kσ 内的比例 ───────────────
# 8 朝向 + 蒙特卡洛 dropout 的合成离散度乘缩放系数 c(使两集合并后 ±2σ 覆盖 95.4%)。
sys.path.insert(0, ROOT)
from error_model import op_site_scale   # noqa: E402
cal = np.load(os.path.join(ROOT, 'mc_d4_calibration_tiles.npz'))
C_SC = op_site_scale()
kk = np.linspace(0, 3, 301)
ax = ax_e
ax.plot(kk, 100 * (2 * stats.norm.cdf(kk) - 1), color=COL['ref'], lw=0.9, ls=(0, (3, 2)),
        label='Gaussian expectation', zorder=2)
C_SET = {n_: float(np.quantile(cal['e_%d' % n_] / cal['sd_%d' % n_], 0.954) / 2.0) for n_ in (0, 1)}
CAL = {}
for n_, lab, col in ((0, 'Independently generated set', COL['hero']), (1, 'Distance-excluded set', COL['warn'])):
    r_ = cal['e_%d' % n_] / (C_SET[1 - n_] * cal['sd_%d' % n_])
    ax.plot(kk, [100 * np.mean(r_ <= k) for k in kk], color=col, lw=1.3, label=lab, zorder=3)
    CAL[n_] = {k: float(100 * np.mean(r_ <= k)) for k in (1, 2)}
for k in (1, 2):
    ax.axvline(k, color=COL['grid'], lw=0.6, zorder=1)
    ax.text(k + 0.04, 3, '±%dσ: %.0f%% / %.0f%%' % (k, CAL[0][k], CAL[1][k]), fontsize=FS_TICK, color=COL['ink'],
            va='bottom')
ax.set_xlim(0, 3); ax.set_ylim(0, 100)
ax.set_xticks([0, 1, 2, 3]); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_xlabel(r'Error bound ($k$ × calibrated $\sigma_{\rm op}$)', labelpad=2)
ax.set_ylabel('Tiles within bound (%)', labelpad=2)
ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.42), frameon=False, fontsize=FS_TICK, handlelength=1.6,
          borderaxespad=0.2)
S['calibration'] = dict(c=C_SC, c_by_set=C_SET, coverage=CAL, coverage_note='cross: each set uses c fitted on the other')
print('[fig6e] c = %.2f;±1σ 覆盖 %.1f / %.1f%%,±2σ 覆盖 %.1f / %.1f%%'
      % (C_SC, CAL[0][1], CAL[1][1], CAL[0][2], CAL[1][2]))

# ─────────────── f  逐瓦片预测误差与实际误差:离散度能区分难测与易测的瓦片 ───────────────
# 横轴为校准后的算子误差 c × (8 朝向 + 蒙特卡洛 dropout 的合成离散度),纵轴为相对有限元的实际误差;
# 虚线为 |误差| = σ 与 2σ。点落在 2σ 线下方的比例即 e 中 k = 2 处的覆盖率。
ax = ax_f
RHO = {}
for n_, lab, col in ((0, 'Independently generated set', COL['hero']), (1, 'Distance-excluded set', COL['warn'])):
    sg = C_SC * cal['sd_%d' % n_]
    er = cal['e_%d' % n_]
    ax.scatter(sg, er, s=5, color=col, alpha=0.55, linewidths=0, zorder=3, rasterized=True)
    RHO[n_] = float(stats.spearmanr(sg, er)[0])
_g = np.array([1e-4, 0.2])
ax.plot(_g, _g, color=COL['ref'], lw=0.8, ls=(0, (3, 2)), zorder=2)
ax.plot(_g, 2 * _g, color=COL['ref'], lw=0.8, ls=(0, (1, 1.5)), zorder=2)
ax.text(0.10, 0.085, '1σ', fontsize=FS_TICK, color=COL['ref'], ha='left', va='top')
ax.text(0.05, 0.135, '2σ', fontsize=FS_TICK, color=COL['ref'], ha='right', va='bottom')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(2e-4, 0.2); ax.set_ylim(1e-6, 0.3)
ax.set_xticks([1e-3, 1e-2, 1e-1]); ax.set_xticklabels(['0.001', '0.01', '0.1'])
ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]); ax.set_yticklabels(['10$^{-5}$', '10$^{-4}$', '0.001', '0.01', '0.1'])
ax.set_xlabel(r'Predicted error, calibrated $\sigma_{\rm op}$ (in $B$)', labelpad=2)
ax.set_ylabel(r'Actual error $|B_{\rm op} - B_{\rm FE}|$', labelpad=2)
ax.text(0.03, 0.97, 'Spearman ρ = %.2f / %.2f' % (RHO[0], RHO[1]), transform=ax.transAxes, ha='left', va='top',
        fontsize=FS_TICK, color=COL['ink'])
S['calibration']['spearman'] = RHO
print('[fig6f] 预测误差与实际误差的秩相关 %.2f / %.2f' % (RHO[0], RHO[1]))

save(fig, os.path.join(OUT, 'Fig6'))
plt.close(fig)

json.dump(S, open(os.path.join(ROOT, 'terrain_control_stats.json'), 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in S.items() if k != 'relief_bins'},
                 ensure_ascii=False))
print('Fig. 6 done. A 级比例 %s;信噪比中位 %s' % (' / '.join('%.1f' % v for v in fr['A']),
                                              ' / '.join('%.2f' % s[2] for s in est)))
