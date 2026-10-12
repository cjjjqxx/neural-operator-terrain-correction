# -*- coding: utf-8 -*-
"""
Fig. 7 | The terrain signal in the observations themselves.



视觉语法(与图 8 逐条相同,只换信号色相:图 7 青绿 = 观测中的事实):
  实测估计   实心圆 + 同色实线区间;回归线 = 信号色实线,置信带 = 信号浅色
  对照/参照  中灰实心圆 + 灰色区间
  个体单元   浅灰小点;纵轴范围外的单元以边缘三角形标出
  分档统计   白心墨边圆
  参考线     无效应(0)= 墨色细实线;理论预测(斜率 1、1:1)= 灰色虚线
  文字       统计量墨色;参考线标注、表头灰色;不用彩色字
数值全部从产物现读;本脚本不依赖 figures/v2 以外的任何绘图辅助文件。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from nature_style_v2 import (set_style, COL, W2, MM, save, SERIES, FS_TICK, FS_LABEL,
                             panel_fig)

set_style(base=7)

# ---------------------------------------------------------------- 视觉语法
SIG, SIG_L = SERIES['evidence']          # 信号色:青绿
INK = COL['ink']
CTRL = COL['ref']                        # 对照/参照 #8A8A8A
UNIT = COL['ref_light']                  # 个体单元 #C8C8C8
GUIDE = '#9A9A9A'                        # 理论参照虚线
TXT2 = '#6B6B6B'                         # 次要文字
LAND, LAND_EDGE = '#E9E6E1', '#8C8279'   # 示意图的地层底色与地表线
MS_EST, MS_BIN = 4.6, 4.4               # 估计点、分档点直径(pt)
LW_CI, LW_FIT, LW_ZERO = 1.2, 1.4, 0.6
DASH = (0, (3, 2))
ARROW = '-|>,head_length=0.32,head_width=0.16'


def est(ax, x, y, color, ms=MS_EST, z=6):
    """估计点(实心)。不加白描边:白边会在短区间与圆点交界处切出缺口。"""
    ax.plot(x, y, 'o', ms=ms, mfc=color, mec=color, mew=0.5, ls='none', zorder=z)


def binpt(ax, x, y, z=7):
    """分档统计点(白心墨边)。"""
    ax.plot(x, y, 'o', ms=MS_BIN, mfc='white', mec=INK, mew=0.9, ls='none', zorder=z)


def hci(ax, lo, hi, y, color, lw=LW_CI, z=5):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle='butt', zorder=z)


def arrow(ax, x0, y0, x1, y1, color, lw=0.8, style=ARROW, z=3):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0),
                zorder=z)


def minus(s):
    return s.replace('-', '−')


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figs_v2')
ROOT = os.path.join(HERE, '..', '..')
os.makedirs(OUT, exist_ok=True)


def jload(name):
    with open(os.path.join(ROOT, name), encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------- 数据(全部现读)
pair = jload('paired_null_fairness_test.json')['全部']
# 配对半径与置换次数是检验脚本的设定,产物里不存;从产出它的脚本现读,并与 p 值的下限互验
import re
with open(os.path.join(ROOT, 'paired_null_fairness_test.py'), encoding='utf-8') as _f:
    _m = re.search(r'^RADIUS_KM,\s*N_PERM,\s*SEED\s*=\s*([\d.]+),\s*(\d+),', _f.read(), re.M)
RADIUS_KM, N_PERM = float(_m.group(1)), int(_m.group(2))
assert abs(pair['p_B'] * (N_PERM + 1) - 1) < 1e-6, (pair['p_B'], N_PERM)   # p = 1/(N+1):高于全部置换
c15 = jload('c15_ordinal_test.json')
dav = jload('davies_validation.json')
# 斜率的 95% 区间按 Davies 格点聚类自助(同一格点内测点不独立),与正文 3.3 同一口径
for _k, _v in jload('magnitude_scaling_test.json')['davies'].items():
    dav[_k]['ci_lo'], dav[_k]['ci_hi'] = _v['ci_cluster']
pts = np.load(os.path.join(ROOT, 'davies_points.npz'))
ordered = [g for g in c15['groups'] if g['in_ordinal_scale']]
notass = [g for g in c15['groups'] if not g['in_ordinal_scale']][0]
d1, d2 = [d['cliffs_delta'] for d in c15['adjacent_deltas']]
_s1_p = c15['step1_mw_p']
_s1_e = int(np.floor(np.log10(_s1_p)))
S1_P = '$p$ = %.0f × 10$^{%d}$' % (_s1_p / 10.0 ** _s1_e, _s1_e)

# ---------------------------------------------------------------- 版式(毫米)
FIG_W, FIG_H = 183.0, 112.0
fig = plt.figure(figsize=(W2, FIG_H * MM))


def ax_mm(x0, top, w, h):
    """x0 = 坐标区左缘距画布左缘,top = 坐标区上缘距画布上缘(mm)。"""
    return fig.add_axes([x0 / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H])


def ftext(x, top, s, **kw):
    kw.setdefault('fontsize', FS_TICK)
    kw.setdefault('color', INK)
    return fig.text(x / FIG_W, 1 - top / FIG_H, s, **kw)


CARD_X, CARD_W = 5.0, 44.0               # 检验卡:左缘与宽度
DATA_X = 62.0                            # 数据区左缘(纵轴标题在其左侧)
ROW = {'a': 2.0, 'b': 37.5, 'c': 74.0}   # 各行上缘
SEP = (35.6, 72.1)                       # 行间细分隔线


def card(L, top, title, lines, sch_h):
    """检验卡:字母 + 粗体标题(同一基线),其下示意坐标区(等比例,单位 mm),再下文字。"""
    panel_fig(fig, L, 1.5 / FIG_W, 1 - (top + 3.0) / FIG_H)
    ftext(CARD_X, top + 3.0, title, fontsize=FS_LABEL, fontweight='bold', va='baseline')
    sx = ax_mm(CARD_X, top + 4.8, CARD_W, sch_h)
    sx.set_xlim(0, CARD_W)
    sx.set_ylim(0, sch_h)
    sx.set_aspect('equal')
    sx.axis('off')
    sx.set_xticks([])
    sx.set_yticks([])
    ftext(CARD_X, top + 6.0 + sch_h, '\n'.join(lines), va='top', linespacing=1.3)
    return sx


def borehole(ax, x, z, depth, color=INK, lw=1.1, ms=3.2):
    ax.plot([x, x], [z, z - depth], color=color, lw=lw, solid_capstyle='butt', zorder=5)
    ax.plot(x, z + 0.5, marker='v', ms=ms, color=color, mew=0, zorder=5)


for t in SEP:
    fig.add_artist(plt.Line2D([1.5 / FIG_W, 181.5 / FIG_W], [1 - t / FIG_H] * 2,
                              color='#E3E3E3', lw=0.6))

# ══ (a) 邻近测点配对 ═════════════════════════════════════════════════════
top = ROW['a']
sx = card('a', top, 'Neighbouring-site pairs',
          ['Within-pair scatter, %s pairs:' % format(pair['n'], ','),
           'actual vs permuted correction',
           'Exceeds all %d permutations ($p$ ≤ %.4f)' % (N_PERM, pair['p_B'])], 12.0)
xs = np.linspace(0, CARD_W, 240)
surf = lambda x: 5.6 - 2.6 * np.cos(2 * np.pi * (x - 15.0) / 36.0)
sx.fill_between(xs, 0, surf(xs), color=LAND, lw=0)
sx.plot(xs, surf(xs), color=LAND_EDGE, lw=0.7)
XA1, XA2 = 15.0, 27.0
for xb in (XA1, XA2):
    borehole(sx, xb, surf(xb), 2.4)
sx.plot([XA1, XA1, XA2, XA2], [8.5, 9.1, 9.1, 8.5], color=TXT2, lw=0.6)
sx.text(0.5 * (XA1 + XA2), 9.3, '≤ %g km' % RADIUS_KM, ha='center', va='bottom', fontsize=FS_TICK, color=TXT2)
sx.text(CARD_W - 0.3, 0.5, 'shared background', ha='right', va='bottom', fontsize=FS_TICK,
        color=TXT2)

# 数据:单条数轴。两个对照与真实校正落在同一条线上;箭头自对照的点估计指向真实值,长度 = 优势
real, nB, nA = pair['real'], pair['nullB'], pair['nullA']
A_XL, A_YL = (-1.45, 1.05), (-1.50, 1.30)
ax = ax_mm(DATA_X, top + 4.0, 116.0, 20.0)
ax.plot([0, 0], A_YL, color=INK, lw=LW_ZERO, zorder=1)
ax.plot(A_XL, [0, 0], color=UNIT, lw=0.6, zorder=1)
hci(ax, pair['nullB_lo'], pair['nullB_hi'], 0, CTRL)
est(ax, nB, 0, CTRL)
est(ax, nA, 0, CTRL)
est(ax, real, 0, SIG, ms=MS_EST + 0.8)
for xv, lab, col in ((nA, 'Permuted,\nmagnitude only', TXT2),
                     (nB, 'Permuted,\npair structure kept', TXT2),
                     (real, 'Actual\ncorrection', INK)):
    ax.text(xv, 0.30, lab, ha='center', va='bottom', fontsize=FS_TICK, color=col, linespacing=1.15)
Y_ARR = (-0.60, -1.15)
for yy, start, m in zip(Y_ARR, (nB, nA), (pair['marginB'], pair['marginA'])):
    ax.plot([start, start], [-0.17, yy], color=UNIT, lw=0.5, zorder=1)
    arrow(ax, start, yy, real, yy, SIG)
    ax.text(real + 0.035, yy, '+%.2f pp' % m, ha='left', va='center', fontsize=FS_TICK, color=INK)
ax.plot([real, real], [Y_ARR[1] - 0.12, -0.15], color=SIG, lw=0.5, ls=(0, (2, 1.6)), zorder=2)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xlim(*A_XL)
ax.set_ylim(*A_YL)
ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
ax.set_xticklabels(['−1.0', '−0.5', '0', '0.5', '1.0'])
ax.set_xlabel('Reduction in within-pair scatter (%)', labelpad=2)

# ══ (b) 原研究者的现场判断 ═══════════════════════════════════════════════
top = ROW['b']
sx = card('b', top, "Original investigators' judgement",
          ['Operator |$B$ − 1| across the original',
           "authors' ordered categories",
           'First two categories: %s' % S1_P], 13.0)
STEP_W, STEP_H, STEP_X0 = 7.6, 2.2, 0.8
HILL = (0.7, 1.6, 2.7)
xs_all, zs_all = [], []
for i, a_ in enumerate(HILL):
    x0 = STEP_X0 + i * STEP_W
    xx_ = np.linspace(x0, x0 + STEP_W, 80)
    zz_ = STEP_H * (i + 1) + a_ * np.sin(np.pi * (xx_ - x0) / STEP_W) ** 2
    xs_all.append(xx_)
    zs_all.append(zz_)
xs_s = np.concatenate(xs_all)
zs_s = np.concatenate(zs_all)
sx.fill_between(xs_s, 0, zs_s, color=LAND, lw=0, step=None)
sx.plot(xs_s, zs_s, color=LAND_EDGE, lw=0.7)
for i in range(1, 3):                    # 阶梯竖边
    xe = STEP_X0 + i * STEP_W
    sx.plot([xe, xe], [STEP_H * i, STEP_H * (i + 1)], color=LAND_EDGE, lw=0.7)
X_END = STEP_X0 + 3 * STEP_W
arrow(sx, STEP_X0 + 0.6, STEP_H + HILL[0] + 2.4, X_END - 0.4, 3 * STEP_H + HILL[2] + 2.0, SIG,
      lw=0.9)
sx.text(X_END + 1.6, 9.2, 'Terrain effect', ha='left', va='center', fontsize=FS_TICK, color=INK)
sx.text(X_END + 1.6, 6.6, 'judged by the', ha='left', va='center', fontsize=FS_TICK, color=TXT2)
sx.text(X_END + 1.6, 4.0, 'original authors', ha='left', va='center', fontsize=FS_TICK, color=TXT2)

XB, X_NA, X_SEP = [0, 1.25, 2.5], 3.85, 3.17
TL = ['Not\nsignificant', 'Not\ncorrected', 'Corrected\nby authors', 'Not\nassessed']
B_TOP, B_H = top + 4.0, 18.5
b_axes = []
for i, (key, yl, yt, ytl, lab) in enumerate((
        ('median_absB1', (0, 0.0158), [0, 0.005, 0.010, 0.015], ['0', '0.005', '0.010', '0.015'],
         'Median |$B$ − 1|'),
        ('frac_gt5pct', (0, 25), [0, 10, 20], ['0', '10', '20'], '|$B$ − 1| > 0.05 (%)'))):
    ax = ax_mm(DATA_X + 2.0 + i * 64.0, B_TOP, 50.0, B_H)
    v = [g[key] for g in ordered]
    assert max(v + [notass[key]]) < yl[1]
    ax.plot(XB, v, color=SIG, lw=0.8, zorder=3)
    for xi, vi in zip(XB, v):
        est(ax, xi, vi, SIG)
    est(ax, X_NA, notass[key], CTRL)
    ax.plot([X_SEP, X_SEP], yl, color=UNIT, lw=0.5, zorder=1)
    ax.set_xlim(-0.55, 4.40)
    ax.set_ylim(*yl)
    ax.set_yticks(yt)
    ax.set_yticklabels(ytl)
    ax.set_ylabel(lab, labelpad=2)
    ax.set_xticks(XB + [X_NA])
    ax.set_xticklabels(TL, linespacing=1.12)
    ax.tick_params(axis='x', length=0, pad=2.5)
    ax.spines['bottom'].set_visible(False)
    b_axes.append((ax, v))
b1, v1 = b_axes[0]
# 相邻类别的 Cliff's δ 写在各自连线旁:第一段在线左上方(右对齐,不越过第二个点),第二段在线上方
b1.text(0.5 * (XB[0] + XB[1]) + 0.10, 0.5 * (v1[0] + v1[1]) + 0.0010, minus('$\\delta$ = %+.2f' % d1),
        ha='right', va='bottom', fontsize=FS_TICK, color=INK)
b1.text(0.5 * (XB[1] + XB[2]) + 0.10, max(v1[1], v1[2]) + 0.0006, minus('$\\delta$ = %+.2f' % d2),
        ha='center', va='bottom', fontsize=FS_TICK, color=INK)
# 每类测点数:写在两个面板共用的类别标签之下(左右两面板类别相同,只写一次)
for xi, g in zip(XB + [X_NA], ordered + [notass]):
    b1.text(xi, -0.47, '$n$ = %s' % format(g['n'], ','), transform=b1.get_xaxis_transform(),
            ha='center', va='top', fontsize=FS_TICK, color=TXT2)

# ══ (c) 对照 Davies (2013) 独立场 ═══════════════════════════════════════
top = ROW['c']
K = '>= 0.05'
sl, clo, chi = dav[K]['slope'], dav[K]['ci_lo'], dav[K]['ci_hi']
sx = card('c', top, 'Independent heat-flow map',
          ['Residual from the Davies (2013) value',
           'vs relative terrain factor',
           'Slope %.2f [%.2f, %.2f] for |$B$ − 1| ≥ 0.05' % (sl, clo, chi)], 15.5)
# 示意:只交代残差的定义 —— 测点值相对 Davies 区域值的偏离。两口钻孔都在坡腰
# (既非谷地也非山脊),一正一负,不暗示残差方向与地形有关
xs = np.linspace(0, CARD_W, 240)
X_VAL, LAM = 5.5, 26.0                   # 谷底位置、波长;坡腰 = 谷底 ± 波长/4
surf = lambda x: 2.8 - 2.0 * np.cos(2 * np.pi * (x - X_VAL) / LAM)
sx.fill_between(xs, 0, surf(xs), color=LAND, lw=0)
sx.plot(xs, surf(xs), color=LAND_EDGE, lw=0.7)
Z_REG = 9.0
sx.plot([0, CARD_W], [Z_REG, Z_REG], color=CTRL, lw=0.9, ls=DASH)
sx.text(0.3, Z_REG + 0.7, 'Davies (2013) value', ha='left', va='bottom', fontsize=FS_TICK,
        color=TXT2)
XS_DN, XS_UP = X_VAL + LAM / 4, X_VAL + 1.25 * LAM
for xb, dz in ((XS_UP, 3.8), (XS_DN, -3.6)):
    zb = surf(xb)
    sx.plot([xb, xb], [zb + 1.0, min(Z_REG, Z_REG + dz) - 0.6], color=UNIT, lw=0.5,
            ls=(0, (1, 1.2)), zorder=2)
    sx.plot(xb, zb + 0.5, marker='v', ms=3.0, color=INK, mew=0, zorder=5)
    arrow(sx, xb, Z_REG, xb, Z_REG + dz, SIG, lw=0.8,
          style='<|-|>,head_length=0.30,head_width=0.15')
    sx.plot(xb, Z_REG + dz, 'o', ms=3.0, mfc=INK, mec=INK, zorder=6)
sx.text(XS_UP - 1.3, Z_REG + 1.9, 'residual', ha='right', va='center', fontsize=FS_TICK, color=INK)
sx.text(XS_DN + 1.3, Z_REG - 1.8, '(either sign)', ha='left', va='center', fontsize=FS_TICK,
        color=TXT2)

# 数据 c:残差对相对传递因子
x = pts['rel_B_minus_1'] * 100.0
y = pts['resid'] * 100.0
_lr = stats.linregress(x, y)                       # 截距取自同一批点;斜率须与产物一致
assert abs(_lr.slope - sl) < 2e-3, (_lr.slope, sl)
assert x.size == dav[K]['n'], (x.size, dav[K]['n'])
x0m, y0m = x.mean(), y.mean()                      # 回归线过 (x̄, ȳ)
C_YL, C_XL = (-96.0, 96.0), (-54.0, 53.0)
up = y > C_YL[1]
assert (y >= C_YL[0]).all() and x.min() > C_XL[0] and x.max() < 40.5
ax = ax_mm(DATA_X + 2.0, top + 3.5, 54.0, 26.0)
ax.scatter(x[~up], y[~up], s=1.2, c=CTRL, alpha=0.30, lw=0, zorder=2, rasterized=True)
ax.scatter(x[up], np.full(up.sum(), C_YL[1] - 3.0), marker='^', s=4, c=CTRL, alpha=0.45, lw=0,
           zorder=2, rasterized=True, clip_on=False)
ax.plot([-50, 40], [0, 0], color=INK, lw=LW_ZERO, zorder=3)
ax.plot([-50, 40], [-50, 40], color=GUIDE, lw=0.8, ls=DASH, zorder=4)
xr = np.linspace(x.min(), x.max(), 50)
ax.fill_between(xr, y0m + clo * (xr - x0m), y0m + chi * (xr - x0m), color=SIG_L, alpha=0.5, lw=0,
                zorder=4)
ax.plot(xr, y0m + sl * (xr - x0m), color=SIG, lw=LW_FIT, zorder=6)
# 五档残差中位数:横坐标取该档 B/<B>−1 的中位数
q = dav['quintiles_ge005']
qx = [float(np.median(x[(x >= 100 * t['lo']) & (x <= 100 * t['hi'])])) for t in q]
qy = [t['resid_median_pct'] for t in q]
binpt(ax, qx, qy)
ax.text(43.6, 40.0, '1:1', ha='left', va='center', fontsize=FS_TICK, color=TXT2)
ax.text(43.6, y0m + sl * (x.max() - x0m), 'Fit', ha='left', va='center', fontsize=FS_TICK, color=INK)
ax.set_xlim(*C_XL)
ax.set_ylim(*C_YL)
ax.spines['bottom'].set_bounds(C_XL[0], 40)
ax.set_xticks([-40, -20, 0, 20, 40])
ax.set_xticklabels(['−40', '−20', '0', '20', '40'])
ax.set_yticks([-80, -40, 0, 40, 80])
ax.set_yticklabels(['−80', '−40', '0', '40', '80'])
ax.set_xlabel('Relative terrain factor $B/\\langle B\\rangle$ − 1 (%)', labelpad=1.5,
              x=(0.5 * (C_XL[0] + 40) - C_XL[0]) / (C_XL[1] - C_XL[0]))
ax.set_ylabel('Residual vs Davies field (%)', labelpad=1.5)

# ══ (d) 斜率随纳入阈值 ═══════════════════════════════════════════════════
panel_fig(fig, 'd', 123.0 / FIG_W, 1 - (top + 3.0) / FIG_H)
keys = ['全部', '>= 0.05', '>= 0.10']
labs = ['All sites', '|$B$ − 1| ≥ 0.05', '|$B$ − 1| ≥ 0.10']
YD = [2.0, 1.0, 0.0]
D_XL, D_YL, D_COL = (-0.08, 1.08), (-0.6, 3.6), 1.44
axd = ax_mm(147.0, top + 3.5, 25.0, 26.0)
axd.plot([0, 0], [D_YL[0], 2.45], color=INK, lw=LW_ZERO, zorder=1)
axd.plot([1, 1], [D_YL[0], 2.45], color=GUIDE, lw=0.8, ls=DASH, zorder=1)
for k, yy in zip(keys, YD):
    assert D_XL[0] < dav[k]['ci_lo'] and dav[k]['ci_hi'] < 1.0
    hci(axd, dav[k]['ci_lo'], dav[k]['ci_hi'], yy, SIG)
    est(axd, dav[k]['slope'], yy, SIG)
    axd.text(D_COL, yy, format(dav[k]['n'], ','), ha='right', va='center', fontsize=FS_TICK,
             color=INK)
axd.text(D_COL, 2.7, '$n$', ha='right', va='bottom', fontsize=FS_TICK, color=TXT2)
axd.text(0.03, 2.7, 'No\nsignal', ha='left', va='bottom', fontsize=FS_TICK, color=TXT2,
         linespacing=1.1)
axd.text(0.97, 2.7, 'Full\namplitude', ha='right', va='bottom', fontsize=FS_TICK, color=TXT2,
         linespacing=1.1)
axd.set_yticks(YD)
axd.set_yticklabels(labs)
axd.spines['left'].set_visible(False)
axd.tick_params(axis='y', length=0, pad=2)
axd.set_xlim(*D_XL)
axd.set_ylim(*D_YL)
axd.spines['bottom'].set_bounds(*D_XL)
axd.set_xticks([0, 0.5, 1.0])
axd.set_xticklabels(['0', '0.5', '1'])
axd.set_xlabel('Regression slope', labelpad=1.5,
               x=(0.5 * (D_XL[0] + D_XL[1]) - D_XL[0]) / (D_COL + 0.02 - D_XL[0]))

save(fig, os.path.join(OUT, 'Fig7'))

# ---------------------------------------------------------------- 关键值(供核对)
print('图7   画布 %.0f × %.0f mm' % (FIG_W, FIG_H))
print('图7a  配对 n=%s | 真实 %+.2f%% | 保留结构对照 %+.2f%% [%.2f, %.2f] | 仅量级对照 %+.2f%%'
      % (format(pair['n'], ','), real, nB, pair['nullB_lo'], pair['nullB_hi'], nA))
print('      优势 %.2f / %.2f pp | p ≤ %.4f' % (pair['marginB'], pair['marginA'], pair['p_B']))
print('图7b  中位|B-1| %s | |B-1|>0.05 占比 %s | 未评估 %.4f / %.1f%%'
      % (' / '.join('%.4f' % g['median_absB1'] for g in ordered),
         ' / '.join('%.1f' % g['frac_gt5pct'] for g in ordered),
         notass['median_absB1'], notass['frac_gt5pct']))
print('      n %s | Cliff δ %+.2f / %+.2f | 第一级单侧 p = %.3g(三类 JT z = %.2f,不报)'
      % (' / '.join(format(g['n'], ',') for g in ordered + [notass]), d1, d2,
         _s1_p, c15['jonckheere_z']))
print('图7c  n=%d | 斜率 %.2f [%.2f, %.2f] | 截距 %.2f%% | 上缘三角 %d | 五档中位 %s | 档中心 x %s'
      % (x.size, sl, clo, chi, _lr.intercept, up.sum(),
         ' / '.join('%.1f' % v for v in qy), ' / '.join('%.1f' % v for v in qx)))
print('图7d  ' + ' | '.join('%s: %.2f [%.2f, %.2f] n=%s' % (k, dav[k]['slope'], dav[k]['ci_lo'],
                                                          dav[k]['ci_hi'], format(dav[k]['n'], ','))
                            for k in keys))
