# -*- coding: utf-8 -*-
"""
Fig. 8 | Valley siting bias and its regional consequence.



视觉语法与图 7(fig07_observational_tests.py)逐条相同,只把信号色换成紫色(采样偏倚 = 本文的发现)。
示意中的等温线用全套的强调色(砖红);全图其余元素只用紫、灰两族。
数值全部从产物现读;本脚本不依赖 figures/v2 以外的任何绘图辅助文件。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
import numpy as np
import matplotlib.pyplot as plt
from nature_style_v2 import set_style, COL, W2, MM, save, SERIES, FS_TICK, panel_fig

set_style(base=7)

# ---------------------------------------------------------------- 视觉语法(与图 7 相同)
SIG, SIG_L = SERIES['bias']              # 信号色:紫
INK = COL['ink']
CTRL = COL['ref']                        # 对照/参照 #8A8A8A
UNIT = COL['ref_light']                  # 个体单元 #C8C8C8
TXT2 = '#6B6B6B'
LAND, LAND_EDGE = '#E9E6E1', '#8C8279'   # 示意图的地层底色与地表线(与图 7 示意卡相同)
ISO = COL['accent']                      # 等温线
MS_EST, MS_BIN = 4.6, 4.4
LW_BIN, LW_FIT, LW_ZERO = 0.8, 1.4, 0.6
BIN_C = '#555555'                        # 分档区间线:深灰,弱于信号色的拟合线
DASH = (0, (3, 2))
ARROW = '-|>,head_length=0.32,head_width=0.16'


def binpt(ax, x, y, z=7):
    ax.plot(x, y, 'o', ms=MS_BIN, mfc='white', mec=INK, mew=0.9, ls='none', zorder=z)


def minus(s):
    return s.replace('-', '−')


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figs_v2')
ROOT = os.path.join(HERE, '..', '..')
os.makedirs(OUT, exist_ok=True)


def jload(name):
    with open(os.path.join(ROOT, name), encoding='utf-8') as f:
        return json.load(f)


pa = jload('province_artifact_test.json')
cells = np.load(os.path.join(ROOT, 'province_cells_2deg.npz'))

# ---------------------------------------------------------------- 版式(毫米)
FIG_W, FIG_H = 183.0, 111.0
fig = plt.figure(figsize=(W2, FIG_H * MM))


def ax_mm(x0, top, w, h):
    """x0 = 坐标区左缘距画布左缘,top = 坐标区上缘距画布上缘(mm)。"""
    return fig.add_axes([x0 / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H])


def letter(Lt, x, top):
    panel_fig(fig, Lt, x / FIG_W, 1 - top / FIG_H)


TOP1, TOP2 = 4.5, 61.0                   # 上下两行坐标区上缘

v = np.load(os.path.join(ROOT, 'valley_bias_matched.npz'))
S, L = v['kind'] == 'site', v['kind'] == 'local'
dd = 100.0 * ((v['B_fem'][S] > 1).astype(float) - (v['B_fem'][L] > 1).astype(float))
xx = np.log10(np.clip(v['relief'][S], 10, None))
blk = (np.floor(v['lon'][S] / 2) * 1000 + np.floor(v['lat'][S] / 2)).astype(int)
vt = jload('valley_trend_test.json')
edges = np.quantile(xx, np.linspace(0, 1, 8))
bx = np.array([np.median(xx[(xx >= a) & (xx <= b)]) for a, b in zip(edges[:-1], edges[1:])])
bidx = np.clip(np.digitize(xx, edges[1:-1]), 0, len(bx) - 1)
rng = np.random.default_rng(20260930)
ub = np.unique(blk); mem = {b: np.where(blk == b)[0] for b in ub}
bm = np.array([dd[bidx == k].mean() for k in range(len(bx))])
xg = np.linspace(xx.min(), xx.max(), 60)
boots_bins, boots_fit = [], []
for _ in range(1000):
    ii = np.concatenate([mem[b] for b in rng.choice(ub, len(ub))])
    boots_bins.append([dd[ii][bidx[ii] == k].mean() for k in range(len(bx))])
    A = np.column_stack([np.ones(ii.size), xx[ii] - 3.0]); c, *_ = np.linalg.lstsq(A, dd[ii], rcond=None)
    boots_fit.append(c[0] + c[1] * (xg - 3.0))
lo, hi = np.percentile(boots_bins, [2.5, 97.5], axis=0)
flo, fhi = np.percentile(boots_fit, [2.5, 97.5], axis=0)
fit = vt['excess_at_1000m_pp'] + vt['slope_pp_per_decade'] * (xg - 3.0)
# ---- 计算部分到此为止;以下为绘图 ----
assert len(bx) == 7

# ══ (a) 机制示意 ═════════════════════════════════════════════════════════
# 单位 mm(等比例)。剖面:脊 18、66,谷 42、90(右端止于谷底);等温线深 6、12、18,
# 振幅随深度按 exp(−kd) 衰减、与地表同相 —— 谷下离地表更近且彼此更密(聚拢),脊下相反。
letter('a', 1.5, TOP1 - 0.5)
SW, SH = 106.0, 46.0
sx = ax_mm(4.0, TOP1, SW, SH)
sx.set_xlim(0, SW)
sx.set_ylim(0, SH)
sx.set_aspect('equal')
sx.axis('off')
sx.set_xticks([])
sx.set_yticks([])                        # 关掉的轴仍会留刻度标签对象,画布越界检查会误报
LAMB, X_R, Z0, AMP = 48.0, 18.0, 27.0, 6.0
KW = 2 * np.pi / LAMB
X_END = X_R + 1.5 * LAMB                 # = 90,第二道谷底
zs = lambda x: Z0 + AMP * np.cos(KW * (x - X_R))
xs = np.linspace(0, X_END, 600)
sx.fill_between(xs, 2.0, zs(xs), color=LAND, lw=0, zorder=1)
sx.plot(xs, zs(xs), color=LAND_EDGE, lw=0.9, zorder=3)
for d in (6.0, 12.0, 18.0):
    sx.plot(xs, Z0 - d + AMP * np.exp(-KW * d) * np.cos(KW * (xs - X_R)), color=ISO, lw=0.7, zorder=2)
sx.text(X_END - 1.0, Z0 - 18.0 - AMP * np.exp(-KW * 18.0) - 1.0, 'isotherms', ha='right', va='top',
        fontsize=FS_TICK, color=TXT2)


def hole(x, col, depth=7.5):
    z = zs(x)
    sx.plot([x, x], [z, z - depth], color=col, lw=1.3, solid_capstyle='butt', zorder=5)
    sx.plot(x, z + 0.75, marker='v', ms=3.6, color=col, mew=0, zorder=5)


X_V1, X_R2 = X_R + 0.5 * LAMB, X_R + LAMB
for xb in (X_V1 - 2.2, X_V1 + 2.2, X_END - 1.6):     # 谷底钻孔(紫)
    hole(xb, SIG)
X_RB = X_R + 3.5                                     # 脊上钻孔(灰)
hole(X_RB, CTRL)
# 脊、谷的标注:粗体一行说明 B,其下两行说明等温线与读数;细引线指向对应钻孔
for xc, xb, head, sub in ((X_R, X_RB, 'Ridge: $\\boldsymbol{B}$ < 1', 'isotherms spread,\nflux reads low'),
                          (X_V1, X_V1 - 2.2, 'Valley: $\\boldsymbol{B}$ > 1', 'isotherms crowd,\nflux reads high')):
    sx.text(xc, 44.6, head, ha='center', va='bottom', fontsize=FS_TICK, color=INK, fontweight='bold')
    sx.text(xc, 44.3, sub, ha='center', va='top', fontsize=FS_TICK, color=TXT2, linespacing=1.15)
    sx.plot([xb, xb], [37.4, zs(xb) + 2.0], color=UNIT, lw=0.5, zorder=4)
# 瓦片起伏 = 瓦片内高程最大值减最小值:两条细虚线自最高的脊顶与最低的谷底引出,双箭头量其差
X_ARR = X_END + 3.6
sx.plot([X_R2, X_ARR + 1.0], [Z0 + AMP] * 2, color=TXT2, lw=0.5, ls=DASH, zorder=4)
sx.plot([X_END, X_ARR + 1.0], [Z0 - AMP] * 2, color=TXT2, lw=0.5, ls=DASH, zorder=4)
sx.annotate('', xy=(X_ARR, Z0 + AMP), xytext=(X_ARR, Z0 - AMP),
            arrowprops=dict(arrowstyle='<|-|>,head_length=0.3,head_width=0.14', color=INK, lw=0.6,
                            shrinkA=0, shrinkB=0), zorder=4)
sx.text(X_ARR + 1.6, Z0 + 1.6, 'Tile relief', ha='left', va='bottom', fontsize=FS_TICK, color=INK)
sx.text(X_ARR + 1.6, Z0 + 1.0, 'max − min\nelevation', ha='left', va='top', fontsize=FS_TICK,
        color=TXT2, linespacing=1.15)
sx.text(4.0, 3.2, 'In high-relief terrain, boreholes favour valley floors', ha='left', va='bottom',
        fontsize=FS_TICK, color=INK)

# ---- b 的绘图 ----
letter('b', 116.0, TOP1 - 0.5)
ax = ax_mm(130.0, TOP1 + 1.5, 50.0, 38.5)
A_XL = (xx.min() - 0.06, xx.max() + 0.06)
A_YL = (-22.0, 31.0)
assert lo.min() > A_YL[0] and hi.max() < A_YL[1] and flo.min() > A_YL[0] and fhi.max() < A_YL[1]
ax.plot(A_XL, [0, 0], color=INK, lw=LW_ZERO, zorder=1)
ax.fill_between(xg, flo, fhi, color=SIG_L, alpha=0.45, lw=0, zorder=2)
ax.plot(xg, fit, color=SIG, lw=LW_FIT, zorder=3)
ax.vlines(bx, lo, hi, color=BIN_C, lw=LW_BIN, zorder=4)
binpt(ax, bx, bm, z=5)
ax.text(0.035, 0.975, '+%.1f pp per 10-fold relief\n95%% CI %.1f–%.1f\none-sided $p$ = %.3f'
        % (vt['slope_pp_per_decade'], vt['slope_ci'][0], vt['slope_ci'][1], vt['p_one_sided']),
        transform=ax.transAxes, ha='left', va='top', fontsize=FS_TICK, color=INK, linespacing=1.3)
ax.set_xlim(*A_XL)
ax.set_ylim(*A_YL)
ax.set_xticks(np.log10([30, 100, 300, 1000, 3000]))
ax.set_xticklabels(['30', '100', '300', '1,000', '3,000'])
ax.set_yticks([-20, -10, 0, 10, 20, 30])
ax.set_yticklabels(['−20', '−10', '0', '10', '20', '30'])
ax.set_xlabel('Tile relief (m)', labelpad=2)
ax.set_ylabel('Excess of boreholes with $B$ > 1\nover local random sites (pp)', labelpad=2,
              linespacing=1.25)

# ══ (c) 2° 格子:点面积 ∝ 测点数 ════════════════════════════════════════════
letter('c', 1.5, TOP2 - 3.0)
ax = ax_mm(16.0, TOP2, 51.0, 40.0)
vf, ch, ns = cells['valley_frac_pct'], cells['change_pct'], cells['n_sites']
assert vf.size == ns.size == pa['n_cells']
SZ = 0.09                                # 点面积(pt²)/ 测点数
C_XL = (vf.min() - 3, vf.max() + 3)
C_YL = (-4.6, 2.6)
ax.plot(C_XL, [0, 0], color=INK, lw=LW_ZERO, zorder=1)
inside = (ch >= C_YL[0]) & (ch <= C_YL[1])
ax.scatter(vf[inside], ch[inside], s=SZ * ns[inside], c=UNIT, lw=0.3, edgecolors=CTRL, alpha=0.85,
           zorder=2)
m_hi, m_lo = ch > C_YL[1], ch < C_YL[0]
_dy = 0.03 * (C_YL[1] - C_YL[0])
for sel, yv, mk in ((m_hi, C_YL[1] - _dy, '^'), (m_lo, C_YL[0] + _dy, 'v')):
    ax.scatter(vf[sel], np.full(sel.sum(), yv), marker=mk, s=14, c=UNIT, lw=0.4, edgecolors=CTRL,
               zorder=3, clip_on=False)
qe = np.quantile(vf, np.linspace(0, 1, 7))
c_med = []
for a_, b_ in zip(qe[:-1], qe[1:]):
    m = (vf >= a_) & (vf <= b_)
    q25, q50, q75 = np.percentile(ch[m], [25, 50, 75])
    xc = np.median(vf[m])
    ax.plot([xc, xc], [q25, q75], color=BIN_C, lw=LW_BIN, zorder=4, solid_capstyle='butt')
    binpt(ax, xc, q50, z=5)
    c_med.append((xc, q50, q25, q75))
# 统计量写在右上角(谷地占比 > 70%、变化 > 0.8% 处无格子;上缘三角形在其上方)
assert not ((vf > 70) & (ch > 0.8) & (ch <= C_YL[1])).any()
ax.text(0.97, 0.90, minus('$\\rho$ = %.2f\n%d cells' % (pa['cell_valley_spearman'], pa['n_cells'])),
        transform=ax.transAxes, ha='right', va='top', fontsize=FS_TICK, color=INK, linespacing=1.3)
# 点大小图例(左下角,该处无格子):最小档 = 格内最少测点数(即入选下限),最大档 = 最多测点数取整到百,
# 中档 = 二者几何中值的数量级。横向间距按毫米排,避免大圆压住相邻数字。
k_lo = int(ns.min())
k_hi = int(np.round(ns.max(), -2))
k_mid = int(10 ** np.round(np.log10(np.sqrt(k_lo * k_hi))))
u_mm = (C_XL[1] - C_XL[0]) / 51.0        # 每毫米的数据单位(坐标区宽 51 mm)
LEG_T, LEG_Y0, LEG_DY = -2.50, -3.02, -0.55   # 标题行与三行图例的纵坐标(竖排,大圆不压相邻数字)
x_mk = C_XL[0] + 3.4 * u_mm              # 圆心:离左缘 3.4 mm,容得下最大档
for i_, kk in enumerate((k_lo, k_mid, k_hi)):
    ax.scatter([x_mk], [LEG_Y0 + i_ * LEG_DY], s=SZ * kk, c=UNIT, lw=0.3, edgecolors=CTRL, zorder=6)
    ax.text(x_mk + 2.4 * u_mm, LEG_Y0 + i_ * LEG_DY, format(kk, ','), ha='left', va='center',
            fontsize=FS_TICK, color=TXT2)
ax.text(C_XL[0] + 1.4 * u_mm, LEG_T, 'Sites per cell', ha='left', va='center', fontsize=FS_TICK,
        color=TXT2)
assert not ((vf < C_XL[0] + 20 * u_mm) & (ch < LEG_T + 0.3) & (ch >= C_YL[0])).any()   # 图例区无格子
ax.set_xlim(*C_XL)
ax.set_ylim(*C_YL)
ax.set_xticks([20, 40, 60, 80, 100])
ax.set_yticks([-4, -3, -2, -1, 0, 1, 2])
ax.set_yticklabels(['−4', '−3', '−2', '−1', '0', '1', '2'])
ax.set_xlabel('Boreholes in valley setting (%)', labelpad=2)
ax.set_ylabel('Change in regional mean flux (%)', labelpad=2)

# ══ (d) 八个构造省:实测 → 校正后 ══════════════════════════════════════════
# 按谷地占比降序(正文:"谷地占比最高的四个省"显著下降,排序直接对应这句话)。
# 空心圆 = 实测均值,箭头尖 = 校正后均值;箭头短于符号时改以实点标出校正后均值。
NAME = {'Rocky Mountains (US)': 'Rocky Mts (US)', 'Himalaya-Tibet': 'Himalaya–Tibet'}
prov = sorted(pa['provinces'], key=lambda r: -r['valley_frac_pct'])
yy = np.arange(len(prov))[::-1].astype(float)
letter('d', 75.0, TOP2 - 3.0)
D_X0, D_W = 121.0, 33.0                  # 坐标区左缘与宽度(mm)
D_XL = (63.0, 101.0)
X_N, X_VF, X_CH = 111.5, 118.0, 182.0     # 数值列右缘(mm):测点数、谷地占比、变化与区间
ax = ax_mm(D_X0, TOP2, D_W, 40.0)
mm2d = (D_XL[1] - D_XL[0]) / D_W         # 每毫米对应的数据单位
col_x = lambda x_mm: (x_mm - D_X0) / D_W  # 画布毫米 → 坐标区分数(列文字用 yaxis transform)
tr = ax.get_yaxis_transform()
R_OPEN = 0.5 * 3.8 * 25.4 / 72           # 空心圆半径(mm)
n_dot = []
for y, r in zip(yy, prov):
    sig = r['ci_hi'] < 0
    col = SIG if sig else CTRL
    m0, m1 = r['mean_measured'], r['mean_corrected']
    assert D_XL[0] < min(m0, m1) and max(m0, m1) < D_XL[1]
    ax.plot(m0, y, 'o', ms=3.8, mfc='white', mec=col, mew=0.9, zorder=4)
    if abs(m1 - m0) / mm2d > R_OPEN + 1.3:       # 够长:箭头自圆边指向校正后均值
        x_start = m0 + np.sign(m1 - m0) * R_OPEN * mm2d
        ax.annotate('', xy=(m1, y), xytext=(x_start, y),
                    arrowprops=dict(arrowstyle=ARROW, color=col, lw=1.2, shrinkA=0, shrinkB=0),
                    zorder=5)
    else:                                        # 太短:实点 = 校正后均值
        ax.plot(m1, y, 'o', ms=1.7, mfc=col, mec=col, zorder=5)
        n_dot.append(r['province'])
    ax.text(col_x(X_N), y, format(r['n'], ','), transform=tr, ha='right', va='center',
            fontsize=FS_TICK, color=INK)
    ax.text(col_x(X_VF), y, '%d' % r['valley_frac_pct'], transform=tr, ha='right', va='center',
            fontsize=FS_TICK, color=INK)
    ax.text(col_x(X_CH), y, minus('%+.1f [%+.1f, %+.1f]' % (r['change_pct'], r['ci_lo'], r['ci_hi'])),
            transform=tr, ha='right', va='center', fontsize=FS_TICK, color=INK if sig else TXT2)
Y_HEAD = len(prov) - 0.45
for xm, s_ in ((X_N, '$n$'), (X_VF, 'Valley\n(%)'), (X_CH, 'Change (%)\n[95% CI]')):
    ax.text(col_x(xm), Y_HEAD, s_, transform=tr, ha='right', va='bottom', fontsize=FS_TICK,
            color=TXT2, linespacing=1.1)
ax.text(0.5, Y_HEAD, '○ measured → corrected', transform=tr, ha='center', va='bottom',
        fontsize=FS_TICK, color=TXT2)
ax.set_yticks(yy)
ax.set_yticklabels([NAME.get(r['province'], r['province']) for r in prov])
ax.tick_params(axis='y', length=0, pad=(D_X0 - X_N + 8.0) / 25.4 * 72)
ax.spines['left'].set_visible(False)
ax.set_ylim(-0.6, len(prov) - 0.45)
ax.set_xlim(*D_XL)
ax.set_xticks([70, 80, 90, 100])
ax.set_xlabel('Regional mean heat flow (mW m$^{-2}$)', labelpad=2)

save(fig, os.path.join(OUT, 'Fig8'))

# ---------------------------------------------------------------- 关键值(供核对)
print('图8   画布 %.0f × %.0f mm' % (FIG_W, FIG_H))
print('图8b  七档超出 %s' % ' / '.join(minus('%+.1f' % b_) for b_ in bm))
print('      七档 95%% CI %s' % ' | '.join('[%.1f, %.1f]' % (a_, b_) for a_, b_ in zip(lo, hi)))
print('      斜率 %+.1f pp/十倍起伏, 95%% CI %.1f–%.1f, 单侧 p = %.3f | n 对 = %d, 块 = %d'
      % (vt['slope_pp_per_decade'], vt['slope_ci'][0], vt['slope_ci'][1], vt['p_one_sided'],
         vt['n_pairs'], vt['n_blocks']))
print('图8c  %d 格 | Spearman ρ = %.3f | 范围外 上 %d、下 %d | 六档中位 %s | 图例键 %d / %d / %d'
      % (pa['n_cells'], pa['cell_valley_spearman'], m_hi.sum(), m_lo.sum(),
         ' / '.join('%.2f' % m_[1] for m_ in c_med), k_lo, k_mid, k_hi))
print('图8d  ' + ' | '.join('%s n=%d %.1f→%.1f %+.1f%% [%.1f, %.1f] 谷地 %d%%'
                           % (r['province'], r['n'], r['mean_measured'], r['mean_corrected'],
                              r['change_pct'], r['ci_lo'], r['ci_hi'], r['valley_frac_pct'])
                           for r in prov))
print('      置信区间整体 < 0:%s | 以实点表示校正后均值:%s'
      % (', '.join(r['province'] for r in prov if r['ci_hi'] < 0), ', '.join(n_dot)))
