# -*- coding: utf-8 -*-
"""
Fig. 4 | Terrain transfer factor across a range of relief.

核心结论:从平坦到极端地形,算子预测的 B 场在形态与幅度上都与有限元参考解一致,
残差比信号本身小一到两个数量级;B 的正负严格跟随山脊/山谷几何。

"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from nature_style_v2 import (set_style, COL, CMAP_B, CMAP_RES, CMAP_DEM, W2,
                             cbar, save, despine_all, panel_fig, FS_PANEL)

set_style(base=7)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figs_v2')
ROOT = os.path.join(HERE, '..', '..')
os.makedirs(OUT, exist_ok=True)

val = np.load(os.path.join(ROOT, 'validation_independent_500.npz'))
dems, B_fem, B_fno, reliefs = val['dems'], val['B_fem'], val['B_fno'], val['reliefs']

# 每档挑 B 场结构最明显的一个瓦片(标准差最大),覆盖平坦到极端
EDGES = [0, 50, 200, 500, 1e9]
ROW_LAB = ['Flat\n<50 m', 'Moderate\n50–200 m',
           'High\n200–500 m', 'Extreme\n>500 m']
sel = []
for i in range(4):
    m = np.where((reliefs >= EDGES[i]) & (reliefs < EDGES[i + 1]))[0]
    sel.append(int(m[np.argmax([B_fem[j].std() for j in m])]) if len(m) else None)

NR = 4
L, R = 0.112, 0.992          # 左侧仅留行标签所需宽度
TOP, BOT = 0.945, 0.358      # 底部为色标 + f 面板留足空间
HGAP, WGAP = 0.022, 0.012
COLW = 0.148          # 四个方形面板的宽度
PROFW = 0.180         # 剖面面板更宽
rowh = (TOP - BOT - (NR - 1) * HGAP) / NR

FIG_H = 202 / 25.4
fig = plt.figure(figsize=(W2, FIG_H))

COL_TITLES = ['DEM elevation', 'Finite-element $B$', 'Operator $B$',
              'Residual', '$B$ along centre row']
xs = [L, L + COLW + WGAP, L + 2 * (COLW + WGAP), L + 3 * (COLW + WGAP)]
# e 列的纵轴刻度数字要占 4.8 mm,间隙必须比它宽,否则数字压到 d 列残差图上。
# 右边界仍留在 0.986 < R,没有越界。
xs.append(xs[3] + COLW + 5.5 * WGAP)
IMW = rowh * FIG_H / W2
IMX = [x + (COLW - IMW) / 2 for x in xs[:4]]

for x, t, lt in zip(IMX + [xs[4] - 0.030], COL_TITLES, 'abcde'):
    fig.text(x + 0.016, TOP + 0.016, t, ha='left', va='bottom',
             fontsize=7.2, fontweight='bold', color=COL['ink'])
    fig.text(x, TOP + 0.016, lt, ha='left', va='bottom',
             fontsize=FS_PANEL, fontweight='bold', color=COL['ink'])

# B 的偏离幅度在平坦档与极端档之间相差两个数量级。若四行共用一套色标,
# 前三行会被压成一片近白,读者无法判断算子在中低起伏地形上的表现。
# 因此按行自适应色标,并把每行的实际取值范围直接标在面板上,
# 绝对幅度的跨行比较交给最右列带数值坐标轴的剖面图。

for r, idx in enumerate(sel):
    if idx is None:
        continue
    y = TOP - (r + 1) * rowh - r * HGAP
    dem, bf, bn = dems[idx], B_fem[idx], B_fno[idx]
    res = bn - bf

    # 行标签居中对齐:各行文字长短不一,右对齐会让左缘参差不齐
    fig.text(L / 2, y + rowh / 2, ROW_LAB[r], ha='center', va='center',
             fontsize=7.2, linespacing=1.25, color=COL['ink'])

    ax = fig.add_axes([xs[0], y, COLW, rowh]); despine_all(ax)
    im_d = ax.imshow(dem, cmap=CMAP_DEM, aspect='equal')
    ax.text(0.03, 0.955, f'relief {reliefs[idx]:.0f} m', transform=ax.transAxes,
            fontsize=7.2, color='white', va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.18', fc='#00000099', ec='none'))

    b_lim = max(abs(bf.min() - 1), abs(bf.max() - 1))
    rlim = float(np.nanmax(np.abs(res)))

    ax = fig.add_axes([xs[1], y, COLW, rowh]); despine_all(ax)
    im_b = ax.imshow(bf, cmap=CMAP_B, vmin=1 - b_lim, vmax=1 + b_lim, aspect='equal')
    ax.text(0.03, 0.955, f'{bf.min():.3f}–{bf.max():.3f}', transform=ax.transAxes,
            fontsize=7.2, color=COL['ink'], va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.18', fc='#FFFFFFCC', ec='none'))

    ax = fig.add_axes([xs[2], y, COLW, rowh]); despine_all(ax)
    ax.imshow(bn, cmap=CMAP_B, vmin=1 - b_lim, vmax=1 + b_lim, aspect='equal')
    ax.text(0.03, 0.955, 'same scale', transform=ax.transAxes,
            fontsize=7.2, color=COL['ink'], va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.18', fc='#FFFFFFCC', ec='none'))

    ax = fig.add_axes([xs[3], y, COLW, rowh]); despine_all(ax)
    im_r = ax.imshow(res, cmap=CMAP_RES, vmin=-rlim, vmax=rlim, aspect='equal')
    ax.text(0.03, 0.955, f'±{rlim:.3f}', transform=ax.transAxes,
            fontsize=7.2, color=COL['ink'], va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.18', fc='#FFFFFFCC', ec='none'))

    # 剖面:过瓦片中心行
    ax = fig.add_axes([xs[4], y, PROFW, rowh])
    c = bf.shape[0] // 2
    xx = np.arange(bf.shape[1])
    ax.plot(xx, bf[c], color=COL['ref'], lw=1.0, label='Finite element')
    ax.plot(xx, bn[c], color=COL['hero'], lw=0.9, ls=(0, (3, 1.6)), label='Operator')
    ax.axhline(1.0, color=COL['grid'], lw=0.5, zorder=0)
    ax.set_xlim(0, bf.shape[1] - 1)
    ax.tick_params(labelsize=7.2, pad=1.5)
    # 四行统一两位小数;步长只取 1/2/5,避免 0.025 这类刻度被两位小数四舍五入成 0.98/1.03
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if r < NR - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Pixel along centre row', fontsize=7.2, labelpad=2)
    if r == 0:
        _prof_handles = ax.get_legend_handles_labels()
fig.legend(*_prof_handles, loc='upper left', bbox_to_anchor=(xs[4], BOT - 0.043),
           bbox_transform=fig.transFigure, frameon=False, fontsize=7.2,
           handlelength=1.8, handletextpad=0.5, labelspacing=0.3, borderpad=0.0,
           borderaxespad=0.0)

cbar(fig, im_d, [IMX[0], BOT - 0.055, IMW, 0.013],
     'Elevation (m), bottom row')

cax = fig.add_axes([IMX[1], BOT - 0.055, IMX[2] + IMW - IMX[1], 0.013])
cb = fig.colorbar(im_b, cax=cax, orientation='horizontal')
# 只留中间一个刻度。色标铺满整列之后,两端的 ridge / valley 正好顶到左右
# 相邻色标的标签上("ridge"撞"Elevation (m), bottom row","valley"撞
# "Residual, scaled per row");而方向在下面的 label 里已经写了
# "(ridge < 1 < valley)",两端刻度是重复信息。
cb.set_ticks([1.0])
cb.set_ticklabels(['$B$ = 1'])
cb.set_label('$B$ (ridge < 1 < valley), scaled per row',
             fontsize=7.2, labelpad=1.5)
cb.ax.tick_params(labelsize=7.2, length=1.8, width=0.5, pad=1.2)
cb.ax.minorticks_off()
cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor('#888888')

cax = fig.add_axes([IMX[3], BOT - 0.055, IMW, 0.013])
cb = fig.colorbar(im_r, cax=cax, orientation='horizontal')
cb.set_ticks([-rlim, 0, rlim]); cb.set_ticklabels(['−', '0', '+'])
cb.set_label('Residual, scaled per row',
             fontsize=7.2, labelpad=1.5)
cb.ax.tick_params(labelsize=7.2, length=1.8, width=0.5, pad=1.2)
cb.ax.minorticks_off()
cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor('#888888')

# ── f | 应用三步:中心像元 -> 深度平均 -> 校正通量 ─────────────────────
# 上面四行讲的是"算子能不能复现有限元的 B 场";这一行讲的是"拿到 B 场之后
# 怎么变成一个测点的校正通量"。用讨论里那个安第斯测点,数字与正文同源。
pts = np.load(os.path.join(ROOT, 'rebuild_points.npz'))
_i = int(np.argmin((pts['lon'] - (-69.93)) ** 2 + (pts['lat'] - (-32.68)) ** 2))
B_SURF = float(pts['B_surface'][_i])
B_DEP = float(pts['B'][_i])
Q_M = float(pts['q_measured'][_i])
Q_C = float(pts['q_corrected'][_i])
D_FB = 338.0

# f(D):从产物现算,不写死 —— 硬编码会让这张图脱离流水线,
# depth_decay_characterize.py 重跑后曲线变了图却不会变(audit_narrative_binding 会报)。
# 同一份产物也是 rebuild_correction.py 中 _F_DEPTH_V 的来源,两者一致性由
# check_pipeline_consistency.py 断言。
import json as _json
_prof = _json.load(open(os.path.join(ROOT, 'depth_decay_profiles.json'), encoding='utf-8'))
_Dset = sorted({int(k) for r in _prof for k in r.get('f', {})})
FD_M = np.array([0.0] + [float(d) for d in _Dset])
FD_V = np.array([1.0] + [float(np.median([r['f'][str(d)] for r in _prof
                                          if str(d) in r['f']])) for d in _Dset])

FY, FH = 0.082, 0.132             # f 面板的纵向位置与高度
fig.text(L, FY + FH + 0.020, 'f', ha='left', va='bottom',
         fontsize=FS_PANEL, fontweight='bold', color=COL['ink'])
fig.text(L + 0.016, FY + FH + 0.016,
         'Applying the field to one site: centre pixel, depth average, corrected flux',
         ha='left', va='bottom', fontsize=7.2, fontweight='bold', color=COL['ink'])

PW = 0.187                        # 三个子面板等宽
PGAP = 0.115                      # 面板之间留给箭头的间距
px = [L + 0.010, L + 0.010 + PW + PGAP, L + 0.010 + 2 * (PW + PGAP)]

ax = fig.add_axes([px[0], FY, PW, FH]); despine_all(ax)
_at = np.load(os.path.join(ROOT, 'andes_example_tile.npz'))
_bf = _at['B']
assert abs(float(_bf[_bf.shape[0] // 2, _bf.shape[1] // 2]) - B_SURF) < 1e-3
_lim = max(abs(_bf.min() - 1), abs(_bf.max() - 1))
ax.imshow(_bf, cmap=CMAP_B, vmin=1 - _lim, vmax=1 + _lim, aspect='equal')
_c = _bf.shape[0] // 2
ax.plot([_c], [_c], marker='o', ms=4.2, mfc='none', mec=COL['ink'], mew=1.0)
ax.plot([_c, _c], [0, _bf.shape[0] - 1], color=COL['ink'], lw=0.4, alpha=0.45)
ax.plot([0, _bf.shape[1] - 1], [_c, _c], color=COL['ink'], lw=0.4, alpha=0.45)
ax.set_title('Centre pixel of the tile', fontsize=7.2, pad=2.5, color=COL['ink'])

# f2 | 深度平均:B 的地表偏离在钻孔区间上按 f(D) 收缩
ax = fig.add_axes([px[1], FY, PW, FH])
_zz = np.linspace(0, 1200, 200)
ax.plot(_zz, np.interp(_zz, FD_M, FD_V), color=COL['ref'], lw=1.1)
_fD = float(np.interp(D_FB, FD_M, FD_V))
ax.axvline(D_FB, color=COL['hero'], lw=0.8, ls=(0, (3, 1.6)))
ax.plot([D_FB], [_fD], marker='o', ms=3.4, color=COL['hero'], zorder=5)
ax.annotate('D = %.0f m\nf = %.3f' % (D_FB, _fD), xy=(D_FB, _fD),
            xytext=(D_FB + 240, _fD + 0.05), fontsize=7.2, color=COL['ink'],
            arrowprops=dict(arrowstyle='-', lw=0.5, color=COL['ink']))
ax.set_xlim(0, 1200); ax.set_ylim(0.55, 1.02)
ax.set_xlabel('Borehole interval depth, $D$ (m)', fontsize=7.2, labelpad=2)
ax.set_ylabel('Retained fraction $f(D)$', fontsize=7.2, labelpad=2)
ax.tick_params(labelsize=7.2, pad=1.5)
ax.set_title('Depth average over the logged interval', fontsize=7.2,
             pad=2.5, color=COL['ink'])

# f3 | 校正通量:q / B
ax = fig.add_axes([px[2], FY, PW, FH])
_bars = ax.bar([0, 1], [Q_M, Q_C], width=0.52,
               color=[COL['grid'], COL['hero']], edgecolor=COL['ink'], linewidth=0.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Measured', 'Corrected'], fontsize=7.2)
ax.set_ylim(0, max(Q_M, Q_C) * 1.24)
ax.set_ylabel(r'Heat flow (mW m$^{-2}$)', fontsize=7.2, labelpad=2)
ax.tick_params(labelsize=7.2, pad=1.5)
for _b, _v in zip(_bars, [Q_M, Q_C]):
    ax.text(_b.get_x() + _b.get_width() / 2, _v + max(Q_M, Q_C) * 0.03,
            '%.1f' % _v, ha='center', va='bottom', fontsize=7.2, color=COL['ink'])
for _s in ('top', 'right'):
    ax.spines[_s].set_visible(False)
ax.set_title('Corrected flux, $q_{\\rm obs}\\,/\\,B_D$', fontsize=7.2, pad=2.5, color=COL['ink'])
_CAP_Y = FY - 0.041
for _px, _s in ((px[0], r'$B_s$ = %.3f' % B_SURF),
                (px[1], r'$B_D$ = %.3f' % B_DEP),
                (px[2], '+%.0f%%' % (100 * (Q_C - Q_M) / Q_M))):
    fig.text(_px + PW / 2, _CAP_Y, _s, ha='center', va='top', fontsize=7.2,
             fontweight='normal', color=COL['ink'])

# 面板之间的箭头
for _k in (0, 1):
    # 箭头止于下一面板纵轴标签之前:长 0.044,尾端离该面板左边界还有 0.065。
    # 宽度 0.022 —— 长宽比约 2:1;再粗就成了矮胖三角,不像箭头。
    _x0 = px[_k] + PW + 0.006
    fig.patches.append(plt.Arrow(_x0, FY + FH / 2, 0.044, 0, width=0.022,
                                 transform=fig.transFigure, color=COL['ink'],
                                 alpha=0.80, figure=fig))

save(fig, os.path.join(OUT, 'Fig3'))
print('Fig. 4 done (含 f 面板:B_surface %.3f -> B %.3f -> q %.1f/%.1f).'
      % (B_SURF, B_DEP, Q_M, Q_C))
