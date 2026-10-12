# -*- coding: utf-8 -*-
"""Fig. 2 | 算子对有限元解的复现精度（原 Fig. 2 与 Fig. 3 合并）。


合并后主图只讲独立集 —— 正文明说"下文的具体数值取自完全独立的500瓦片集"，
那主图就该是它。空间留出集的同类统计降为附图，供"相同趋势在空间留出集上
同样存在"这一句引用。

顺带修掉图号出场顺序：原稿正文里 图2a 出现在 图1a 之前，而 图3b,c 又出现在
图3a 之前。合并后 a→f 的出场顺序与正文一致。

布局：2 列 × 3 行。a/b 是密度图必须等比，故取近似方形；下面四个是统计图。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.ticker import NullFormatter
from nature_style_v2 import (set_style, COL, W2, panel_fig, save, relief_colors,
                             DENSITY_RAMP, FS_TICK, FS_LABEL)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from step3_train_large import FNO3D                              # noqa: E402

set_style(base=7)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figs_v2')
ROOT = os.path.join(HERE, '..', '..')
os.makedirs(OUT, exist_ok=True)

CLASS_COLS = relief_colors(5)
EDGES = [0, 50, 100, 200, 500, 1e9]
NAMES = ['<50', '50–100', '100–200', '200–500', '>500']


def compute_r2(p, y):
    ssr = float(np.sum((y - p) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1 - ssr / sst if sst > 1e-12 else np.nan


# ── 空间留出集：需要跑一次推理（只用于 a 面板）───────────────────
ck = torch.load(os.path.join(ROOT, 'fno3d_large_model.pt'),
                map_location='cuda', weights_only=False)
model = FNO3D(modes=24, width=96, H_out=41, W_out=41, n_layers=6).cuda().eval()
model.load_state_dict(ck['state'])
from step3_train_large import D4Average  # noqa: E402
model = D4Average(model).eval()
print('推理:8 朝向平均(D4Average)')
X_mean, X_std, Y_mean, Y_std = ck['X_mean'], ck['X_std'], ck['Y_mean'], ck['Y_std']

d = np.load(os.path.join(ROOT, 'global_fem_dataset.npz'))
X = (d['X'] - X_mean) / X_std
Y = (np.clip(d['Y'], 0.05, 5.0) - Y_mean) / Y_std
labels = KMeans(n_clusters=12, random_state=42, n_init='auto').fit_predict(
    np.column_stack([d['lon'], d['lat']]))
i_te = np.where(np.isin(labels, [0, 4, 8]))[0]

Xt = torch.tensor(X[i_te], dtype=torch.float32).cuda()
preds = []
for b in range(0, len(i_te), 8):
    with torch.no_grad():
        preds.append(model(Xt[b:b + 8]).cpu().numpy())
pred_norm = np.concatenate(preds)
pred_phys = pred_norm * Y_std + Y_mean
true_phys = Y[i_te] * Y_std + Y_mean
r2_spatial = compute_r2(pred_norm, Y[i_te])
mae_spatial = float(np.mean(np.abs(true_phys - pred_phys)))
reliefs_sp = d['relief'][i_te]

# ── 独立 500 瓦片集：c–f 全部用它 ──────────────────────────────
val = np.load(os.path.join(ROOT, 'validation_independent_500.npz'))
B_fem, B_fno = val['B_fem'], val['B_fno']
reliefs, per_tile_r2 = val['reliefs'], val['per_tile_r2']

r2_t, mae_t, n_t = [], [], []
for i in range(5):
    m = (reliefs >= EDGES[i]) & (reliefs < EDGES[i + 1])
    n_t.append(int(m.sum()))
    if m.sum():
        f, p = B_fem[m].ravel(), B_fno[m].ravel()
        r2_t.append(compute_r2(p, f))
        mae_t.append(float(np.mean(np.abs(f - p))))
    else:
        r2_t.append(np.nan)
        mae_t.append(np.nan)

fig = plt.figure(figsize=(W2, 118 / 25.4))
lim = (0.45, 1.60)
xb = np.arange(5)
COLX = [0.075 - 0.165 * 0.235, 0.405 - 0.165 * 0.235, 0.745 - 0.165 * 0.235]
Y_TOP, Y_BOT = 0.570 + 0.350 * 1.025, 0.125 + 0.325 * 1.03

# ─────────────── a  空间留出集：预测 vs 参考 ───────────────
ax = fig.add_axes([0.075, 0.570, 0.235, 0.350])
hb = ax.hexbin(true_phys.ravel(), pred_phys.ravel(), gridsize=58, bins='log',
               mincnt=1, cmap=DENSITY_RAMP, extent=(*lim, *lim), linewidths=0,
               rasterized=True, zorder=2)
ax.plot(lim, lim, color=COL['ink'], lw=0.7, ls=(0, (4, 2)), zorder=3)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
ax.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
ax.set_axisbelow(True)
for _s in ('left', 'bottom'):
    ax.spines[_s].set_position(('outward', 3))
ax.set_xlabel('Finite-element reference $B$', labelpad=2)
ax.set_ylabel('Operator-predicted $B$', labelpad=2)
ax.set_title('Spatially held-out test set', fontsize=7.2, pad=4, color=COL['ink'])
# 统计量放面板内左上：密度点云沿 1:1 线分布，那个角本来就是空的
ax.text(0.035, 0.965,
        f'$R^2$ = {r2_spatial:.3f}\nMAE = {mae_spatial:.3f}\n'
        f'$n$ = {len(reliefs_sp):,} tiles',
        transform=ax.transAxes, ha='left', va='top', fontsize=7.2,
        linespacing=1.35, color=COL['ink'])
panel_fig(fig, 'a', COLX[0], Y_TOP)

# ─────────────── b  独立集：预测 vs 参考 ───────────────
ax = fig.add_axes([0.405, 0.570, 0.235, 0.350])
_Bf, _Bn = B_fem.ravel(), B_fno.ravel()
ax.hexbin(_Bf, _Bn, gridsize=58, bins='log', mincnt=1, cmap=DENSITY_RAMP,
          extent=(*lim, *lim), linewidths=0, rasterized=True, zorder=2)
ax.plot(lim, lim, color=COL['ink'], lw=0.7, ls=(0, (4, 2)), zorder=3)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
ax.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
ax.set_axisbelow(True)
for _s in ('left', 'bottom'):
    ax.spines[_s].set_position(('outward', 3))
ax.set_xlabel('Finite-element reference $B$', labelpad=2)
ax.set_ylabel('Operator-predicted $B$', labelpad=2)
_r2i = float(val['r2_total']); _maei = float(np.mean(np.abs(_Bf - _Bn)))
ax.set_title('Independent 500-tile set', fontsize=7.2, pad=4, color=COL['ink'])
ax.text(0.035, 0.965, f'$R^2$ = {_r2i:.3f}\nMAE = {_maei:.3f}\n$n$ = 500 tiles',
        transform=ax.transAxes, ha='left', va='top', fontsize=7.2,
        linespacing=1.35, color=COL['ink'])
panel_fig(fig, 'b', COLX[1], Y_TOP)

# a 与 b 是同一种密度图，共用一条色标 —— 各带一条属于重复信息
cax = ax.inset_axes([0.80, 0.07, 0.045, 0.36])   # 放进 b 的右下空白处,避免挤占 c
cb = fig.colorbar(hb, cax=cax)
cb.set_label('Pixels per bin', fontsize=FS_LABEL, labelpad=2)
cb.ax.tick_params(labelsize=FS_TICK, length=1.6, width=0.4, pad=1.2)
cb.set_ticks([1, 1e2, 1e4, 1e6]); cb.set_ticklabels(['1', '10²', '10⁴', '10⁶'])
cb.ax.minorticks_off()
cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor('#888888')

# ─────────────── c  B 分布对比 ───────────────
ax = fig.add_axes([0.745, 0.570, 0.235, 0.350])
bins_h = np.linspace(0.5, 1.6, 76)
ax.hist(_Bf, bins=bins_h, color=COL['ref'], alpha=0.55,
        label='Finite element (reference)', lw=0, zorder=2)
ax.hist(_Bn, bins=bins_h, histtype='step', color=COL['hero'],
        lw=1.0, label='Neural operator (predicted)', zorder=3)
try:
    from scipy.stats import wasserstein_distance
    wd = wasserstein_distance(_Bf, _Bn)
    ax.text(0.98, 0.64, f'Wasserstein\ndistance\n= {wd:.5f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=FS_TICK, linespacing=1.3, color=COL['ink'])
except Exception:
    pass
ax.set_title('$B$ distribution, independent set', fontsize=7.2, pad=4, color=COL['ink'])
ax.set_xlabel('$B$', labelpad=2)
ax.set_ylabel('Number of pixels', labelpad=2)
ax.set_yscale('log')
ax.set_ylim(0.5, 3e7)
ax.legend(fontsize=FS_TICK, loc='upper left', handlelength=1.1,
          handletextpad=0.45, labelspacing=0.28, borderpad=0.2)
panel_fig(fig, 'c', COLX[2], Y_TOP)

# ─────────────── d  按起伏分档的 R² ───────────────
ax = fig.add_axes([0.075, 0.125, 0.235, 0.325])
ax.bar(xb, r2_t, 0.62, color=CLASS_COLS, edgecolor='white', lw=0.4, zorder=2)
for xi, v, n in zip(xb, r2_t, n_t):
    ax.text(xi, v + 0.02, f'{v:.3f}\nn={n}', ha='center', va='bottom', fontsize=FS_TICK,
            linespacing=1.15, color=COL['ink'])
ax.set_xticks(xb); ax.set_xticklabels(NAMES, fontsize=FS_TICK, rotation=25, ha='right')
ax.set_ylim(0, max(r2_t) * 1.26)
ax.set_ylabel('Pixel-wise $R^2$', labelpad=2)
ax.set_xlabel('Tile relief (m)', labelpad=2)
panel_fig(fig, 'd', COLX[0], Y_BOT)

# ─────────────── e  逐瓦片 R² vs 起伏 ───────────────
ax = fig.add_axes([0.405, 0.125, 0.235, 0.325])
for i in range(5):
    m = (reliefs >= EDGES[i]) & (reliefs < EDGES[i + 1])
    if m.sum():
        ax.scatter(reliefs[m], per_tile_r2[m], s=4.5, alpha=0.55,
                   c=[CLASS_COLS[i]], edgecolors='none', label=NAMES[i], zorder=2)
ax.set_xscale('log')
ax.set_xticks([30, 100, 300, 1000, 3000])
ax.set_xticklabels(['30', '100', '300', '1,000', '3,000'])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel('Tile relief (m)', labelpad=2)
ax.set_ylabel('Per-tile $R^2$', labelpad=2)
ax.set_ylim(0.20, 1.03)
leg = ax.legend(fontsize=FS_TICK, loc='lower right', markerscale=2.0,
                handletextpad=0.35, labelspacing=0.22, borderpad=0.2,
                title='Relief (m)', title_fontsize=FS_TICK)
leg._legend_box.align = 'left'
panel_fig(fig, 'e', COLX[1], Y_BOT)

# ─────────────── f  按起伏分档的 MAE ───────────────
ax = fig.add_axes([0.745, 0.125, 0.235, 0.325])
ax.bar(xb, mae_t, 0.62, color=CLASS_COLS, edgecolor='white', lw=0.4, zorder=2)
for xi, v in zip(xb, mae_t):
    ax.text(xi, v + max(mae_t) * 0.035, f'{v:.4f}', ha='center', fontsize=FS_TICK,
            color=COL['ink'])
ax.set_xticks(xb); ax.set_xticklabels(NAMES, fontsize=FS_TICK, rotation=25, ha='right')
ax.set_ylim(0, max(mae_t) * 1.28)
ax.set_ylabel('Mean absolute error in $B$', labelpad=2)
ax.set_xlabel('Tile relief (m)', labelpad=2)
panel_fig(fig, 'f', COLX[2], Y_BOT)

save(fig, os.path.join(OUT, 'Fig2'))
print(f'Fig. 2 done.  spatial R2={r2_spatial:.4f}  independent R2={_r2i:.4f}')
