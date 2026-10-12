# -*- coding: utf-8 -*-
"""Extended Data Fig. 6 | 空间留出集上的分档精度。


三个面板与正文 Fig. 2 的 d/e/f 一一对应,只是换成空间留出集的数据,
因此配色、分档与坐标范围都与之保持一致 —— 两图并看才有意义。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from step3_train_large import FNO3D, compute_r2
from nature_style_v2 import (set_style, COL, W2, panel, save, relief_colors,
                             DENSITY_RAMP)

set_style(base=7)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figs_v2')
ROOT = os.path.join(HERE, '..', '..')
os.makedirs(OUT, exist_ok=True)

CLASS_COLS = relief_colors(5)     # 与 Fig. 4 / ExtendedDataFig. 1 同一色阶

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

km = KMeans(n_clusters=12, random_state=42, n_init='auto')
labels = km.fit_predict(np.column_stack([d['lon'], d['lat']]))
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
reliefs = d['relief'][i_te]
per_tile_r2 = np.array([compute_r2(pred_norm[i:i + 1], Y[i_te][i:i + 1])
                        for i in range(len(i_te))])

EDGES = [0, 50, 100, 200, 500, 1e9]
NAMES = ['<50', '50–\n100', '100–\n200', '200–\n500', '>500']
# 柱状图 x 轴要换行以免标签重叠,图例里却不该断行,另备一套
LEG_NAMES = ['<50', '50–100', '100–200', '200–500', '>500']

val = np.load(os.path.join(ROOT, 'validation_independent_500.npz'))

fig = plt.figure(figsize=(W2, 72 / 25.4))


def relief_ticks(ax):
    """2026-10-03:起伏刻度写法与图 1b、图 6、图 9a 统一(30/100/300/1,000/3,000),原先是 10²、10³。"""
    from matplotlib.ticker import NullFormatter
    ax.set_xticks([30, 100, 300, 1000, 3000])
    ax.set_xticklabels(['30', '100', '300', '1,000', '3,000'])
    ax.xaxis.set_minor_formatter(NullFormatter())

# ─────────────── c  按起伏分档的 R² ───────────────
ax = fig.add_axes([0.075, 0.235, 0.250, 0.620])
r2s, ns = [], []
for i in range(5):
    m = (reliefs >= EDGES[i]) & (reliefs < EDGES[i + 1])
    ns.append(int(m.sum()))
    r2s.append(compute_r2(pred_norm[m], Y[i_te][m]) if m.sum() else np.nan)
xb = np.arange(5)
ax.bar(xb, r2s, 0.62, color=CLASS_COLS, edgecolor='white', lw=0.4, zorder=2)
for xi, v, n in zip(xb, r2s, ns):
    ax.text(xi, v + 0.02, f'{v:.3f}\nn={n}', ha='center', va='bottom', fontsize=7.2,
            linespacing=1.15, color=COL['ink'])
ax.set_xticks(xb)
ax.set_xticklabels(LEG_NAMES, fontsize=7.2, rotation=30, ha='right')
ax.set_ylim(0, max(r2s) * 1.22)
ax.set_ylabel('Pixel-wise $R^2$', labelpad=2)
ax.set_xlabel('Tile relief (m)', labelpad=2)
panel(ax, 'a', x=-0.185, y=1.03)

# ─────────────── d  逐瓦片 R² vs 起伏 ───────────────
ax = fig.add_axes([0.400, 0.235, 0.250, 0.620])
for i in range(5):
    m = (reliefs >= EDGES[i]) & (reliefs < EDGES[i + 1])
    if m.sum():
        ax.scatter(reliefs[m], per_tile_r2[m], s=3.2, alpha=0.45,
                   c=[CLASS_COLS[i]], edgecolors='none', rasterized=True,
                   label=LEG_NAMES[i], zorder=2)
ax.set_xscale('log')
relief_ticks(ax)
ax.set_xlabel('Tile relief (m)', labelpad=2)
ax.set_ylabel('Per-tile $R^2$', labelpad=2)
ax.set_ylim(0.20, 1.03)
ax.set_axisbelow(True)
leg = ax.legend(fontsize=7.2, loc='lower right', markerscale=2.4,
                handletextpad=0.35, labelspacing=0.22, borderpad=0.25,
                title='Relief (m)', title_fontsize=7.2,
                framealpha=0.92, facecolor='white', edgecolor='none')
leg.set_zorder(6)
leg._legend_box.align = 'left'
panel(ax, 'b', x=-0.185, y=1.03)

# ───────── e  逐瓦片平均绝对误差 vs 起伏 ─────────
ax = fig.add_axes([0.725, 0.235, 0.250, 0.620])
per_tile_mae = np.abs(pred_phys - true_phys).mean(axis=(1, 2, 3))
for i in range(5):
    m = (reliefs >= EDGES[i]) & (reliefs < EDGES[i + 1])
    if m.sum():
        ax.scatter(reliefs[m], per_tile_mae[m], s=3.2, alpha=0.45,
                   c=[CLASS_COLS[i]], edgecolors='none', rasterized=True,
                   label=NAMES[i].replace(chr(10), ''), zorder=2)
ax.set_xscale('log'); ax.set_yscale('log')
relief_ticks(ax)
ax.set_axisbelow(True)
ax.set_xlabel('Tile relief (m)', labelpad=2)
ax.set_ylabel('Per-tile mean absolute error in $B$', labelpad=2)
# e 与 d 共用同一套起伏分档配色,图例只在 d 上给一份 —— 同一份图例
# 挂两遍是"业余感"清单里的常见项(图注注明两图共用配色)。
panel(ax, 'c', x=-0.185, y=1.03)


# 分档数值落盘 —— 正文 3.1 说「相同趋势在空间留出集上同样存在」,
# 没有产物就只能停在这句虚话上。这里把 a 面板的分档 R² 与瓦片数写出来。
import json as _json
_json.dump({'edges_m': EDGES[:-1], 'names': LEG_NAMES,
            'r2_by_relief': [float(v) for v in r2s],
            'n_by_relief': ns,
            'r2_overall': float(r2_spatial)},
           open(os.path.join(ROOT, 'spatial_split_relief_bins.json'), 'w',
                encoding='utf-8'), ensure_ascii=False, indent=2)
print('  spatial_split_relief_bins.json 已写出')

save(fig, os.path.join(OUT, 'ExtendedDataFig6'))
print('Extended Data Fig. 6 done.')
