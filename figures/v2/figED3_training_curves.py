# -*- coding: utf-8 -*-
"""
Extended Data Fig. 3 | Training convergence diagnostics.

核心结论:训练与验证损失同步下降并稳定收敛,验证 R² 在早停点前已进入平台,
说明报告的精度不是训练不足或过拟合某一时刻的偶然结果。

"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from nature_style_v2 import set_style, COL, W2, panel, save

set_style(base=7)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figs_v2')
ROOT = os.path.join(HERE, '..', '..')
os.makedirs(OUT, exist_ok=True)

import torch
# 训练历史保存在模型 checkpoint 里(training_checkpoint.pt 的 history 为空)
ck = torch.load(os.path.join(ROOT, 'fno3d_large_model.pt'),
                map_location='cpu', weights_only=False)
hist = ck['history']
ep = np.array([h['epoch'] for h in hist])
tr = np.array([h['train_loss'] for h in hist])
va = np.array([h['val_loss'] for h in hist])
r2 = np.array([h['r2'] for h in hist])
lr = np.array([h.get('lr', np.nan) for h in hist])
best = int(ep[np.nanargmax(r2)])

FIG_H_MM = 76.0
AX_B, AX_H = 10.5 / FIG_H_MM, 0.680 * 86 / FIG_H_MM
fig = plt.figure(figsize=(W2, FIG_H_MM / 25.4))

# ─────────────── a  损失曲线 ───────────────
ax = fig.add_axes([0.080, AX_B, 0.355, AX_H])
ax.plot(ep, tr, color=COL['ref'], lw=1.0, label='Training loss', zorder=3)
ax.plot(ep, va, color=COL['hero'], lw=1.0, label='Validation loss', zorder=4)
ax.axvline(best, color=COL['accent'], lw=0.6, ls=(0, (3, 2)), zorder=2)
ax.text(best, ax.get_ylim()[1], f' Best epoch {best}', fontsize=7.2,
        color=COL['accent'], ha='left', va='top')
ax.set_yscale('log')
ax.yaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5]))
ax.yaxis.set_major_formatter(FixedFormatter(['0.02', '0.05', '0.1', '0.2', '0.5']))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel('Epoch', labelpad=2)
ax.set_ylabel('Loss (MSE + correlation term)', labelpad=2)
# 对数纵轴上这条网格线只落得下 10^-1 一条,孤零零一横看着像误画;
# Nature 的 "We avoid" 清单本来也明列 background gridlines。
ax.set_axisbelow(True)
ax.legend(fontsize=7.2, loc='upper right', handlelength=1.2,
          handletextpad=0.5, labelspacing=0.3)
panel(ax, 'a', x=-0.155, y=1.03)

# ─────────────── b  验证 R² 与学习率 ───────────────
ax = fig.add_axes([0.575, AX_B, 0.330, AX_H])
ax.plot(ep, r2, color=COL['hero'], lw=1.0, zorder=4)
ax.axvline(best, color=COL['accent'], lw=0.6, ls=(0, (3, 2)), zorder=2)
ax.scatter([best], [np.nanmax(r2)], s=11, color=COL['accent'], zorder=5,
           edgecolors='white', linewidths=0.4)
ax.annotate(f'$R^2$ = {np.nanmax(r2):.4f}', xy=(best, np.nanmax(r2)),
            xytext=(0.30, 0.68), textcoords='axes fraction',
            fontsize=7.2, color=COL['accent'], ha='left', va='center',
            arrowprops=dict(arrowstyle='-', lw=0.5, color=COL['accent'], shrinkB=3),
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.92))
ax.set_xlabel('Epoch', labelpad=2)
ax.set_ylabel('Validation $R^2$', color=COL['hero'], labelpad=2)
ax.tick_params(axis='y', colors=COL['hero'])
ax.set_ylim(max(0, np.nanmin(r2[r2 > 0]) - 0.02), 1.0)

ax2 = ax.twinx()
ax2.plot(ep, lr, color=COL['ref'], lw=0.8, ls=(0, (5, 1.5, 1, 1.5)), zorder=3)
ax2.set_yscale('log')
ax2.set_ylabel('Learning rate', color=COL['ref'], labelpad=3)
ax2.tick_params(axis='y', colors=COL['ref'])
ax2.spines['right'].set_visible(True)
ax2.spines['right'].set_color(COL['ref'])
ax2.spines['top'].set_visible(False)
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)
panel(ax, 'b', x=-0.155, y=1.03)

save(fig, os.path.join(OUT, 'ExtendedDataFig3'))
print(f'Extended Data Fig. 3 done.  best epoch {best}, R2={np.nanmax(r2):.4f}')
