# -*- coding: utf-8 -*-
"""
Fig. 4 | The global surface transfer factor field.


break-even 与加速比的计算**仍保留在本脚本**,并照旧写出 figs_v2/fig6_breakeven.json ——
emit_numbers.py 读这个文件,正文引用的盈亏平衡瓦片数与加速比都从这里来,数值不变。

"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json as _json0
import numpy as np
from matplotlib.colors import SymLogNorm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from nature_style_v2 import (set_style, CMAP_B, W2, RECT_AR, MAP_EXTENT, COL, FS_TICK, FS_LABEL, map_ax, cbar, save,
                             panel_fig)
from matplotlib.colors import ListedColormap, BoundaryNorm

set_style(base=7)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figs_v2')
ROOT = os.path.join(HERE, '..', '..')
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(ROOT, 'fem_speed_benchmark.json'), encoding='utf-8') as _fh:
    _fb = _json0.load(_fh)
with open(os.path.join(ROOT, 'fno_speed_benchmark.json'), encoding='utf-8') as _fh:
    _nb = _json0.load(_fh)
FEM_MS = _fb['fem_total_ms']
N_ORIENT = 8
FNO_MS = _nb['batch8_call_ms']
FNO_BATCH_MS = N_ORIENT * _nb['fno_batch_ms']
SPEEDUP = FEM_MS / FNO_MS
# 规模档位一律从产物读,不再用 46,000 这个估算值。
with open(os.path.join(ROOT, 'global_B_field_05deg.json'), encoding='utf-8') as _fh:
    _gb = _json0.load(_fh)
N_LAND_05 = int(_gb['n_solved'])          # 全球陆地 0.5° 格点(已算)
N_LAND_01 = N_LAND_05 * 25                # 同一掩膜细分到 0.1°

MAP_L_MM, MAP_W_MM, MAP_TOP_MM = 8.5, 172.5, 4.0
MAP_H_MM = MAP_W_MM / RECT_AR
ROW_MM = MAP_H_MM + 7.0 + 2.4 + 10.0      # 一行 = 地图 + 色标间距 + 色标 + 刻度与行距
MAP_TOP_B_MM = MAP_TOP_MM + ROW_MM
FIG_H_MM = MAP_TOP_B_MM + ROW_MM - 3.0
fig = plt.figure(figsize=(W2, FIG_H_MM / 25.4))

import json as _json
with open(os.path.join(ROOT, 'fno3d_large_metrics.json'), encoding='utf-8') as _fh:
    _m = _json.load(_fh)
N_TILES_TRAIN = int(_m['n_total'])
TRAIN_HOURS = float(_m['train_time_s']) / 3600.0
N_SITES = int(np.load(os.path.join(ROOT, 'rebuild_points.npz'))['B'].shape[0])
print('[fig4] 训练瓦片 %s | 训练 %.2f h | 测点 %s | 陆地 0.5° %s'
      % (format(N_TILES_TRAIN, ','), TRAIN_HOURS, format(N_SITES, ','),
         format(N_LAND_05, ',')))

onetime = N_TILES_TRAIN * FEM_MS / 1000 + TRAIN_HOURS * 3600
be = onetime / ((FEM_MS - FNO_BATCH_MS) / 1000)

_g = np.load(os.path.join(ROOT, 'global_B_field_05deg.npz'))
_GB, _glat, _glon = _g['B'], _g['lats'], _g['lons']

_H = FIG_H_MM
ax = fig.add_axes([MAP_L_MM / 183.0, 1 - (MAP_TOP_MM + MAP_H_MM) / _H,
                   MAP_W_MM / 183.0, MAP_H_MM / _H],
                  projection=ccrs.PlateCarree(central_longitude=0))
map_ax(ax, labels=True, graticule=False)
B_EDGES = [0.80, 0.90, 0.95, 0.98, 1.02, 1.05, 1.10, 1.20]
NEUTRAL = '#E3E3E3'
_cols = [CMAP_B(0.12), CMAP_B(0.24), CMAP_B(0.36), NEUTRAL, CMAP_B(0.64), CMAP_B(0.76), CMAP_B(0.88)]
UNDER, OVER = CMAP_B(0.02), CMAP_B(0.98)
_cm = ListedColormap(_cols)
_cm.set_under(UNDER); _cm.set_over(OVER); _cm.set_bad(alpha=0.0)
_norm = BoundaryNorm(B_EDGES, _cm.N)
_LON, _LAT = np.meshgrid(_glon, _glat)
sc = ax.pcolormesh(_LON, _LAT, np.ma.masked_invalid(_GB), cmap=_cm, norm=_norm,
                   transform=ccrs.PlateCarree(), shading='nearest', rasterized=True, zorder=2)

# 色标:离散分档,两端箭头表示 < 0.80 与 > 1.20;标题放在色标左侧(参考图版式),省出一行高度
CB_W_MM, CB_H_MM = 104.0, 2.4
CB_L_MM = MAP_L_MM + MAP_W_MM / 2 - CB_W_MM / 2 + 14.0
CB_Y_MM = _H - (MAP_TOP_MM + MAP_H_MM) - 7.0 - CB_H_MM
_cb = cbar(fig, sc, [CB_L_MM / 183.0, CB_Y_MM / _H, CB_W_MM / 183.0, CB_H_MM / _H], '',
           ticks=B_EDGES, extend='both')
_cb.set_ticklabels(['%.2f' % v for v in B_EDGES])
fig.text((CB_L_MM - 3.0) / 183.0, (CB_Y_MM + CB_H_MM / 2) / _H, 'Surface transfer factor $B$',
         ha='right', va='center', fontsize=FS_LABEL, color=COL['ink'])

# 环形图:各档的陆地面积占比(按纬度余弦加权),中心为两侧偏离超过 2% 的面积占比
_w = np.broadcast_to(np.cos(np.radians(_glat))[:, None], _GB.shape)
_ok = np.isfinite(_GB)
_b, _ww = _GB[_ok], _w[_ok]
_bins = [-np.inf] + B_EDGES + [np.inf]
SHARE = np.array([_ww[(_b >= lo) & (_b < hi)].sum() for lo, hi in zip(_bins[:-1], _bins[1:])]) / _ww.sum()
D_MM = 25.0
_cx = MAP_L_MM + (-136.0 + 180.0) / 360.0 * MAP_W_MM
_cy = MAP_TOP_MM + (MAP_EXTENT[3] - (-24.0)) / (MAP_EXTENT[3] - MAP_EXTENT[2]) * MAP_H_MM
axd = fig.add_axes([(_cx - D_MM / 2) / 183.0, 1 - (_cy + D_MM / 2) / _H, D_MM / 183.0, D_MM / _H])
axd.pie(SHARE, colors=[UNDER] + _cols + [OVER], startangle=90, counterclock=False,
        wedgeprops=dict(width=0.30, edgecolor='white', linewidth=0.5))
axd.set_aspect('equal')
_lo, _hi = 100 * SHARE[:4].sum(), 100 * SHARE[5:].sum()
axd.text(0, 0.24, 'Land area', ha='center', va='center', fontsize=FS_TICK - 0.6, color=COL['ink'])
axd.text(-0.07, -0.02, '%d%%' % round(_lo), ha='right', va='center', fontsize=FS_TICK + 0.4,
         color=CMAP_B(0.12), fontweight='bold')
axd.text(0, -0.02, '|', ha='center', va='center', fontsize=FS_TICK + 0.4, color='#9A9A9A')
axd.text(0.07, -0.02, '%d%%' % round(_hi), ha='left', va='center', fontsize=FS_TICK + 0.4,
         color=CMAP_B(0.88), fontweight='bold')
axd.text(0, -0.30, '$B$ < 0.98 | > 1.02', ha='center', va='center', fontsize=FS_TICK - 1.2, color='#6A6A6A')
print('[fig4] 陆地面积占比:B < 0.98 %.1f%% | 0.98–1.02 %.1f%% | B > 1.02 %.1f%%' % (_lo, 100 * SHARE[4], _hi))

pts = np.load(os.path.join(ROOT, 'rebuild_points.npz'))
p_lon, p_lat = pts['lon'], pts['lat']
p_rel = 100.0 * (pts['q_corrected'] - pts['q_measured']) / pts['q_measured']
DQ_EDGES = [-20, -10, -5, -2, 2, 5, 10, 20]
NEUTRAL_PT = '#CDCDCD'
_colq = [CMAP_B(0.88), CMAP_B(0.76), CMAP_B(0.64), NEUTRAL_PT, CMAP_B(0.36), CMAP_B(0.24), CMAP_B(0.12)]
_cmq = ListedColormap(_colq)
_cmq.set_under(OVER); _cmq.set_over(UNDER)
_nq = BoundaryNorm(DQ_EDGES, _cmq.N)
axb = fig.add_axes([MAP_L_MM / 183.0, 1 - (MAP_TOP_B_MM + MAP_H_MM) / _H,
                    MAP_W_MM / 183.0, MAP_H_MM / _H],
                   projection=ccrs.PlateCarree(central_longitude=0))
map_ax(axb, labels=True, graticule=False)
_neu = np.abs(p_rel) < 2
axb.scatter(p_lon[_neu], p_lat[_neu], s=0.9, c=NEUTRAL_PT, transform=ccrs.PlateCarree(), linewidths=0.0,
            rasterized=True, zorder=3)
_ord = np.argsort(np.abs(p_rel))
_ord = _ord[~_neu[_ord]]
scb = axb.scatter(p_lon[_ord], p_lat[_ord], s=2.2, c=p_rel[_ord], cmap=_cmq, norm=_nq,
                  transform=ccrs.PlateCarree(), linewidths=0.0, rasterized=True, zorder=4)
CBB_Y_MM = _H - (MAP_TOP_B_MM + MAP_H_MM) - 7.0 - CB_H_MM
_cbb = cbar(fig, scb, [CB_L_MM / 183.0, CBB_Y_MM / _H, CB_W_MM / 183.0, CB_H_MM / _H], '',
            ticks=DQ_EDGES, extend='both')
_cbb.set_ticklabels([('%+d' % v).replace('-', '−') for v in DQ_EDGES])
_cbb.ax.invert_xaxis()                                 # 左蓝(+)右红(−),与 a 的色标同向
fig.text((CB_L_MM - 3.0) / 183.0, (CBB_Y_MM + CB_H_MM / 2) / _H,
         'Relative correction $\\Delta q\\,/\\,q_{\\mathrm{obs}}$ (%)',
         ha='right', va='center', fontsize=FS_LABEL, color=COL['ink'])

# 环形图:各档测点占比,自蓝(热流被调高)至红(被调低),与 a 的环形图同序
_binsq = [np.inf] + DQ_EDGES[::-1] + [-np.inf]
SHARE_Q = np.array([np.mean((p_rel < hi) & (p_rel >= lo)) for hi, lo in zip(_binsq[:-1], _binsq[1:])])
_cyb = MAP_TOP_B_MM + (MAP_EXTENT[3] - (-24.0)) / (MAP_EXTENT[3] - MAP_EXTENT[2]) * MAP_H_MM
axdb = fig.add_axes([(_cx - D_MM / 2) / 183.0, 1 - (_cyb + D_MM / 2) / _H, D_MM / 183.0, D_MM / _H])
axdb.pie(SHARE_Q, colors=[UNDER] + _colq[::-1] + [OVER], startangle=90, counterclock=False,
         wedgeprops=dict(width=0.30, edgecolor='white', linewidth=0.5))
axdb.set_aspect('equal')
_up, _dn = 100 * np.mean(p_rel > 2), 100 * np.mean(p_rel < -2)
axdb.text(0, 0.24, 'Sites', ha='center', va='center', fontsize=FS_TICK - 0.6, color=COL['ink'])
axdb.text(-0.07, -0.02, '%d%%' % round(_up), ha='right', va='center', fontsize=FS_TICK + 0.4,
          color=CMAP_B(0.12), fontweight='bold')
axdb.text(0, -0.02, '|', ha='center', va='center', fontsize=FS_TICK + 0.4, color='#9A9A9A')
axdb.text(0.07, -0.02, '%d%%' % round(_dn), ha='left', va='center', fontsize=FS_TICK + 0.4,
          color=CMAP_B(0.88), fontweight='bold')
axdb.text(0, -0.30, '> +2% | < −2%', ha='center', va='center', fontsize=FS_TICK - 1.2, color='#6A6A6A')
panel_fig(fig, 'a', 0.004, 1 - (MAP_TOP_MM + 1.0) / _H)
panel_fig(fig, 'b', 0.004, 1 - (MAP_TOP_B_MM + 1.0) / _H)
FIG4B = dict(n_sites=int(len(p_rel)), pct_raised_gt2=float(_up), pct_lowered_gt2=float(_dn),
             pct_neutral=float(100 * _neu.mean()), n_gt20=int((p_rel > 20).sum()), n_lt20=int((p_rel < -20).sum()))
print('[fig4b] 测点 %s | ±2%% 以内 %.1f%% | >+2%% %.1f%% | <−2%% %.1f%% | >+20%% %d 个 | <−20%% %d 个'
      % (format(len(p_rel), ','), FIG4B['pct_neutral'], _up, _dn, FIG4B['n_gt20'], FIG4B['n_lt20']))

save(fig, os.path.join(OUT, 'Fig4'))
import json as _json
_json.dump({'breakeven_tiles': float(be),
            'speedup': float(SPEEDUP),
            'fem_ms_per_tile': FEM_MS,
            'fno_ms_per_tile': FNO_MS,
            'fno_ms_per_tile_breakeven': FNO_BATCH_MS,
            'n_tiles_train': int(N_TILES_TRAIN),
            'train_hours': float(TRAIN_HOURS),
            'fig4b': FIG4B},
           open(os.path.join(OUT, 'fig6_breakeven.json'), 'w', encoding='utf-8'),
           indent=2)
print(f'Fig. 4 done.  speedup x{SPEEDUP:.0f}, break-even {be:,.0f}')
