# -*- coding: utf-8 -*-
"""
Extended Data Fig. 2 | Kriging uncertainty of the correction field.

核心结论:校正场在观测密集区可信,在观测稀疏区不确定度显著上升;读者据此判断
网格化场哪些部分由实测锚定、哪些是插值外推。


"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from nature_style_v2 import (set_style, COL, CMAP_U, W2, panel, panel_fig,
                             map_ax, cbar, save, FS_TICK, FS_LABEL)

set_style(base=7)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figs_v2')
os.makedirs(OUT, exist_ok=True)

d = np.load(os.path.join(HERE, '..', '..', 'kriged_heatflow_05deg_global.npz'))
std_dq, lons_g, lats_g = d['std_dq'], d['lons'], d['lats']
LON, LAT = np.meshgrid(lons_g, lats_g)
valid = np.isfinite(d['krige_qm'])

land_grid = rasterize(
    [(g, 1) for g in cfeature.LAND.geometries()],
    out_shape=std_dq.shape, transform=from_origin(-180, 90, 0.5, 0.5),
    fill=0, all_touched=True).astype(bool)[::-1]
land_mask = valid & land_grid

# 直接观测格点:格点内含至少一个实测点(与图注定义一致)
p = np.load(os.path.join(HERE, '..', '..', 'rebuild_points.npz'))
ix = np.clip(((p['lon'] + 180) / 0.5).astype(int), 0, len(lons_g) - 1)
iy = np.clip(((p['lat'] + 90) / 0.5).astype(int), 0, len(lats_g) - 1)
has_obs = np.zeros_like(std_dq, dtype=bool)
has_obs[iy, ix] = True

_R_EARTH = 6371.0088


def _xyz(lo, la):
    lo, la = np.radians(lo), np.radians(la)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                     np.sin(la)], axis=-1)


_obs_ij = np.argwhere(has_obs & land_grid)
_tree = cKDTree(_xyz(lons_g[_obs_ij[:, 1]], lats_g[_obs_ij[:, 0]]))
_chord, _ = _tree.query(_xyz(LON.ravel(), LAT.ravel()), workers=-1)
# 单位球上的弦长换成球面大圆距离
dist_km = (2.0 * _R_EARTH * np.arcsin(np.clip(_chord / 2.0, 0, 1))).reshape(LON.shape)

REGIONS = [('N. America', -170, -10, 15, 84), ('S. America', -80, -35, -55, 10),
           ('Europe', -10, 60, 35, 82),       ('Africa', -20, 55, -35, 38),
           ('Asia', 60, 180, 5, 78),          ('Oceania', 95, 180, -50, 5)]

H_MM = 116.0
MAP_W = 0.840
map_h = MAP_W * 183.0 / 2.535 / H_MM
fig = plt.figure(figsize=(W2, H_MM / 25.4))
ax = fig.add_axes([0.055, 0.425, MAP_W, map_h], projection=ccrs.PlateCarree(central_longitude=0))
map_ax(ax, labels=True, graticule=False)
_DB = [0, 50, 100, 150, 200, 300, 450, 700, 1200]
_dcm = LinearSegmentedColormap.from_list('dist', CMAP_U(np.linspace(0.0, 1.0, len(_DB))), N=len(_DB))
_dnorm = BoundaryNorm(_DB, len(_DB), extend='max')
im = ax.pcolormesh(LON, LAT, np.ma.array(dist_km, mask=~land_mask), cmap=_dcm, norm=_dnorm,
                   transform=ccrs.PlateCarree(), rasterized=True, zorder=2, shading='auto')
cax = fig.add_axes([0.902, 0.47, 0.010, 0.44])
cb = fig.colorbar(im, cax=cax, ticks=_DB, extend='max')
cb.set_label('Distance to nearest\nobserved cell (km)', fontsize=FS_LABEL, labelpad=2, linespacing=1.2)
cb.ax.tick_params(labelsize=FS_TICK, length=1.8, width=0.4, pad=1.2)
cb.outline.set_linewidth(0.4)
panel_fig(fig, 'a', 0.012, 0.975)

names, cov, med = [], [], []
for nm, lo, hi, la0, la1 in REGIONS:
    m = land_mask & (LON >= lo) & (LON <= hi) & (LAT >= la0) & (LAT <= la1)
    if m.sum() == 0:
        continue
    names.append(nm)
    cov.append(100 * (m & has_obs).sum() / m.sum())
    med.append(float(np.median(dist_km[m & ~has_obs])))
order = np.argsort(cov)[::-1]
names = [names[i] for i in order]
cov, med = np.array(cov)[order], np.array(med)[order]
y = np.arange(len(names))
g_cov = 100 * (land_mask & has_obs).sum() / land_mask.sum()
g_med = float(np.median(dist_km[land_mask & ~has_obs]))

for k, (vals, glob, lab, fmt, colr, pl) in enumerate((
        (cov, g_cov, 'Cells with a direct\nobservation (%)', '%.1f', CMAP_U(0.72), 'b'),
        (med, g_med, 'Median distance to nearest\nobservation (km)', '%.0f', CMAP_U(0.72), 'c'))):
    axk = fig.add_axes([0.135 + k * 0.475, 0.110, 0.330, 0.212])
    axk.barh(y, vals, 0.62, color=colr, edgecolor='white', lw=0.4, zorder=2)
    axk.set_xlim(0, vals.max() * 1.32)
    _pad = vals.max() * 0.03
    _mm2x = vals.max() * 1.32 / (0.330 * 183.0)   # 每毫米对应的数据单位(轴宽 60.4 mm)
    for yi, v in zip(y, vals):
        x0 = v + _pad
        # 7.2 pt Arial:数字宽约 1.41 mm、小数点约 0.71 mm,另加白底两侧各 0.2 mm
        _w = (sum(0.71 if ch == '.' else 1.41 for ch in fmt % v) + 0.4) * _mm2x
        if x0 < glob + _pad * 0.5 and x0 + _w > glob - _pad * 0.5:
            x0 = glob + _pad
        axk.text(x0, yi, fmt % v, va='center', ha='left', fontsize=FS_TICK, color=COL['ink'],
                 bbox=dict(fc='white', ec='none', pad=0.6), zorder=4)
    axk.axvline(glob, color=COL['ink'], lw=0.7, ls=(0, (3, 2)), zorder=1)
    # 全球值标签放在虚线右侧、带白底,不再被虚线从中穿过
    axk.annotate('global ' + fmt % glob, xy=(glob, -0.75), xytext=(3, 0), textcoords='offset points',
                 ha='left', va='bottom', fontsize=FS_TICK, color=COL['ink'],
                 bbox=dict(fc='white', ec='none', pad=0.6), zorder=4)
    axk.set_yticks(y); axk.set_yticklabels(names, fontsize=FS_TICK)
    axk.invert_yaxis()
    axk.set_ylim(len(names) - 0.4, -1.3)
    axk.set_xlabel(lab, fontsize=FS_LABEL, labelpad=2, linespacing=1.2)
    axk.tick_params(axis='x', labelsize=FS_TICK)
    panel(axk, pl, x=-0.30, y=1.02)

save(fig, os.path.join(OUT, 'ExtendedDataFig5'))
import json as _json
_json.dump({'median_dist_km_global': float(np.median(dist_km[land_mask & ~has_obs])),
            'obs_cell_frac_pct_global': float(100 * (land_mask & has_obs).sum() / land_mask.sum()),
            'per_region': {nm: {'obs_cell_frac_pct': float(c),
                                'median_dist_km': float(m)}
                           for nm, c, m in zip(names, cov, med)}},
           open(os.path.join(OUT, 'figED2_support.json'), 'w', encoding='utf-8'),
           indent=2, ensure_ascii=False)

print('Extended Data Fig. 5 done.  含观测格点 {:.1f}%,插值格点到最近观测的中位距离 {:.0f} km'
      .format(100 * (land_mask & has_obs).sum() / land_mask.sum(),
              float(np.median(dist_km[land_mask & ~has_obs]))))
