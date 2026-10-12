# -*- coding: utf-8 -*-
"""
extract_borehole_depths.py -- 为最终测点解析钻孔深度区间

IHFC GHFDB 字段(官方 data description):
  C4 q_top     - 热流所对应区间的顶深 (m)
  C5 q_bottom  - 热流所对应区间的底深 (m)
列索引: P1..P13 = 0..12, C1 = 13, 故 C4 -> 16, C5 -> 17。

同一测点(父记录去重后)可能仍对应多条子记录, 取其区间的并集:
  top = min(q_top), bottom = max(q_bottom)
缺失时留 NaN, 由下游按全库中位数填补。
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

RAW = os.environ.get('IHFC_GHFDB', os.path.join('data', 'IHFC_2024_GHFDB_v.2026.03.txt'))
IDX_LAT, IDX_LON, IDX_QTOP, IDX_QBOT, IDX_TOPOFLAG = 3, 4, 16, 17, 27


def fnum(s):
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return np.nan


def main():
    d = np.load('rebuild_points.npz')
    lon, lat = d['lon'], d['lat']
    key_of = {}
    for i, (lo, la) in enumerate(zip(np.round(lon, 6), np.round(lat, 6))):
        key_of.setdefault((lo, la), []).append(i)

    n = len(lon)
    top = np.full(n, np.nan)
    bot = np.full(n, np.nan)
    topo_corrected = np.zeros(n, dtype=bool)

    matched = 0
    with open(RAW, encoding='latin-1') as f:
        for line in f.readlines()[11:]:
            p = line.rstrip('\n').split('\t')
            if len(p) <= IDX_TOPOFLAG:
                continue
            la, lo = fnum(p[IDX_LAT]), fnum(p[IDX_LON])
            if not (np.isfinite(la) and np.isfinite(lo)):
                continue
            idxs = key_of.get((round(lo, 6), round(la, 6)))
            if idxs is None:
                continue
            matched += 1
            t, b = fnum(p[IDX_QTOP]), fnum(p[IDX_QBOT])
            flagged = p[IDX_TOPOFLAG].strip() == '[Present and corrected]'
            for i in idxs:
                if np.isfinite(t):
                    top[i] = t if not np.isfinite(top[i]) else min(top[i], t)
                if np.isfinite(b):
                    bot[i] = b if not np.isfinite(bot[i]) else max(bot[i], b)
                if flagged:
                    topo_corrected[i] = True

    has_bot = np.isfinite(bot)
    print(f'测点 {n:,}, 匹配原始记录 {matched:,}')
    print(f'有 q_bottom 的测点: {has_bot.sum():,} ({100*has_bot.mean():.1f}%)')
    med = float(np.nanmedian(bot))
    print(f'q_bottom 中位数 {med:.0f} m; 25/75 分位 '
          f'{np.nanpercentile(bot,25):.0f}/{np.nanpercentile(bot,75):.0f} m')
    print(f'标记为已做地形校正的测点: {topo_corrected.sum():,} ({100*topo_corrected.mean():.1f}%)')

    # 有效深度: 优先 q_bottom; 缺失用中位数填补
    depth_eff = np.where(has_bot, bot, med)
    # 顶深默认为 0(绝大多数陆地钻孔自地表起算)
    top_eff = np.where(np.isfinite(top), top, 0.0)

    np.savez('borehole_depths.npz', lon=lon, lat=lat, q_top=top, q_bottom=bot,
             depth_eff=depth_eff, top_eff=top_eff,
             topo_already_corrected=topo_corrected, median_depth=med)
    print('已保存 borehole_depths.npz')


if __name__ == '__main__':
    main()
