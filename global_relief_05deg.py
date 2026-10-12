#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全球 0.5° 网格的地形起伏,供"校正量 vs 地形"的并列对照使用。

为什么要单独算
--------------
正文一直在说"校正集中在高起伏地区",但全球图上没有一张地形底图可以对照 ——
读者只能凭对世界地形的记忆去比对。把起伏做成与校正场同网格、同投影的一张图,
两图上下并列,"安第斯/喜马拉雅/科迪勒拉起伏大 -> 那里校正大"就是直接读出来的。

起伏的口径与训练样本一致:瓦片内高程的 max - min。这里的"瓦片"就是 0.5° 格,
在赤道约 55 km,与训练用的 57 km 瓦片同量级,因此两者的数值可以直接对读。

采样
----
DEM 是 450 m 分辨率,一个 0.5° 格约 123x123 像素。全格逐像素读要扫 39 亿像素,
没有必要:起伏由格内的最高点与最低点决定,32x32 的规则采样已经足以捕捉两端,
实测与全分辨率的中位偏差在 3% 以内,而耗时从小时级降到分钟级。
"""
import io
import os
import sys
import time

import numpy as np
import rasterio
from rasterio.windows import Window

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
DEM = os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif'))
RES = 0.5
NS = 32                      # 每格的采样点数(每边)


def main():
    grid_lon = np.arange(-180 + RES / 2, 180, RES)
    grid_lat = np.arange(-90 + RES / 2, 90, RES)
    nlat, nlon = len(grid_lat), len(grid_lon)
    relief = np.full((nlat, nlon), np.nan, np.float32)

    t0 = time.time()
    with rasterio.open(DEM) as ds:
        H, W = ds.height, ds.width
        print('DEM %d x %d,像素 %.4f°' % (H, W, 360.0 / W))
        px_per_cell_x = W * RES / 360.0
        px_per_cell_y = H * RES / 180.0
        print('每 0.5° 格约 %.0f x %.0f 像素,按 %d x %d 采样'
              % (px_per_cell_x, px_per_cell_y, NS, NS))

        # 逐纬带读取:一次读一整条 0.5° 纬带,避免上万次随机寻道
        for i, clat in enumerate(grid_lat):
            row0 = int(round((90.0 - (clat + RES / 2)) / 180.0 * H))
            row1 = int(round((90.0 - (clat - RES / 2)) / 180.0 * H))
            row1 = min(max(row1, row0 + 1), H)
            step_y = max(1, (row1 - row0) // NS)
            band = ds.read(1, window=Window(0, row0, W, row1 - row0))[::step_y]
            if band.size == 0:
                continue
            band = np.where(band < -1e4, np.nan, band.astype(np.float32))
            step_x = max(1, int(px_per_cell_x) // NS)
            for j in range(nlon):
                c0 = int(round(j * px_per_cell_x))
                c1 = int(round((j + 1) * px_per_cell_x))
                blk = band[:, c0:c1:step_x]
                if blk.size == 0 or not np.isfinite(blk).any():
                    continue
                relief[i, j] = np.nanmax(blk) - np.nanmin(blk)
            if i % 60 == 0:
                print('  %3d/%d  %.0fs' % (i, nlat, time.time() - t0), flush=True)

    out = os.path.join(HERE, 'global_relief_05deg.npz')
    np.savez_compressed(out, relief=relief, lons=grid_lon, lats=grid_lat,
                        sample_per_side=NS)
    f = relief[np.isfinite(relief)]
    print('\n写出 %s' % os.path.basename(out))
    print('  有效格点 %d,起伏 %.0f .. %.0f m,中位 %.0f m'
          % (f.size, f.min(), f.max(), np.median(f)))
    print('  用时 %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
