# -*- coding: utf-8 -*-
"""逐测点地形指标:瓦片起伏与脊谷指数(36,392 个测点)。

rebuild_points.npz 只存了校正结果,没有测点所在瓦片的地形描述;而两处需要它:
  · 逐测点可信度(site_reliability.py)—— 算子误差随起伏变化,按起伏分档才符合物理;
  · 校正量—地形散点图 —— 横轴必须是地形本身,不能用模型输出 |B−1|,否则是拿模型
    输出解释模型给出的校正量,有循环论证之嫌。

读取入口与训练、应用完全相同(tile_reader.read_tile),三个量:
  relief_m      锥化后瓦片的高差 max − min —— 与有限元域深规则、独立验证集的
                起伏分档同一口径
  relief_raw_m  未锥化瓦片的高差
  rel_elev      脊谷指数 = (中心像元高程 − 瓦片平均高程) / 未锥化起伏;正为脊、负为谷
                (与 davies_terrain_baseline.py 的裸地形代理同一定义)
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
DEM_DEFAULT = os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif'))
_DS = None


def _init(dem_path):
    global _DS
    import rasterio
    sys.path.insert(0, HERE)
    _DS = rasterio.open(dem_path)


def one(args):
    i, lon, lat = args
    from tile_reader import read_tile, taper_dem, RES
    try:
        raw = read_tile(_DS, lon, lat, apply_taper=False)
    except Exception:
        raw = None
    if raw is None:
        return i, np.nan, np.nan, np.nan
    c = RES // 2
    rr = float(raw.max() - raw.min())
    tp = taper_dem(raw)
    re = (float(raw[c, c]) - float(raw.mean())) / rr if rr > 0 else 0.0
    return i, float(tp.max() - tp.min()), rr, re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dem', default=DEM_DEFAULT)
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    lon, lat = d['lon'], d['lat']
    n = len(lon)
    out = np.full((n, 3), np.nan)
    t0 = time.time()
    with Pool(a.procs, initializer=_init, initargs=(a.dem,)) as p:
        for k, (i, r, rr, re) in enumerate(p.imap_unordered(
                one, [(i, float(lon[i]), float(lat[i])) for i in range(n)], chunksize=64), 1):
            out[i] = (r, rr, re)
            if k % 5000 == 0:
                print('  %s/%s  %.1f min' % (format(k, ','), format(n, ','),
                                             (time.time() - t0) / 60), flush=True)
    ok = np.isfinite(out[:, 0])
    print('完成 %s 个,失败 %d 个,用时 %.1f min'
          % (format(int(ok.sum()), ','), int((~ok).sum()), (time.time() - t0) / 60))
    print('起伏(锥化后)中位 %.0f m;< 25 m 者 %.1f%%(训练集的起伏门槛之下)'
          % (np.nanmedian(out[:, 0]), 100 * np.mean(out[ok, 0] < 25)))
    np.savez_compressed(os.path.join(HERE, 'site_terrain.npz'), lon=lon, lat=lat,
                        relief_m=out[:, 0].astype('float32'),
                        relief_raw_m=out[:, 1].astype('float32'),
                        rel_elev=out[:, 2].astype('float32'))
    print('已保存 site_terrain.npz')


if __name__ == '__main__':
    main()
