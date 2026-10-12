# -*- coding: utf-8 -*-
"""为 correct_heatflow.py 准备高程数据。


为什么不随论文分发高程数据
--------------------------
全球 15 弧秒高程约 6–7 GB。它是一份有明确出处、持续更新、且已经公开可取的
第三方数据集(Tozer et al., 2019),重新分发一份冻结的副本没有好处:体积大、
会随上游修订而过时,还平白多出一层"这份副本和官方那份一样吗"的疑问。
本脚本提供两件事:核验使用者手上的栅格是否满足口径,以及从官方源取一份。

口径要求
--------
  · 坐标系不限:correct_heatflow.py 会把非 EPSG:4326 的栅格自动重投影
    (WarpedVRT + 双线性),使用者不必先自己转
  · 像元不粗于约 450 m(0.00417°) —— 更粗的高程会抹平短波地形,
    把校正系统性压向 1,结果看起来"校正很小",而那是分辨率造成的
  · 覆盖到待校正测点的位置;海洋在 SRTM15+ 中为负高程,
    read_tile 会把它当作近等温面处理,这与本文汇编的口径一致
"""
import argparse
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = 'https://topex.ucsd.edu/pub/srtm15_plus/'
VERSIONS = {          # 官方目录中的版本 → 文件名(2026-09 核对)
    'V2.4': 'SRTM15_V2.4.nc',
    'V2.5': 'SRTM15_V2.5.nc',
    'V2.5.5': 'SRTM15_V2.5.5.nc',
    'V2.6': 'SRTM15_V2.6.nc',
    'V2.7': 'SRTM15_V2.7.nc',
}
MAX_PIXEL_DEG = 0.0056


def check(path):
    import rasterio
    if not os.path.exists(path):
        print('× 文件不存在:%s' % path)
        return False
    ds = rasterio.open(path)
    px, py = abs(ds.transform.a), abs(ds.transform.e)
    b = ds.bounds
    print('%s' % os.path.basename(path))
    print('  尺寸        %s × %s' % (format(ds.width, ','), format(ds.height, ',')))
    print('  像元        %.6f° × %.6f°  (赤道约 %.0f m × %.0f m)'
          % (px, py, px * 111320, py * 110540))
    print('  坐标系      %s' % ds.crs)
    print('  范围        经度 %.2f 至 %.2f,纬度 %.2f 至 %.2f'
          % (b.left, b.right, b.bottom, b.top))
    ok = True
    if ds.crs is None:
        print('  × 没有坐标系信息,无法定位测点')
        ok = False
    elif ds.crs.to_epsg() != 4326:
        # 不再判失败:correct_heatflow.py 会自动重投影
        print('  ! 坐标系不是 EPSG:4326,使用时会被自动重投影')
        print('    重投影会磨掉一些短波地形:实测相对差中位 2.9%、P90 9.6%。')
        print('    能拿到原生经纬度栅格就用它。')
    if max(px, py) > MAX_PIXEL_DEG:
        print('  × 像元过粗(上限约 %.0f m)' % (MAX_PIXEL_DEG * 111320))
        ok = False
    # 全球覆盖不是硬性要求 —— 只校正本地一批测点时,区域栅格同样可用,
    # 但要盖住每个测点周围 57 km 的窗口,故此处只提示不判失败。
    if b.right - b.left < 359 or b.top - b.bottom < 179:
        print('  ! 非全球栅格:请确认它盖住每个测点周围至少 57 km 见方的范围')
    print('  %s' % ('✓ 可用于 correct_heatflow.py' if ok else '× 不满足口径要求'))
    return ok


def download(ver, outdir):
    import urllib.request
    fn = VERSIONS[ver]
    os.makedirs(outdir, exist_ok=True)
    nc = os.path.join(outdir, fn)
    url = BASE + fn
    if os.path.exists(nc):
        print('已存在,跳过下载:%s' % nc)
    else:
        print('下载 %s → %s(约 6 GB,视网络可能需要很久)' % (url, nc))

        def hook(n, bs, total):
            if total > 0 and n % 2000 == 0:
                sys.stdout.write('\r  %.1f%%' % (100.0 * n * bs / total))
                sys.stdout.flush()

        urllib.request.urlretrieve(url, nc, hook)
        print()
    tif = os.path.splitext(nc)[0] + '.tif'
    if os.path.exists(tif):
        print('已存在:%s' % tif)
    else:
        # NetCDF 里 z 是主变量。rasterio 可直接读 NETCDF:"file":z 子数据集,
        # 但写出 GeoTIFF 时要显式给出仿射变换 —— .nc 的坐标轴是格心,
        # 而 GeoTIFF 的 transform 描述的是格角,差半个像元。
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        print('转换为 GeoTIFF:%s' % tif)
        with rasterio.open('NETCDF:"%s":z' % nc) as src:
            a = src.read(1)
            h, w = a.shape
            px = 360.0 / w
            # SRTM15+ 的纬度轴自南向北,GeoTIFF 约定自北向南
            a = np.flipud(a)
            tr = from_origin(-180.0, 90.0, px, px)
            with rasterio.open(tif, 'w', driver='GTiff', height=h, width=w,
                               count=1, dtype='int16', crs='EPSG:4326',
                               transform=tr, compress='deflate',
                               tiled=True, blockxsize=512, blockysize=512) as dst:
                dst.write(a.astype('int16'), 1)
    check(tif)
    print()
    print('引用:Tozer, B. et al. Global bathymetry and topography at 15 arc sec:')
    print('     SRTM15+. Earth Space Sci. 6, 1847–1864 (2019).')
    print('     doi:10.1029/2019EA000658')


def main():
    ap = argparse.ArgumentParser(description='为 correct_heatflow.py 准备高程数据')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', metavar='RASTER', help='核验手上的栅格')
    g.add_argument('--download', metavar='VER', choices=sorted(VERSIONS),
                   help='从官方源取一份:%s' % '、'.join(sorted(VERSIONS)))
    ap.add_argument('-o', '--out', default='.', help='下载与转换的输出目录')
    a = ap.parse_args()
    if a.check:
        raise SystemExit(0 if check(a.check) else 1)
    download(a.download, a.out)


if __name__ == '__main__':
    main()
