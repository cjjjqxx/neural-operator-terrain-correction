# -*- coding: utf-8 -*-
"""地理坐标 DEM 的经向尺度畸变对 B 的影响。

问题
----
SRTM15+ 为 EPSG:4326,像元恒为 0.004167 度。纬向像元恒等于约 464 m,但经向像元
是 464 x cos(纬度) m。现有流水线读取 128 x 128 个像元的窗口,并对所有纬度统一
按 cell_m = 464 m 处理,因此求解器把一块

    59.4 km (南北) x 59.4 x cos(纬度) km (东西)

的地形当成正方形来解。东西方向被拉伸 1/cos(纬度) 倍,该方向的地形坡度因而被
系统性低估,且偏差随纬度单调增大 —— 恰好是一种空间结构化的伪信号。

"""
import sys, os, json, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import zoom

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FN, CELL_M, TILE_KM = 128, 40, 464, 57
DEG_M = 111320.0


def taper_dem(dem):
    zm = dem.mean(); h, w = dem.shape
    k = max(2, int(h * 0.12))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, k)))
    wy, wx = np.ones(h), np.ones(w)
    wy[:k] = ramp; wy[-k:] = ramp[::-1]
    wx[:k] = ramp; wx[-k:] = ramp[::-1]
    return zm + (dem - zm) * np.outer(wy, wx)


def read_window_LEGACY(ds, lon, lat):
    """旧读法(有 bug):经纬向读同样多的像元,于是东西向真实跨度只有
    标称值的 cos(纬度) 倍。保留此函数专供本对比脚本使用,不要在别处调用。"""
    half = int(TILE_KM * 500 / CELL_M)
    r, c = ds.index(lon, lat)
    raw = ds.read(1, window=((r - half, r + half),
                             (c - half, c + half))).astype(float)
    raw[raw < -1000] = np.nan
    if raw.size < 100 or np.isnan(raw).mean() > 0.5 or (raw == 0).mean() > 0.5:
        return None
    m = ~np.isfinite(raw)
    if m.any():
        raw[m] = np.nanmean(raw)
    return taper_dem(zoom(raw, (RES / raw.shape[0], RES / raw.shape[1]), order=1))


def solve(dem):
    from fem3d import Fem3D
    relief = float(dem.max() - dem.min())
    f = Fem3D(nx=FN, ny=FN, nz=10)
    f.build(dem, CELL_M, depth=max(3000, relief * 1.5))
    f.solve_heat(); f.compute_B()
    G = f.B_grid
    return float(G[G.shape[0] // 2, G.shape[1] // 2]), relief


def one(args):
    lon, lat, dem_path = args
    try:
        import rasterio
        from tile_reader import read_tile
        with rasterio.open(dem_path) as ds:
            cur = read_window_LEGACY(ds, lon, lat)      # 旧读法
            fix = read_tile(ds, lon, lat)               # 生产读法(tile_reader)
        if cur is None or fix is None:
            return None
        Bc, rel_c = solve(cur)
        Bf, rel_f = solve(fix)
        return dict(lon=lon, lat=lat, B_current=Bc, B_fixed=Bf,
                    relief_current=rel_c, relief_fixed=rel_f,
                    cos_lat=float(np.cos(np.radians(lat))))
    except Exception as e:
        return {'error': repr(e)[:100]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dem', required=True)
    ap.add_argument('--per-band', type=int, default=12)
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()

    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    lon, lat, B0 = d['lon'], d['lat'], d['B_surface']
    ab = np.abs(B0 - 1.0)
    rng = np.random.default_rng(20260815)
    jobs = []
    BANDS = [(0, 30), (30, 45), (45, 60), (60, 90)]
    for lo, hi in BANDS:
        # 只取有实质校正量的测点,平坦点无从看出差别
        idx = np.where((np.abs(lat) >= lo) & (np.abs(lat) < hi) & (ab >= 0.02))[0]
        if len(idx) == 0:
            continue
        for i in rng.choice(idx, min(a.per_band, len(idx)), replace=False):
            jobs.append((float(lon[i]), float(lat[i]), a.dem))
    print(f'测点 {len(jobs)} 个(|B-1| >= 0.02,按纬度分四带)', flush=True)

    with Pool(a.procs) as p:
        res = [r for r in p.map(one, jobs) if r and 'B_current' in r]
    print(f'成功 {len(res)} / {len(jobs)}\n')

    print(f'{"纬度带":<10}{"n":>4}{"cos(lat)中位":>13}{"|B-1| 现行":>13}'
          f'{"|B-1| 修正":>13}{"相对变化":>11}')
    summary = {}
    for lo, hi in BANDS:
        sub = [r for r in res if lo <= abs(r['lat']) < hi]
        if not sub:
            continue
        cur = np.array([abs(r['B_current'] - 1) for r in sub])
        fix = np.array([abs(r['B_fixed'] - 1) for r in sub])
        chg = 100 * np.median((fix - cur) / np.maximum(cur, 1e-9))
        summary[f'{lo}-{hi}'] = dict(n=len(sub), median_current=float(np.median(cur)),
                                     median_fixed=float(np.median(fix)),
                                     pct_change=float(chg))
        print(f'{f"{lo}-{hi}°":<10}{len(sub):>4}'
              f'{np.median([r["cos_lat"] for r in sub]):>13.3f}'
              f'{np.median(cur):>13.4f}{np.median(fix):>13.4f}{chg:>+10.1f}%')

    allc = np.array([abs(r['B_current'] - 1) for r in res])
    allf = np.array([abs(r['B_fixed'] - 1) for r in res])
    tot = 100 * np.median((allf - allc) / np.maximum(allc, 1e-9))
    summary['all'] = dict(n=len(res), pct_change=float(tot))
    print(f'\n全样本中位相对变化 {tot:+.1f}%')
    print('正值表示现行做法低估了地形效应。')

    json.dump(dict(summary=summary, rows=res),
              open(os.path.join(HERE, 'latitude_scale_test.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('\n已保存 latitude_scale_test.json')


if __name__ == '__main__':
    main()
