# -*- coding: utf-8 -*-
"""窗口(footprint)收敛性 —— 单元尺寸固定,只变窗口。

为什么要重做
------------
原 footprint_depth_expand.py 固定 res_px=128 且 nx=40,于是有限元单元尺寸
= 窗口宽度 / 40,随窗口同步变粗(半宽 10 km 时 500 m,半宽 50 km 时 2,500 m)。
窗口变大纳入更多长波会使 |B-1| 增大,网格变粗抹掉短波会使 |B-1| 减小,两者方向
相反且被绑在一起,因此那组数据无法回答"57 km 够不够"。

本脚本把单元尺寸固定在生产值 57 km / 40 = 1,425 m,让 nx 随窗口等比例增加,
于是窗口是唯一变量。配套的 mesh_resolution_test.py 则固定窗口只变网格。两者合起来
把谱截断的长波端与短波端分别定量。

窗口   57    80    114   160  km
nx     40    56    80    112       (单元恒为 1,425 m)
"""
import sys, os, json, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
# 本脚本按原生像元读取:像元数 npx = 窗口宽度 / 464 m,read_tile 再按 npx 重采样,
# 所以实际像元恰为约 464 m,建网格时按 464 m 是自洽的。这与生产瓦片
# (固定 128 像元、约 445 m)不同,不可照搬 tile_reader.CELL_M。
CELL_M = 464.0                      # DEM 原生像元
ELEM_M = 57000.0 / 40               # 生产单元尺寸 1,425 m,全程固定
WINDOWS_KM = [int(x) if float(x).is_integer() else float(x)
              for x in os.environ.get('FP_WINDOWS', '57,80,114,160').split(',')]
OUT_JSON = os.environ.get('FP_OUT', 'footprint_fixed_resolution_test.json')
BANDS = [(0.05, 0.10, 10), (0.10, 0.20, 10), (0.20, 9.0, 10)]   # 只取有意义的档位


def taper_dem(dem):
    zm = dem.mean(); h, w = dem.shape
    k = max(2, int(h * 0.12))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, k)))
    wy, wx = np.ones(h), np.ones(w)
    wy[:k] = ramp; wy[-k:] = ramp[::-1]
    wx[:k] = ramp; wx[-k:] = ramp[::-1]
    return zm + (dem - zm) * np.outer(wy, wx)


def one(args):
    lon, lat, dem_path = args
    try:
        import rasterio
        from fem3d import Fem3D
        out = {'lon': lon, 'lat': lat, 'B': {}, 'nx': {}, 'relief': {}}
        from tile_reader import read_tile
        with rasterio.open(dem_path) as ds:
            for wkm in WINDOWS_KM:
                npx = int(round(wkm * 1000 / CELL_M))
                dem = read_tile(ds, lon, lat, tile_km=wkm, res=npx)
                if dem is None:
                    return None
                nx = int(round(wkm * 1000 / ELEM_M))      # 单元尺寸恒定
                relief = float(dem.max() - dem.min())
                f = Fem3D(nx=nx, ny=nx, nz=10)
                f.build(dem, CELL_M, depth=max(3000, relief * 1.5))
                f.solve_heat(); f.compute_B()
                G = f.B_grid
                out['B'][wkm] = float(G[G.shape[0] // 2, G.shape[1] // 2])
                out['nx'][wkm] = nx
                out['relief'][wkm] = relief
        return out
    except Exception as e:
        return {'error': repr(e)[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dem', required=True)
    ap.add_argument('--procs', type=int, default=6)
    a = ap.parse_args()

    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    lon, lat, B0 = d['lon'], d['lat'], d['B_surface']
    ab = np.abs(B0 - 1.0)
    rng = np.random.default_rng(20260815)
    jobs, meta = [], []
    for lo, hi, k in BANDS:
        idx = np.where((ab >= lo) & (ab < hi))[0]
        for i in rng.choice(idx, min(k, len(idx)), replace=False):
            jobs.append((float(lon[i]), float(lat[i]), a.dem))
            meta.append((float(ab[i]), f'{lo}-{hi}'))
    print(f'测点 {len(jobs)} 个(|B-1| >= 0.05)', flush=True)
    print(f'窗口 {WINDOWS_KM} km,对应 nx {[int(round(w*1000/ELEM_M)) for w in WINDOWS_KM]},'
          f'单元恒为 {ELEM_M:.0f} m', flush=True)

    with Pool(a.procs) as p:
        res = p.map(one, jobs)
    rows = [(m, r) for m, r in zip(meta, res) if r and 'B' in r]
    print(f'成功 {len(rows)} / {len(jobs)}\n')

    print(f'{"|B-1| 档":<12}{"n":>4}' + ''.join(f'{f"{w}km":>10}' for w in WINDOWS_KM)
          + f'{"57→160 变化":>14}')
    summ = {}
    for lo, hi, _ in BANDS:
        sub = [r for (m, r) in rows if lo <= m[0] < hi]
        if not sub:
            continue
        med = {w: float(np.median([abs(s['B'][w] - 1) for s in sub])) for w in WINDOWS_KM}
        chg = 100 * (med[WINDOWS_KM[-1]] - med[WINDOWS_KM[0]]) / med[WINDOWS_KM[0]]
        summ[f'{lo}-{hi}'] = dict(n=len(sub), median_absB1=med, pct_change=float(chg))
        print(f'{f"{lo}-{hi}":<12}{len(sub):>4}'
              + ''.join(f'{med[w]:>10.4f}' for w in WINDOWS_KM) + f'{chg:>+13.1f}%')

    # 逐步增量:看曲线有没有在 57 km 之后趋平
    print(f'\n逐步相对增量(相对上一档):')
    print(f'{"|B-1| 档":<12}' + ''.join(f'{f"{WINDOWS_KM[i]}→{WINDOWS_KM[i+1]}":>12}'
                                        for i in range(len(WINDOWS_KM) - 1)))
    for k, v in summ.items():
        med = v['median_absB1']
        print(f'{k:<12}' + ''.join(
            f'{100*(med[WINDOWS_KM[i+1]]-med[WINDOWS_KM[i]])/med[WINDOWS_KM[i]]:>+11.1f}%'
            for i in range(len(WINDOWS_KM) - 1)))

    json.dump(dict(windows_km=WINDOWS_KM, elem_m=ELEM_M, summary=summ,
                   rows=[{'absB1': m[0], 'band': m[1],
                          'B': {str(k): v for k, v in r['B'].items()}} for m, r in rows]),
              open(os.path.join(HERE, OUT_JSON), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print()
    print('已保存 ' + OUT_JSON)


if __name__ == '__main__':
    main()
