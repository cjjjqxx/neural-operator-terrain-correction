# -*- coding: utf-8 -*-
"""有限元网格分辨率对 B 与 f(D) 的影响 —— 窗口固定,只变 nx。

为什么必须变 nx 而不是 DEM 分辨率
--------------------------------
Fem3D.build() 内部有 zoom(dem_tile, ((ny+1)/Ny, (nx+1)/Nx)),把 128x128 的 DEM
降采样到 (nx+1)x(nx+1) 去匹配有限元网格。因此生产配置下真正的水平分辨率是
57 km / 40 = 1.43 km,而不是 DEM 的 445 m;DEM 多出来的细节在求解前就被丢弃。

已有的 footprint 收敛性测试固定 res_px=128、让 cell_m = 窗口宽度/128,于是窗口
和分辨率是绑在一起变的,两个方向相反的效应(窗口变大纳入更多长波使 |B-1| 增大,
网格变粗抹掉短波使 |B-1| 减小)无法分离。本脚本固定 57 km 窗口,只变 nx,给出
干净的分辨率收敛曲线。

nx=128 时单元尺寸 445 m,与 DEM 像元一致,是这条阶梯的自然终点。

采样按 |B-1| 分层并向极端地形倾斜:地形校正的意义本来就在极端地形上,平坦测点
校正量本就极小。
"""
import sys, os, json, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import zoom

HERE = os.path.dirname(os.path.abspath(__file__))
from tile_reader import CELL_M     # 57 km / 128 像元,约 445 m,与训练集同一口径
TILE_KM, RES_PX = 57, 128
NX_LADDER = [40, 64, 96, 128]
BANDS = [(0.0, 0.02, 5), (0.02, 0.05, 5), (0.05, 0.10, 10),
         (0.10, 0.20, 10), (0.20, 9.0, 10)]     # (lo, hi, 抽样数) —— 偏向极端


def taper_dem(dem):
    zm = dem.mean(); h, w = dem.shape
    k = max(2, int(h * 0.12))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, k)))
    wy, wx = np.ones(h), np.ones(w)
    wy[:k] = ramp; wy[-k:] = ramp[::-1]
    wx[:k] = ramp; wx[-k:] = ramp[::-1]
    return zm + (dem - zm) * np.outer(wy, wx)


def load_tile(ds, lon, lat, tile_km=None):
    """统一走 tile_reader,保证与训练集同几何(经向按 cos(纬度) 修正)。

    vertical_mesh_test.py 也导入本函数,改这一处两边同时生效。
    """
    from tile_reader import read_tile
    return read_tile(ds, lon, lat, tile_km=tile_km or TILE_KM, res=RES_PX)


def b_profile(f):
    """中心列的 B(z) 剖面,用于检查 f(D) 是否也被分辨率带偏。"""
    pts, T, nx, ny, nz = f.pts, f.T, f.nx, f.ny, f.nz
    ix, iy = nx // 2, ny // 2
    node = lambda iz: iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix
    zs, Bs = [], []
    z_surf = pts[2, node(nz)]
    for iz in range(nz, 0, -1):
        up, dn = node(iz), node(iz - 1)
        dz = pts[2, up] - pts[2, dn]
        if dz <= 0.01:
            continue
        zs.append(z_surf - 0.5 * (pts[2, up] + pts[2, dn]))
        Bs.append(-f.K * ((T[up] - T[dn]) / dz) / 0.065)
    return np.array(zs), np.array(Bs)


def one(args):
    lon, lat, dem_path, want_profile = args
    try:
        import rasterio
        from fem3d import Fem3D
        with rasterio.open(dem_path) as ds:
            dem = load_tile(ds, lon, lat)
        if dem is None:
            return None
        relief = float(dem.max() - dem.min())
        out = {'lon': lon, 'lat': lat, 'relief': relief, 'B': {}, 'prof': {}}
        for nx in NX_LADDER:
            f = Fem3D(nx=nx, ny=nx, nz=10)
            f.build(dem, CELL_M, depth=max(3000, relief * 1.5))
            f.solve_heat(); f.compute_B()
            G = f.B_grid
            out['B'][nx] = float(G[G.shape[0] // 2, G.shape[1] // 2])
        if want_profile:                      # 只对极端点做 nz=40 的深度剖面
            for nx in (NX_LADDER[0], NX_LADDER[-1]):
                f = Fem3D(nx=nx, ny=nx, nz=40)
                f.build(dem, CELL_M, depth=max(3000, relief * 1.5))
                f.solve_heat(); f.compute_B()
                G = f.B_grid
                bs = float(G[G.shape[0] // 2, G.shape[1] // 2])
                z, B = b_profile(f)
                out['prof'][nx] = dict(B_surf=bs, z=z.tolist(), B=B.tolist())
        return out
    except Exception as e:
        return {'error': repr(e)[:120], 'lon': lon, 'lat': lat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dem', required=True)
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--profiles', action='store_true',
                    help='额外跑 nz=40 深度剖面(很吃内存与时间,默认关闭)')
    a = ap.parse_args()

    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    lon, lat, B0 = d['lon'], d['lat'], d['B_surface']
    ab = np.abs(B0 - 1.0)
    rng = np.random.default_rng(20260815)
    jobs, meta = [], []
    for lo, hi, k in BANDS:
        idx = np.where((ab >= lo) & (ab < hi))[0]
        pick = rng.choice(idx, min(k, len(idx)), replace=False)
        for i in pick:
            # 只对 |B-1| >= 0.05 的极端点跑深度剖面(nz=40 很慢)
            jobs.append((float(lon[i]), float(lat[i]), a.dem, a.profiles and lo >= 0.10))
            meta.append((float(ab[i]), float(B0[i]), f'{lo}-{hi}'))
    print(f'测点 {len(jobs)} 个,nx 阶梯 {NX_LADDER}(单元尺寸 '
          f'{", ".join(f"{TILE_KM*1000/n:.0f}m" for n in NX_LADDER)})', flush=True)

    with Pool(a.procs) as p:
        res = p.map(one, jobs)

    rows = [(m, r) for m, r in zip(meta, res) if r and 'B' in r]
    print(f'成功 {len(rows)} / {len(jobs)}\n')
    print(f'{"|B-1| 档":<12}{"n":>4}' + ''.join(f'{f"nx={n}":>11}' for n in NX_LADDER)
          + f'{"40→128 变化":>14}')
    summary = {}
    for lo, hi, _ in BANDS:
        sub = [r for (m, r) in rows if lo <= m[0] < hi]
        if not sub:
            continue
        arr = {n: np.array([abs(s['B'][n] - 1) for s in sub]) for n in NX_LADDER}
        base, fine = arr[NX_LADDER[0]], arr[NX_LADDER[-1]]
        chg = 100 * np.median((fine - base) / np.maximum(base, 1e-9))
        summary[f'{lo}-{hi}'] = dict(n=len(sub), median_absB1={str(n): float(np.median(arr[n]))
                                                              for n in NX_LADDER},
                                     pct_change_40_to_128=float(chg))
        print(f'{f"{lo}-{hi}":<12}{len(sub):>4}'
              + ''.join(f'{np.median(arr[n]):>11.4f}' for n in NX_LADDER)
              + f'{chg:>+13.1f}%')

    # f(D) 是否也被分辨率带偏
    prof = [r for (m, r) in rows if r.get('prof')]
    if prof:
        print(f'\nf(500 m) 与 e 折深度(仅 |B-1| ≥ 0.05 的 {len(prof)} 个点):')
        print(f'{"":16}{"nx=40":>12}{"nx=128":>12}{"相对变化":>12}')
        for lab, fn in [('f(500 m)', 500.0), ('f(1000 m)', 1000.0)]:
            vals = {}
            for n in (NX_LADDER[0], NX_LADDER[-1]):
                v = []
                for r in prof:
                    pr = r['prof'][n]; z = np.array(pr['z']); Bz = np.array(pr['B'])
                    bs = pr['B_surf']
                    if abs(bs - 1) < 1e-6:
                        continue
                    sel = z <= fn
                    if sel.sum() < 2:
                        continue
                    beff = np.trapz(np.r_[bs, Bz[sel]], np.r_[0.0, z[sel]]) / z[sel].max()
                    v.append((beff - 1) / (bs - 1))
                vals[n] = float(np.median(v)) if v else float('nan')
            a0, a1 = vals[NX_LADDER[0]], vals[NX_LADDER[-1]]
            print(f'{lab:<16}{a0:>12.4f}{a1:>12.4f}{100*(a1-a0)/a0:>+11.1f}%')
            summary[lab] = dict(nx40=a0, nx128=a1)

    json.dump(dict(nx_ladder=NX_LADDER, summary=summary,
                   rows=[{'absB1': m[0], 'B_surface': m[1], 'band': m[2],
                          'relief': r['relief'], 'B': {str(k): v for k, v in r['B'].items()}}
                         for m, r in rows]),
              open(os.path.join(HERE, 'mesh_resolution_test.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('\n已保存 mesh_resolution_test.json')


if __name__ == '__main__':
    main()
