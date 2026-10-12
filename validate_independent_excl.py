# -*- coding: utf-8 -*-
"""带排除距离的独立集 —— 回应"独立集与训练地形空间重叠"的质疑(审稿 R1-M4)。

原独立集(validate_independent.py,seed 123)与训练集用同样的均匀抽样且不设排除距离,
采样框也比训练窄。本脚本另建一套 500 块:
  · 采样框与训练集(step2_parallel.CONTINENTS)完全相同;
  · 与任何训练瓦片中心的大圆距离 ≥ 81 km(57 km × √2),两块瓦片的地形足迹因此不可能重叠;
  · 筛选门槛、读取、锥化、有限元配置与训练集逐项相同(复用 validate_independent 的工作函数)。
原独立集及其全部数字不动;本集合只作稳健性检验。
输出 validation_independent_excl.npz / .json(含经纬度,中心像元与全场指标,另附线性谱解基线)。
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SEED = 20260929
N_TARGET = 500
EXCL_KM = 57.0 * np.sqrt(2.0)
DEM_DEFAULT = os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif'))


def xyz(lon, lat):
    r = np.radians
    return np.column_stack([np.cos(r(lat)) * np.cos(r(lon)), np.cos(r(lat)) * np.sin(r(lon)),
                            np.sin(r(lat))]) * 6371.0


def main():
    from validate_independent import process_one_tile
    from step2_parallel import CONTINENTS
    tr = np.load(os.path.join(HERE, 'global_fem_dataset.npz'), mmap_mode='r')
    tree = cKDTree(xyz(np.asarray(tr['lon']), np.asarray(tr['lat'])))
    rng = np.random.RandomState(SEED)
    conts = list(CONTINENTS.items())
    results, tried, rejected_near = [], 0, 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=10) as pool:
        while len(results) < N_TARGET:
            cands = []
            while len(cands) < 100:
                nm, (lo0, lo1, la0, la1) = conts[rng.randint(0, len(conts))]
                lon, lat = rng.uniform(lo0, lo1), rng.uniform(la0, la1)
                dmin, _ = tree.query(xyz(np.array([lon]), np.array([lat]))[0])
                if dmin < EXCL_KM:
                    rejected_near += 1
                    continue
                cands.append((lon, lat))
            futs = [pool.submit(process_one_tile, (lon, lat, tried + i, DEM_DEFAULT))
                    for i, (lon, lat) in enumerate(cands)]
            chunk = [f.result() for f in as_completed(futs)]
            results.extend(sorted((r for r in chunk if r is not None), key=lambda r: r['idx']))
            tried += len(cands)
            print('  %d/%d  已试 %d(另因距训练瓦片过近弃 %d)  %.1f min'
                  % (len(results), N_TARGET, tried, rejected_near, (time.time() - t0) / 60), flush=True)
    results = results[:N_TARGET]
    dems = np.stack([r['dem'] for r in results]).astype(np.float32)
    F = np.stack([r['B_fem'] for r in results]).reshape(N_TARGET, 41, 41).astype(np.float32)
    rl = np.array([r['relief'] for r in results])
    lon = np.array([r['lon'] for r in results])
    lat = np.array([r['lat'] for r in results])
    dmin = tree.query(xyz(lon, lat))[0]

    import torch
    from step3_train_large import FNO3D
    ck = torch.load(os.path.join(HERE, 'fno3d_large_model.pt'), map_location='cuda', weights_only=False)
    m = FNO3D(modes=24, width=96, H_out=41, W_out=41, n_layers=6).cuda().eval()
    m.load_state_dict(ck['state'])
    P = []
    with torch.no_grad():
        for s in range(0, N_TARGET, 50):
            x = (dems[s:s + 50, None] - ck['X_mean']) / ck['X_std']
            P.append(m(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).cuda()).cpu().numpy()[:, 0]
                     * ck['Y_std'] + ck['Y_mean'])
    P = np.concatenate(P).astype(np.float32)

    from linear_spectral_baseline import linear_B, NX
    from tile_reader import CELL_M
    from scipy.ndimage import zoom
    L = 128 * CELL_M
    Lin = np.zeros(N_TARGET)
    for i, dem in enumerate(dems.astype(float)):
        z_bot = dem.min() - max(3000.0, 1.5 * (dem.max() - dem.min()))
        nodes = zoom(dem, ((NX + 1) / 128, (NX + 1) / 128), order=1)
        c = NX // 2
        Lin[i] = linear_B(nodes, L, nodes.mean() - z_bot, (nodes[c, c] - z_bot) / 10.0)[1][c, c]

    def r2(p, t):
        return float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())
    c = 20
    Fc, Pc = F[:, c, c].astype(float), P[:, c, c].astype(float)
    out = dict(n=N_TARGET, seed=SEED, excl_km=float(EXCL_KM), min_dist_km=float(dmin.min()),
               median_dist_km=float(np.median(dmin)), tried=tried, rejected_near=rejected_near,
               r2_field=r2(P.astype(float), F.astype(float)),
               r2_center=r2(Pc, Fc), rmse_center=float(np.sqrt(np.mean((Pc - Fc) ** 2))),
               bias_center=float(np.mean(Pc - Fc)),
               sign_agree_center=float(100 * np.mean(np.sign(Pc - 1) == np.sign(Fc - 1))),
               linear_r2_center=r2(Lin, Fc), linear_rmse_center=float(np.sqrt(np.mean((Lin - Fc) ** 2))),
               relief_median=float(np.median(rl)), by_relief={})
    for lo, hi, lab in ((0, 100, '<100 m'), (100, 500, '100-500 m'), (500, 1e9, '>500 m')):
        mk = (rl >= lo) & (rl < hi)
        out['by_relief'][lab] = dict(n=int(mk.sum()), r2_field=r2(P[mk].astype(float), F[mk].astype(float)),
                                     rmse_center=float(np.sqrt(np.mean((Pc[mk] - Fc[mk]) ** 2))),
                                     linear_rmse_center=float(np.sqrt(np.mean((Lin[mk] - Fc[mk]) ** 2))))
    ab = np.abs(Fc - 1)
    out['by_band'] = {}
    for lo, hi, lab in ((0, .02, '<0.02'), (.02, .10, '0.02-0.10'), (.10, 9, '>=0.10')):
        mk = (ab >= lo) & (ab < hi)
        out['by_band'][lab] = dict(n=int(mk.sum()), rmse_center=float(np.sqrt(np.mean((Pc[mk] - Fc[mk]) ** 2))),
                                   linear_rmse_center=float(np.sqrt(np.mean((Lin[mk] - Fc[mk]) ** 2))),
                                   bias_center=float(np.mean(Pc[mk] - Fc[mk])),
                                   linear_bias_center=float(np.mean(Lin[mk] - Fc[mk])))
    np.savez_compressed(os.path.join(HERE, 'validation_independent_excl.npz'), dems=dems, B_fem=F, B_fno=P,
                        B_linear_center=Lin.astype(np.float32), reliefs=rl, lon=lon, lat=lat, dist_train_km=dmin)
    json.dump(out, open(os.path.join(HERE, 'validation_independent_excl.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
