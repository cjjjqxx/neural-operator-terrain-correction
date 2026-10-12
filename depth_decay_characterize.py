# -*- coding: utf-8 -*-
"""
depth_decay_characterize.py -- 刻画 B 的深度衰减函数 f(D)

钻孔热流来自 [0, D] 区间温度梯度的线性拟合, 因此其对应的有效传递因子是
B 在该区间上的**平均值**, 而不是深度 D 处的瞬时值:

    B_eff(D) = (1/D) ∫₀^D B(z) dz

定义保留比例
    f(D) = (B_eff(D) - 1) / (B_surf - 1)

f = 1 表示用地表 B 无偏; f < 1 表示用地表 B 会放大校正量。
本脚本保存完整 B(z) 剖面, 以便对任意 D 做积分。
"""
import sys, time, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from fem3d import Fem3D, Q_REF

NZ = 40
CELL_M = 57000.0 / 128
N_PER_BAND = 10          # 每档取样数
D_LIST = [100, 146, 250, 390, 500, 1000, 1450, 2000]   # 含 IHFC 深度分位数


def B_profile(fem):
    pts, T = fem.pts, fem.T
    nx, ny, nz = fem.nx, fem.ny, fem.nz
    ix, iy = nx // 2, ny // 2

    def node(iz):
        return iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix

    z_surf = pts[2, node(nz)]
    depths, Bs = [], []
    for iz in range(nz, 0, -1):
        n_up, n_dn = node(iz), node(iz - 1)
        dz = pts[2, n_up] - pts[2, n_dn]
        if dz <= 0.01:
            continue
        B = -fem.K * ((T[n_up] - T[n_dn]) / dz) / Q_REF
        depths.append(z_surf - 0.5 * (pts[2, n_up] + pts[2, n_dn]))
        Bs.append(B)
    return np.array(depths), np.array(Bs)


def B_eff(depths, Bs, D, B_surf):
    """[0, D] 区间上 B 的平均值; 用地表值外推到 z=0。"""
    if D <= depths[0]:
        return B_surf
    zs = np.concatenate([[0.0], depths[depths <= D]])
    bs = np.concatenate([[B_surf], Bs[depths <= D]])
    if zs[-1] < D and D <= depths.max():
        zs = np.append(zs, D)
        bs = np.append(bs, np.interp(D, depths, Bs))
    return float(np.trapz(bs, zs) / (zs[-1] - zs[0]))


def main():
    d = np.load('global_fem_dataset.npz')
    X, relief, Y = d['X'], d['relief'], d['Y']
    c = Y.shape[-1] // 2
    B_ref = Y[:, 0, c, c]
    absB1 = np.abs(B_ref - 1)

    bands = [('near_flat', 0, 0.02), ('moderate', 0.02, 0.1),
             ('high', 0.1, 0.2), ('extreme', 0.2, 10)]
    rng = np.random.RandomState(1)
    picks = []
    for name, lo, hi in bands:
        idx = np.where((absB1 >= lo) & (absB1 < hi))[0]
        sel = rng.choice(idx, min(N_PER_BAND, len(idx)), replace=False)
        picks += [(name, int(i)) for i in sel]

    print(f'{len(picks)} 个测点, nz={NZ}', flush=True)
    rows, t0 = [], time.time()
    for k, (band, i) in enumerate(picks):
        f = Fem3D(nx=40, ny=40, nz=NZ)
        f.build(X[i].squeeze().astype(float), CELL_M)
        f.solve_heat()
        depths, Bs = B_profile(f)
        B_surf = float(Bs[0])
        dev0 = B_surf - 1.0
        rec = dict(band=band, idx=i, relief=float(relief[i]), B_surf=B_surf,
                   depths=depths.tolist(), Bs=Bs.tolist(), f={})
        for D in D_LIST:
            if D > depths.max():
                continue
            be = B_eff(depths, Bs, D, B_surf)
            rec['f'][str(D)] = (be - 1.0) / dev0 if abs(dev0) > 1e-9 else 1.0
        rows.append(rec)
        if (k + 1) % 5 == 0:
            print(f'  {k+1}/{len(picks)}  ({time.time()-t0:.0f}s)', flush=True)

    json.dump(rows, open('depth_decay_profiles.json', 'w', encoding='utf-8'))

    print(f'\n完成, 耗时 {time.time()-t0:.0f}s', flush=True)
    print('\n=== 保留比例 f(D) = (B_eff(D)-1)/(B_surf-1) ===', flush=True)
    print('f=1 表示用地表B无偏; f<1 表示用地表B会把校正量放大 1/f 倍\n', flush=True)
    hdr = f'{"档位":<11}{"n":>4}' + ''.join(f'{D:>9}m' for D in D_LIST)
    print(hdr, flush=True)
    for name, _, _ in bands:
        sub = [r for r in rows if r['band'] == name]
        if not sub:
            continue
        line = f'{name:<11}{len(sub):>4}'
        for D in D_LIST:
            vs = [r['f'][str(D)] for r in sub if str(D) in r['f']]
            line += f'{np.median(vs):>10.3f}' if vs else f'{"n/a":>10}'
        print(line, flush=True)

    print(f'\n{"全部(中位数)":<11}{len(rows):>4}', end='', flush=True)
    overall = {}
    for D in D_LIST:
        vs = [r['f'][str(D)] for r in rows if str(D) in r['f']]
        overall[D] = float(np.median(vs)) if vs else None
        print(f'{overall[D]:>10.3f}' if vs else f'{"n/a":>10}', end='', flush=True)
    print(flush=True)
    print(f'\n{"四分位距":<11}{"":>4}', end='', flush=True)
    for D in D_LIST:
        vs = [r['f'][str(D)] for r in rows if str(D) in r['f']]
        print(f'{np.percentile(vs,75)-np.percentile(vs,25):>10.3f}' if vs else f'{"n/a":>10}', end='', flush=True)
    print(flush=True)

    json.dump(overall, open('depth_decay_f.json', 'w', encoding='utf-8'), indent=2)
    print('\n已保存 depth_decay_profiles.json / depth_decay_f.json', flush=True)


if __name__ == '__main__':
    main()
