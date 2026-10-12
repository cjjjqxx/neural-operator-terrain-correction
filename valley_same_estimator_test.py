# -*- coding: utf-8 -*-
"""谷地过度代表性：用同一估计器重算（预审 R1-M7）。

R1 指出的问题：稿件里被相减的两个量来自**不同估计器**。
  58.8%（测点位于 B>1 环境的占比）由**算子预测**在 36,392 个测点上算得；
  52.0 ± 1.1%（随机陆地参照）由**有限元**在 2,000 个随机瓦片上算得。
算子与有限元的符号一致率只有 91.6%，且该数来自偏高起伏的独立集，
**近平坦区间（占样本四分之三）的符号一致率从未报告**。
8.4% 的符号分歧与 6.8 个百分点的效应同量级，因此差值可能部分源于估计器差异。

本脚本把测点侧也改用有限元：在真实测点上随机抽样、实解有限元、算 B>1 占比，
与同样由有限元得到的随机陆地参照比较。两侧同口径，差值才干净。
另单独报告近平坦区间的算子—有限元符号一致率，补上那个从未报告的数。

有限元单瓦片约 1.64 s，默认抽 600 点，约 17 分钟。
"""
import os
import sys
import json
import argparse

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--dem', default=os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif')))
    a = ap.parse_args()

    from tile_reader import read_tile, CELL_M
    from fem3d import Fem3D
    import torch
    from step3_train_large import FNO3D

    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    lon, lat, B_op_all = d['lon'], d['lat'], d['B_surface']
    rng = np.random.default_rng(20260826)
    idx = rng.choice(len(lon), min(a.n, len(lon)), replace=False)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FNO3D(modes=24, width=96, n_layers=6).to(dev)
    model.load_state_dict(torch.load(os.path.join(HERE, 'fno3d_large_model.pt'),
                                     map_location=dev))
    model.eval()

    fem_B, op_B, rel = [], [], []
    for n_done, i in enumerate(idx, 1):
        try:
            dem = read_tile(a.dem, float(lon[i]), float(lat[i]))
            r = float(np.nanmax(dem) - np.nanmin(dem))
            f = Fem3D(nx=40, ny=40, nz=10)
            f.build(dem, CELL_M, depth=max(3000.0, 1.5 * r))
            f.solve_heat(); f.compute_B()
            G = f.B_grid
            fem_B.append(float(G[G.shape[0] // 2, G.shape[0] // 2]))
            op_B.append(float(B_op_all[i]))
            rel.append(r)
        except Exception as e:
            print(f'  跳过 {i}: {e}')
        if n_done % 100 == 0:
            print(f'  {n_done}/{len(idx)}', flush=True)

    fem_B, op_B, rel = np.array(fem_B), np.array(op_B), np.array(rel)
    n = len(fem_B)
    print(f'\n实解完成 n = {n}')

    base = json.load(open(os.path.join(HERE, 'random_land_baseline.json'),
                          encoding='utf-8'))
    p0, n0 = base['frac_B_gt1'] / 100.0, base['n']
    se0 = np.sqrt(p0 * (1 - p0) / n0)

    p_fem = float((fem_B > 1).mean())
    p_op = float((op_B > 1).mean())
    se_fem = np.sqrt(p_fem * (1 - p_fem) / n)

    print()
    print('=== B>1 占比 ===')
    print(f'  随机陆地参照（有限元, n={n0:,}）  {100*p0:.2f}%  ± {100*se0:.2f}')
    print(f'  测点（有限元, n={n}）            {100*p_fem:.2f}%  ± {100*se_fem:.2f}')
    print(f'  测点（算子, 同一批点）           {100*p_op:.2f}%')
    print(f'  测点（算子, 全部 36,392 点）     {100*float((B_op_all>1).mean()):.2f}%'
          '   ← 稿件现用')

    se = np.sqrt(se0 ** 2 + se_fem ** 2)
    z = (p_fem - p0) / se
    p = 2 * stats.norm.sf(abs(z))
    print()
    print('=== 同一估计器（两侧均为有限元）===')
    print(f'  超出 {100*(p_fem-p0):+.2f} 个百分点   z = {z:.3f}   p = {p:.3g}')
    print(f'  对比稿件（跨估计器）: +6.79 pp')

    # 近平坦区间的符号一致率 —— 稿件从未报告
    print()
    print('=== 算子—有限元符号一致率（按 |B_fem − 1| 分档）===')
    ab = np.abs(fem_B - 1.0)
    agree_rows = []
    for lo, hi, k in [(0, .02, '<0.02 近平坦'), (.02, .10, '0.02-0.10'),
                      (.10, 9, '>=0.10')]:
        m = (ab >= lo) & (ab < hi)
        if m.sum() == 0:
            continue
        ag = float((np.sign(fem_B[m] - 1) == np.sign(op_B[m] - 1)).mean())
        print(f'  {k:<14} n={int(m.sum()):>4}   一致率 {100*ag:.1f}%')
        agree_rows.append(dict(band=k, n=int(m.sum()),
                               sign_agreement_pct=round(100 * ag, 1)))

    out = dict(n_sites_solved=n, baseline_pct=round(100 * p0, 2), baseline_n=int(n0),
               sites_fem_pct=round(100 * p_fem, 2), sites_op_pct=round(100 * p_op, 2),
               sites_op_full_pct=round(100 * float((B_op_all > 1).mean()), 2),
               excess_same_estimator_pp=round(100 * (p_fem - p0), 2),
               z=round(float(z), 3), p=float(f'{p:.3g}'),
               sign_agreement=agree_rows,
               note='两侧均由有限元求得,消除 R1-M7 指出的跨估计器污染')
    with open(os.path.join(HERE, 'valley_same_estimator_test.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print('\n已保存 valley_same_estimator_test.json')


if __name__ == '__main__':
    main()
