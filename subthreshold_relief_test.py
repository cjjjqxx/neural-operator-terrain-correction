# -*- coding: utf-8 -*-
"""亚门槛起伏验证 + 中心像素分档误差（预审 C5 / R1-M6）。


R1 另指出一个更细的错配：每个测点校正只用**中心像素**，但所有分档诊断都是
**整块瓦片逐像素**的；中心像素精度只报了一个汇总数（R²=0.932），从未按档拆分，
而进入表 2 的算子项用的是整场分档 RMSE，不是中心像素的。

本脚本做三件事：
  1. 报告应用集 vs 训练集的起伏分布，给出 < 25 m 的占比；
  2. 在应用集里抽取亚门槛测点，实解有限元，测算子中心像素误差；
  3. 用独立 500 瓦片集，按起伏档与 |B−1| 档报告**中心像素** RMSE 与偏差。

有限元单瓦片约 1.64 s，抽样规模默认 200，约 6 分钟。
"""
import os
import sys
import json
import argparse

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def relief_of(dem):
    return float(np.nanmax(dem) - np.nanmin(dem))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200, help='亚门槛抽样瓦片数')
    ap.add_argument('--dem', default=os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif')))
    ap.add_argument('--procs', type=int, default=6)
    a = ap.parse_args()

    out = {}

    # ── 1. 中心像素分档误差（独立 500 瓦片集，无需新求解）──────────────
    v = np.load(os.path.join(HERE, 'validation_independent_500.npz'))
    Bf, Bn, rl = v['B_fem'], v['B_fno'], v['reliefs']
    if Bf.ndim == 4:
        Bf, Bn = Bf[:, 0], Bn[:, 0]
    c = Bf.shape[1] // 2
    cf, cn = Bf[:, c, c], Bn[:, c, c]           # 中心像素
    print('=== 中心像素误差（独立 500 瓦片集）===')
    print(f'{"起伏档":<12}{"n":>5}{"中心RMSE":>11}{"中心偏差":>11}{"整场RMSE":>11}')
    print('-' * 50)
    rows = []
    for lo, hi, k in [(0, 50, '<50m'), (50, 100, '50-100m'), (100, 200, '100-200m'),
                      (200, 500, '200-500m'), (500, 9e9, '>500m')]:
        m = (rl >= lo) & (rl < hi)
        if m.sum() == 0:
            continue
        cr = float(np.sqrt(((cf[m] - cn[m]) ** 2).mean()))
        cb = float((cn[m] - cf[m]).mean())
        fr = float(np.sqrt(((Bf[m] - Bn[m]) ** 2).mean()))
        print(f'{k:<12}{int(m.sum()):>5}{cr:>11.4f}{cb:>+11.4f}{fr:>11.4f}')
        rows.append(dict(band=k, n=int(m.sum()), centre_rmse=round(cr, 5),
                         centre_bias=round(cb, 5), field_rmse=round(fr, 5)))
    out['centre_by_relief'] = rows

    # 按 |B−1| 档（用有限元中心值定义档位，避免用被检验量自身分档）
    ab = np.abs(cf - 1.0)
    print()
    print(f'{"|B-1|档":<12}{"n":>5}{"中心RMSE":>11}{"中心偏差":>11}')
    print('-' * 39)
    rows2 = []
    for lo, hi, k in [(0, .02, '<0.02'), (.02, .05, '0.02-0.05'), (.05, .10, '0.05-0.10'),
                      (.10, .20, '0.10-0.20'), (.20, 9, '>0.20')]:
        m = (ab >= lo) & (ab < hi)
        if m.sum() == 0:
            continue
        cr = float(np.sqrt(((cf[m] - cn[m]) ** 2).mean()))
        cb = float((cn[m] - cf[m]).mean())
        print(f'{k:<12}{int(m.sum()):>5}{cr:>11.4f}{cb:>+11.4f}')
        rows2.append(dict(band=k, n=int(m.sum()), centre_rmse=round(cr, 5),
                          centre_bias=round(cb, 5)))
    out['centre_by_departure'] = rows2

    # ── 2. 应用集起伏分布 vs 训练门槛 ────────────────────────────────
    from tile_reader import read_tile, CELL_M
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    lon, lat = d['lon'], d['lat']
    rng = np.random.default_rng(20260826)
    probe = rng.choice(len(lon), min(1500, len(lon)), replace=False)
    rel = []
    for i in probe:
        try:
            dem = read_tile(a.dem, float(lon[i]), float(lat[i]))
            rel.append(relief_of(dem))
        except Exception:
            continue
    rel = np.array(rel)
    frac_sub = 100.0 * float((rel < 25).mean())
    print()
    print(f'=== 应用集起伏分布（{len(rel)} 点抽样）===')
    print(f'  中位 {np.median(rel):.0f} m   四分位 [{np.percentile(rel,25):.0f}, '
          f'{np.percentile(rel,75):.0f}]   最小 {rel.min():.0f}')
    print(f'  **起伏 < 25 m（训练门槛）占比: {frac_sub:.1f}%**')
    out['application_relief'] = dict(n_probe=int(len(rel)),
                                     median=round(float(np.median(rel)), 1),
                                     frac_below_25m_pct=round(frac_sub, 2))

    # ── 3. 亚门槛测点上实解有限元，测算子中心像素误差 ─────────────────
    sub_idx = [int(probe[j]) for j, r in enumerate(rel) if r < 25]
    if not sub_idx:
        print('\n抽样中没有亚门槛测点,跳过第 3 步')
    else:
        take = sub_idx[:a.n]
        print(f'\n=== 亚门槛测点有限元验证（{len(take)} 点）===')
        from mesh_resolution_test import load_tile  # 与主流程同一读取入口
        from fem3d import Fem3D
        import torch
        from step3_train_large import FNO3D

        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = FNO3D(modes=24, width=96, n_layers=6).to(dev)
        model.load_state_dict(torch.load(os.path.join(HERE, 'fno3d_large_model.pt'),
                                         map_location=dev))
        model.eval()

        fem_v, op_v = [], []
        for n_done, i in enumerate(take, 1):
            try:
                dem = read_tile(a.dem, float(lon[i]), float(lat[i]))
                r = relief_of(dem)
                f = Fem3D(nx=40, ny=40, nz=10)
                f.build(dem, CELL_M, depth=max(3000.0, 1.5 * r))
                f.solve_heat(); f.compute_B()
                G = f.B_grid
                fem_v.append(float(G[G.shape[0] // 2, G.shape[0] // 2]))
                with torch.no_grad():
                    x = torch.tensor(dem[None, None], dtype=torch.float32, device=dev)
                    y = model(x).cpu().numpy()[0, 0]
                op_v.append(float(y[y.shape[0] // 2, y.shape[0] // 2]))
            except Exception as e:
                print(f'  跳过 {i}: {e}')
            if n_done % 50 == 0:
                print(f'  {n_done}/{len(take)}', flush=True)
        fem_v, op_v = np.array(fem_v), np.array(op_v)
        if len(fem_v):
            err = op_v - fem_v
            print(f'  n = {len(fem_v)}')
            print(f'  有限元 |B-1| 中位 {np.median(np.abs(fem_v-1)):.5f}')
            print(f'  算子   |B-1| 中位 {np.median(np.abs(op_v-1)):.5f}')
            print(f'  **中心像素 RMSE {np.sqrt((err**2).mean()):.5f}   '
                  f'偏差 {err.mean():+.5f}**')
            out['subthreshold'] = dict(
                n=int(len(fem_v)),
                fem_median_absB1=round(float(np.median(np.abs(fem_v - 1))), 5),
                op_median_absB1=round(float(np.median(np.abs(op_v - 1))), 5),
                centre_rmse=round(float(np.sqrt((err ** 2).mean())), 5),
                centre_bias=round(float(err.mean()), 5))

    with open(os.path.join(HERE, 'subthreshold_relief_test.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print('\n已保存 subthreshold_relief_test.json')


if __name__ == '__main__':
    main()
