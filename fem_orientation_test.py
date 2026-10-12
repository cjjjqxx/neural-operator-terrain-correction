# -*- coding: utf-8 -*-
"""有限元参考解的方向误差,以及算子在对称参考下的精度。

物理问题对瓦片的旋转与翻转是对称的:地形转 90°,B 场也只是跟着转 90°。但生产有限元
把每个六面体按固定朝向的对角线切成 6 个四面体,离散误差因此带有方向性 —— 同一块地形
换个朝向求解,中心像元的 B 会变。这部分误差是正弦地形的解析基准(只沿一个方向起伏)
测不出来的。

对独立 500 瓦片集的每块瓦片,在二面体群 D4 的 8 个朝向各求一次生产配置有限元,
变换回原朝向后:
  · 8 个值的离散 → 有限元方向误差(按 |B−1| 分档,作为误差预算的实测项)
  · 8 个值的平均 → 对称参考解(方向误差已平均掉)
再用它评价算子:单一朝向预测,以及 8 个朝向预测的平均。
输出 fem_orientation_test.json 与 fem_orientation_test.npz(中心像元与全场)。
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))


def d4(a, k):
    """二面体群第 k 个元素(k = 0..7):先按需上下翻转,再旋转 (k//2)·90°。"""
    b = a[::-1, :] if k % 2 else a
    return np.rot90(b, k // 2)


def d4_inv(a, k):
    b = np.rot90(a, -(k // 2))
    return b[::-1, :] if k % 2 else b


def solve(args):
    i, k, dem = args
    sys.path.insert(0, HERE)
    from fem3d import Fem3D
    from tile_reader import CELL_M
    z = np.ascontiguousarray(d4(dem, k)).astype(float)
    r = float(z.max() - z.min())
    f = Fem3D(nx=40, ny=40, nz=10)
    f.build(z, CELL_M, depth=max(3000.0, 1.5 * r))
    f.solve_heat()
    f.compute_B()
    return i, k, np.ascontiguousarray(d4_inv(f.B_grid, k)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=10)
    a = ap.parse_args()
    v = np.load(os.path.join(HERE, 'validation_independent_500.npz'))
    dems, rl = v['dems'].astype(np.float32), v['reliefs']
    n = len(dems)
    tasks = [(i, k, dems[i]) for i in range(n) for k in range(8)]
    F = np.zeros((n, 8, 41, 41), np.float32)
    t0 = time.time()
    with Pool(a.procs) as p:
        for j, (i, k, G) in enumerate(p.imap_unordered(solve, tasks, chunksize=4), 1):
            F[i, k] = G
            if j % 500 == 0:
                el = time.time() - t0
                print('  %d/%d  %.1f min,预计还需 %.1f min' % (j, len(tasks), el / 60,
                                                           el / j * (len(tasks) - j) / 60), flush=True)
    # 核对:朝向 0 应与存档的独立集有限元一致
    chk = float(np.abs(F[:, 0] - v['B_fem']).max())
    print('核对:朝向 0 与存档 B_fem 最大差 %.2e' % chk)

    # 算子:单一朝向与 8 向平均
    import torch
    sys.path.insert(0, HERE)
    from step3_train_large import FNO3D
    ck = torch.load(os.path.join(HERE, 'fno3d_large_model.pt'), map_location='cuda', weights_only=False)
    m = FNO3D(modes=24, width=96, H_out=41, W_out=41, n_layers=6).cuda().eval()
    m.load_state_dict(ck['state'])

    def pred(x):
        out = []
        with torch.no_grad():
            for s in range(0, len(x), 50):
                zz = (x[s:s + 50, None] - ck['X_mean']) / ck['X_std']
                y = m(torch.from_numpy(np.ascontiguousarray(zz, dtype=np.float32)).cuda()).cpu().numpy()
                out.append(y[:, 0] * ck['Y_std'] + ck['Y_mean'])
        return np.concatenate(out)
    P = np.stack([np.stack([d4_inv(q, k) for q in pred(np.stack([d4(d, k) for d in dems]))])
                  for k in range(8)], axis=1).astype(np.float32)          # (n, 8, 41, 41)

    c = 20
    Fc, Pc = F[:, :, c, c].astype(float), P[:, :, c, c].astype(float)
    F0, Fs = Fc[:, 0], Fc.mean(1)                     # 单一朝向有限元 / 对称参考
    P0, Ps = Pc[:, 0], Pc.mean(1)                     # 单一朝向算子 / 8 向平均
    fem_sd = Fc.std(1, ddof=1)                        # 每块瓦片有限元 8 个朝向的标准差
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

    def r2(p, t):
        return float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())

    out = dict(n=n, check_orientation0_vs_archive=chk, bands={})
    ab = np.abs(Fs - 1)
    print('\n══ 有限元方向误差(中心像元,8 个朝向)══')
    for lo, hi, lab in ((0, .02, '<0.02'), (.02, .05, '0.02-0.05'), (.05, .10, '0.05-0.10'),
                        (.10, 9, '>=0.10')):
        mk = (ab >= lo) & (ab < hi)
        row = dict(n=int(mk.sum()),
                   fem_orient_sd_median=float(np.median(fem_sd[mk])),
                   fem_orient_sd_rms=float(np.sqrt(np.mean(fem_sd[mk] ** 2))),
                   fem_orient_rel_median=float(np.median(fem_sd[mk] / np.maximum(ab[mk], 1e-6))),
                   single_fem_vs_sym_rmse=rmse(F0[mk], Fs[mk]),
                   op1_vs_fem1_rmse=rmse(P0[mk], F0[mk]), op1_vs_sym_rmse=rmse(P0[mk], Fs[mk]),
                   op8_vs_sym_rmse=rmse(Ps[mk], Fs[mk]),
                   op1_bias_vs_sym=float(np.mean(P0[mk] - Fs[mk])), op8_bias_vs_sym=float(np.mean(Ps[mk] - Fs[mk])),
                   sign_agree_op8_sym=float(100 * np.mean(np.sign(Ps[mk] - 1) == np.sign(Fs[mk] - 1))))
        out['bands'][lab] = row
        print('  |B−1| %-10s n=%3d  有限元朝向标准差 中位 %.4f(相对 |B−1| %.0f%%)  '
              '算子(单向)对单向有限元 %.4f / 对对称参考 %.4f;算子(8 向)对对称参考 %.4f'
              % (lab, mk.sum(), row['fem_orient_sd_median'], 100 * row['fem_orient_rel_median'],
                 row['op1_vs_fem1_rmse'], row['op1_vs_sym_rmse'], row['op8_vs_sym_rmse']))
    out['overall'] = dict(
        r2_field_op1_vs_fem1=r2(P[:, 0], F[:, 0]), r2_field_op8_vs_sym=r2(P.mean(1), F.mean(1)),
        rmse_center_op1_vs_fem1=rmse(P0, F0), rmse_center_op8_vs_sym=rmse(Ps, Fs),
        rmse_center_fem1_vs_sym=rmse(F0, Fs),
        frac_gt1_fem1=float(100 * np.mean(F0 > 1)), frac_gt1_sym=float(100 * np.mean(Fs > 1)),
        frac_gt1_op1=float(100 * np.mean(P0 > 1)), frac_gt1_op8=float(100 * np.mean(Ps > 1)))
    print('\n全场 R²:算子单向对单向有限元 %.4f;算子 8 向对对称参考 %.4f'
          % (out['overall']['r2_field_op1_vs_fem1'], out['overall']['r2_field_op8_vs_sym']))
    print('中心 RMSE:单向有限元相对对称参考 %.4f;算子单向对单向有限元 %.4f;算子 8 向对对称参考 %.4f'
          % (out['overall']['rmse_center_fem1_vs_sym'], out['overall']['rmse_center_op1_vs_fem1'],
             out['overall']['rmse_center_op8_vs_sym']))
    print('B > 1 占比:有限元单向 %.1f / 对称 %.1f;算子单向 %.1f / 8 向 %.1f'
          % tuple(out['overall'][k] for k in ('frac_gt1_fem1', 'frac_gt1_sym', 'frac_gt1_op1', 'frac_gt1_op8')))
    for lo, hi in ((0, 100), (100, 500), (500, 1e9)):
        mk = (rl >= lo) & (rl < hi)
        out['relief_%d' % lo] = dict(n=int(mk.sum()), op8_bias_vs_sym=float(np.mean(Ps[mk] - Fs[mk])),
                                     op1_bias_vs_fem1=float(np.mean(P0[mk] - F0[mk])),
                                     op8_rmse_vs_sym=rmse(Ps[mk], Fs[mk]))
        print('起伏 %4.0f–%-5.0f 算子偏差:单向对单向有限元 %+.5f;8 向对对称参考 %+.5f'
              % (lo, min(hi, 9999), out['relief_%d' % lo]['op1_bias_vs_fem1'],
                 out['relief_%d' % lo]['op8_bias_vs_sym']))
    np.savez_compressed(os.path.join(HERE, 'fem_orientation_test.npz'), F=F, P=P, reliefs=rl)
    json.dump(out, open(os.path.join(HERE, 'fem_orientation_test.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('\n已保存 fem_orientation_test.json / .npz,用时 %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
