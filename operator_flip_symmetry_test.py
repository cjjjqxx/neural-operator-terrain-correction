# -*- coding: utf-8 -*-
"""算子对地形翻转/旋转的一致性 —— 单象限谱卷积会不会带来方向偏差。

网络的谱卷积只对非负 k_y 一侧的低频模态施加可学习权重,负 k_y 一侧原样通过
(step3_train_large.py 的 FNO3D)。物理问题对瓦片的翻转与旋转是对称的:把地形上下翻转,
真实的 B 场也只是跟着翻转。因此可以直接检验:
    P_flip = 翻转回来的 pred(翻转后的地形)  与  P = pred(地形)  之差
若这种差异远小于算子相对有限元的误差,方向偏差就不构成实际问题。

在独立 500 瓦片集上,对上下翻转、左右翻转、旋转 90°/180°/270° 各做一次,
报告中心像元与全场的差异,并与同一批瓦片上算子相对有限元的误差对照。
输出 operator_flip_symmetry_test.json。
"""
import json
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step3_train_large import FNO3D  # noqa: E402


def main():
    ck = torch.load(os.path.join(HERE, 'fno3d_large_model.pt'), map_location='cuda', weights_only=False)
    m = FNO3D(modes=24, width=96, H_out=41, W_out=41, n_layers=6).cuda().eval()
    m.load_state_dict(ck['state'])
    v = np.load(os.path.join(HERE, 'validation_independent_500.npz'))
    dems, Bf = v['dems'].astype(np.float32), v['B_fem'].astype(float)

    def pred(x):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 50):
                z = (x[i:i + 50, None] - ck['X_mean']) / ck['X_std']
                y = m(torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32)).cuda()).cpu().numpy()
                out.append(y[:, 0] * ck['Y_std'] + ck['Y_mean'])
        return np.concatenate(out).astype(float)

    P = pred(dems)
    c = 20
    base_c = float(np.sqrt(np.mean((P[:, c, c] - Bf[:, c, c]) ** 2)))
    base_f = float(np.sqrt(np.mean((P - Bf) ** 2)))
    ops = {
        '上下翻转': (lambda a: a[:, ::-1, :], lambda a: a[:, ::-1, :]),
        '左右翻转': (lambda a: a[:, :, ::-1], lambda a: a[:, :, ::-1]),
        '旋转90°': (lambda a: np.rot90(a, 1, axes=(1, 2)), lambda a: np.rot90(a, -1, axes=(1, 2))),
        '旋转180°': (lambda a: np.rot90(a, 2, axes=(1, 2)), lambda a: np.rot90(a, -2, axes=(1, 2))),
        '旋转270°': (lambda a: np.rot90(a, 3, axes=(1, 2)), lambda a: np.rot90(a, -3, axes=(1, 2))),
    }
    out = dict(n=int(len(dems)), op_vs_fem_rmse_center=base_c, op_vs_fem_rmse_field=base_f, transforms={})
    print('算子相对有限元:中心像元 RMSE %.4f,全场 RMSE %.4f(独立 500 瓦片)' % (base_c, base_f))
    all_c = [P[:, c, c]]
    for nm, (fw, bw) in ops.items():
        Q = bw(pred(np.ascontiguousarray(fw(dems))))
        dc = Q[:, c, c] - P[:, c, c]
        df = Q - P
        sign_flip = float(100 * np.mean(np.sign(Q[:, c, c] - 1) != np.sign(P[:, c, c] - 1)))
        qc = float(np.sqrt(np.mean((Q[:, c, c] - Bf[:, c, c]) ** 2)))
        out['transforms'][nm] = dict(rms_center_diff=float(np.sqrt(np.mean(dc ** 2))),
                                     rms_field_diff=float(np.sqrt(np.mean(df ** 2))),
                                     center_sign_change_pct=sign_flip, rmse_center_vs_fem=qc)
        all_c.append(Q[:, c, c])
        print('  %-7s 中心像元差 RMS %.4f(%.0f%% 的算子误差),全场 %.4f,中心符号改变 %.1f%%,变换后对有限元 RMSE %.4f'
              % (nm, np.sqrt(np.mean(dc ** 2)), 100 * np.sqrt(np.mean(dc ** 2)) / base_c,
                 np.sqrt(np.mean(df ** 2)), sign_flip, qc))
    ens = np.mean(all_c, axis=0)
    out['ensemble6_rmse_center_vs_fem'] = float(np.sqrt(np.mean((ens - Bf[:, c, c]) ** 2)))
    print('六种朝向取平均后的中心像元 RMSE %.4f(单一朝向 %.4f)'
          % (out['ensemble6_rmse_center_vs_fem'], base_c))
    json.dump(out, open(os.path.join(HERE, 'operator_flip_symmetry_test.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('已保存 operator_flip_symmetry_test.json')


if __name__ == '__main__':
    main()
