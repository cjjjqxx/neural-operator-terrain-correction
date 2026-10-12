# -*- coding: utf-8 -*-
"""逐点不确定度指标的校准试验:8 朝向离散 + 蒙特卡洛 dropout,能否作为逐测点误差棒?

"""
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step3_train_large import FNO3D, D4Average  # noqa: E402

K = 20
SEED = 20260929
torch.manual_seed(SEED)
ck = torch.load(os.path.join(HERE, 'fno3d_large_model.pt'), map_location='cuda', weights_only=False)
net = FNO3D(modes=24, width=96, H_out=41, W_out=41, n_layers=6).cuda()
net.load_state_dict(ck['state'])
C = 20


def run(dems):
    """返回每块瓦片中心像元:8 向平均(eval)、朝向标准差、dropout 总标准差。"""
    x_all = (dems[:, None].astype(np.float32) - ck['X_mean']) / ck['X_std']
    Pe = np.zeros((len(dems), 8)); Pd_sum = np.zeros(len(dems)); Pd_sq = np.zeros(len(dems))
    with torch.no_grad():
        for s in range(0, len(dems), 50):
            x = torch.tensor(x_all[s:s + 50]).cuda()
            for k in range(8):
                xk = D4Average._fwd(x, k)
                net.eval()
                y = D4Average._inv(net(xk), k)[:, 0, C, C].cpu().numpy() * ck['Y_std'] + ck['Y_mean']
                Pe[s:s + 50, k] = y
                net.train()                                 # 只打开 dropout;组归一化不受影响
                for _ in range(K):
                    y = D4Average._inv(net(xk), k)[:, 0, C, C].cpu().numpy() * ck['Y_std'] + ck['Y_mean']
                    Pd_sum[s:s + 50] += y; Pd_sq[s:s + 50] += y * y
    net.eval()
    n = 8 * K
    drop_sd = np.sqrt(np.maximum(Pd_sq / n - (Pd_sum / n) ** 2, 0))
    return Pe.mean(1), Pe.std(1, ddof=1), drop_sd


def scale_for(sd, e, q=0.683):
    return float(np.quantile(e / np.maximum(sd, 1e-9), q))


def main():
    fo = np.load(os.path.join(HERE, 'fem_orientation_test.npz'))
    v = np.load(os.path.join(HERE, 'validation_independent_500.npz'))
    ex = np.load(os.path.join(HERE, 'validation_independent_excl.npz'))
    sets = {'原独立集': (v['dems'], fo['F'][:, :, C, C].mean(1)),
            '排除距离集': (ex['dems'], ex['B_fem'][:, C, C])}
    res, cache = {}, {}
    for name, (dems, F) in sets.items():
        P, o_sd, d_sd = run(dems)
        e = np.abs(P - F)
        cache[name] = dict(e=e, orient=o_sd, dropout=d_sd, combined=np.hypot(o_sd, d_sd))
        res[name] = dict(n=int(len(e)), rmse=float(np.sqrt(np.mean(e ** 2))))
        for ind in ('orient', 'dropout', 'combined'):
            sd = cache[name][ind]
            res[name][ind] = dict(spearman=float(stats.spearmanr(sd, e)[0]),
                                  ratio_rmse_to_rms_sd=float(np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(sd ** 2))),
                                  cov1_raw=float(np.mean(e <= sd)), cov2_raw=float(np.mean(e <= 2 * sd)),
                                  c68=scale_for(sd, e))
    names = list(sets)
    for a, b in ((names[0], names[1]), (names[1], names[0])):
        for ind in ('orient', 'dropout', 'combined'):
            c = res[a][ind]['c68']
            sd, e = cache[b][ind], cache[b]['e']
            res[b][ind]['transfer_from_' + a] = dict(c=c, cov1=float(np.mean(e <= c * sd)),
                                                     cov2=float(np.mean(e <= 2 * c * sd)))
    for name in names:
        print('== %s(n=%d,中心像元 RMSE %.4f)' % (name, res[name]['n'], res[name]['rmse']))
        for ind in ('orient', 'dropout', 'combined'):
            r = res[name][ind]
            t = [v2 for k2, v2 in r.items() if k2.startswith('transfer_from_')][0]
            print('  %-9s Spearman %.2f  误差/离散 %.2f 倍  原始覆盖 %.0f%%/%.0f%%  用另一集定的 c=%.2f 迁移后覆盖 %.0f%%/%.0f%%'
                  % (ind, r['spearman'], r['ratio_rmse_to_rms_sd'], 100 * r['cov1_raw'], 100 * r['cov2_raw'],
                     t['c'], 100 * t['cov1'], 100 * t['cov2']))
    json.dump(res, open(os.path.join(HERE, 'mc_d4_calibration.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('已保存 mc_d4_calibration.json')


if __name__ == '__main__':
    main()
