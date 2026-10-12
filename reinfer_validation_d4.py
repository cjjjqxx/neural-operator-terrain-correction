# -*- coding: utf-8 -*-
"""V10:两个独立集的算子预测改为 8 朝向平均(D4Average);有限元参考值与瓦片不变,不重算有限元。

更新 validation_independent_500.npz 的 B_fno / B_fno_center / per_tile_r2 / r2_total,
以及 validation_independent_excl.npz 的 B_fno 和 validation_independent_excl.json 的算子指标。
原文件先由编排脚本归档到 _归档/单朝向算子_20260929/。
"""
import json
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step3_train_large import FNO3D, D4Average  # noqa: E402

ck = torch.load(os.path.join(HERE, 'fno3d_large_model.pt'), map_location='cuda', weights_only=False)
net = FNO3D(modes=24, width=96, H_out=41, W_out=41, n_layers=6).cuda().eval()
net.load_state_dict(ck['state'])
model = D4Average(net).eval()
print('推理:8 朝向平均(D4Average)')


def predict(dems):
    out = []
    with torch.no_grad():
        for s in range(0, len(dems), 50):
            x = (dems[s:s + 50, None] - ck['X_mean']) / ck['X_std']
            y = model(torch.tensor(np.ascontiguousarray(x), dtype=torch.float32).cuda()).cpu().numpy()
            out.append(y[:, 0] * ck['Y_std'] + ck['Y_mean'])
    return np.concatenate(out).astype(np.float32)


def r2(p, t):
    p, t = np.asarray(p, float).ravel(), np.asarray(t, float).ravel()
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())


c = 20
# ── 原独立集 ──
p = os.path.join(HERE, 'validation_independent_500.npz')
v = dict(np.load(p))
old = v['B_fno'].copy()
v['B_fno'] = predict(v['dems'].astype(np.float32))
v['B_fno_center'] = v['B_fno'][:, c, c]
v['per_tile_r2'] = np.array([r2(v['B_fno'][i], v['B_fem'][i]) for i in range(len(v['dems']))])
v['r2_total'] = r2(v['B_fno'], v['B_fem'])
np.savez(p, **v)
print('原独立集:全场 R² %.4f → %.4f;中心 RMSE %.4f → %.4f' % (
    r2(old, v['B_fem']), v['r2_total'],
    np.sqrt(np.mean((old[:, c, c] - v['B_fem'][:, c, c]) ** 2)),
    np.sqrt(np.mean((v['B_fno'][:, c, c] - v['B_fem'][:, c, c]) ** 2))))

# ── 排除距离集 ──
p = os.path.join(HERE, 'validation_independent_excl.npz')
e = dict(np.load(p))
e['B_fno'] = predict(e['dems'].astype(np.float32))
np.savez_compressed(p, **e)
F, P, Lin, rl = e['B_fem'].astype(float), e['B_fno'].astype(float), e['B_linear_center'].astype(float), e['reliefs']
Fc, Pc = F[:, c, c], P[:, c, c]
jp = os.path.join(HERE, 'validation_independent_excl.json')
out = json.load(open(jp, encoding='utf-8'))
out.update(r2_field=r2(P, F), r2_center=r2(Pc, Fc), rmse_center=float(np.sqrt(np.mean((Pc - Fc) ** 2))),
           bias_center=float(np.mean(Pc - Fc)),
           sign_agree_center=float(100 * np.mean(np.sign(Pc - 1) == np.sign(Fc - 1))),
           operator='D4Average(8 朝向平均)')
for lo, hi, lab in ((0, 100, '<100 m'), (100, 500, '100-500 m'), (500, 1e9, '>500 m')):
    mk = (rl >= lo) & (rl < hi)
    out['by_relief'][lab].update(r2_field=r2(P[mk], F[mk]),
                                 rmse_center=float(np.sqrt(np.mean((Pc[mk] - Fc[mk]) ** 2))))
ab = np.abs(Fc - 1)
for lo, hi, lab in ((0, .02, '<0.02'), (.02, .10, '0.02-0.10'), (.10, 9, '>=0.10')):
    mk = (ab >= lo) & (ab < hi)
    out['by_band'][lab].update(rmse_center=float(np.sqrt(np.mean((Pc[mk] - Fc[mk]) ** 2))),
                               bias_center=float(np.mean(Pc[mk] - Fc[mk])))
json.dump(out, open(jp, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print('排除距离集:全场 R² %.4f;中心 R² %.4f,RMSE %.4f(线性解 %.4f)' % (
    out['r2_field'], out['r2_center'], out['rmse_center'], out['linear_rmse_center']))
