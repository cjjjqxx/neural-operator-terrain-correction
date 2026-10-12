# -*- coding: utf-8 -*-
"""空间留出集(fold A,聚类 {0,4,8})上,单朝向与 8 朝向平均推理的 R² —— 与 8 向平均的独立集同口径比较。

划分与 step3_train_large.py 完全相同(k-means 12 类,random_state=42,坐标为瓦片中心经纬度)。
单朝向结果须复现 fno3d_large_metrics.json 的 r2_test,作为划分正确的自检。
输出 holdout_d4_r2.json。
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step3_train_large import FNO3D, D4Average, compute_r2  # noqa: E402

d = np.load(os.path.join(HERE, 'global_fem_dataset.npz'))
lab = KMeans(n_clusters=12, random_state=42, n_init='auto').fit_predict(np.stack([d['lon'], d['lat']], 1))
te = np.where(np.isin(lab, [0, 4, 8]))[0]
X = d['X'][te].astype(np.float32)
Y = np.clip(d['Y'][te].astype(np.float32), 0.05, 5.0)
ck = torch.load(os.path.join(HERE, 'fno3d_large_model.pt'), map_location='cuda', weights_only=False)
net = FNO3D(modes=24, width=96, H_out=41, W_out=41, n_layers=6).cuda().eval()
net.load_state_dict(ck['state'])
out = {'n_test': int(len(te))}
for name, m in (('single', net), ('d4', D4Average(net).eval())):
    P = []
    with torch.no_grad():
        for s in range(0, len(X), 64):
            x = (X[s:s + 64].reshape(-1, 1, 128, 128) - ck['X_mean']) / ck['X_std']
            P.append(m(torch.tensor(x).cuda()).cpu().numpy()[:, 0] * ck['Y_std'] + ck['Y_mean'])
    P = np.concatenate(P)
    out['r2_' + name] = compute_r2(P, Y.reshape(P.shape))
    out['mae_' + name] = float(np.abs(P - Y.reshape(P.shape)).mean())
    c = P.shape[-1] // 2
    out['r2_center_' + name] = compute_r2(P[:, c, c], Y.reshape(P.shape)[:, c, c])
ref = json.load(open(os.path.join(HERE, 'fno3d_large_metrics.json'), encoding='utf-8'))['r2_test']
out['r2_test_archived'] = ref
print('留出集 n=%d:单朝向 R² %.4f(存档 %.4f),8 向平均 %.4f;中心像元 %.4f → %.4f' % (
    len(te), out['r2_single'], ref, out['r2_d4'], out['r2_center_single'], out['r2_center_d4']))
assert abs(out['r2_single'] - ref) < 0.003, '划分未复现存档的 R²'
json.dump(out, open(os.path.join(HERE, 'holdout_d4_r2.json'), 'w', encoding='utf-8'), indent=2)
