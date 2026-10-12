# -*- coding: utf-8 -*-
"""全部测点的逐点算子不确定度:8 朝向离散 + 蒙特卡洛 dropout(合成)。

mc_d4_calibration.py 在两个独立集(1,000 块,有有限元真值)上显示:合成离散度
sqrt(朝向标准差² + dropout 标准差²) 与实际误差的秩相关 0.70–0.76,不缩放时 ±1σ 覆盖
66–67%,在一个集上定的缩放系数(≈1.05)迁移到另一集后 ±1σ/±2σ 覆盖 68–69% / 91–92%。
据此对 36,392 个测点全量计算。瓦片读取与生产流程同一入口(tile_reader.read_tile)。
每 2,000 个测点存一次断点(mc_d4_full.partial.npz),中断后重跑自动续上。
输出 mc_d4_uncertainty.npz:B_s_d4(8 向平均,eval)、orient_sd、dropout_sd、combined_sd(B_s 单位,未缩放)。
"""
import os
import sys
import time

import numpy as np
import rasterio
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step3_train_large import FNO3D, D4Average  # noqa: E402
from tile_reader import read_tile  # noqa: E402

DEM_PATH = os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif'))
K, BATCH, C, SAVE_EVERY = 20, 64, 20, 2000
PART = os.path.join(HERE, 'mc_d4_full.partial.npz')
torch.manual_seed(20260929)

d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
lon, lat, Bs_ref = d['lon'], d['lat'], d['B_surface']
n = len(lon)
ck = torch.load(os.path.join(HERE, 'fno3d_large_model.pt'), map_location='cuda', weights_only=False)
net = FNO3D(modes=24, width=96, H_out=41, W_out=41, n_layers=6).cuda()
net.load_state_dict(ck['state'])

out = {k: np.full(n, np.nan, np.float32) for k in ('B_s_d4', 'orient_sd', 'dropout_sd', 'combined_sd')}
start = 0
if os.path.exists(PART):
    z = np.load(PART)
    for k in out:
        out[k] = z[k]
    start = int(z['done'])
    print('续跑:已完成 %d / %d' % (start, n), flush=True)


def infer(x):
    Pe = np.zeros((len(x), 8)); s1 = np.zeros(len(x)); s2 = np.zeros(len(x))
    with torch.no_grad():
        for k in range(8):
            xk = D4Average._fwd(x, k)
            net.eval()
            Pe[:, k] = D4Average._inv(net(xk), k)[:, 0, C, C].cpu().numpy() * ck['Y_std'] + ck['Y_mean']
            net.train()
            for _ in range(K):
                y = D4Average._inv(net(xk), k)[:, 0, C, C].cpu().numpy() * ck['Y_std'] + ck['Y_mean']
                s1 += y; s2 += y * y
    net.eval()
    m = 8 * K
    dsd = np.sqrt(np.maximum(s2 / m - (s1 / m) ** 2, 0))
    osd = Pe.std(1, ddof=1)
    return Pe.mean(1), osd, dsd, np.hypot(osd, dsd)


t0 = time.time()
ds = rasterio.open(DEM_PATH)
idx_buf, dem_buf = [], []
for i in range(start, n):
    dem = read_tile(ds, float(lon[i]), float(lat[i]))
    if dem is not None:
        idx_buf.append(i); dem_buf.append(dem.astype(np.float32))
    if len(dem_buf) >= BATCH or i == n - 1:
        if dem_buf:
            x = torch.tensor((np.stack(dem_buf)[:, None] - ck['X_mean']) / ck['X_std'], dtype=torch.float32).cuda()
            r = infer(x)
            for k, v in zip(('B_s_d4', 'orient_sd', 'dropout_sd', 'combined_sd'), r):
                out[k][idx_buf] = v
        idx_buf, dem_buf = [], []
    if (i + 1) % SAVE_EVERY == 0 or i == n - 1:
        np.savez(PART, done=i + 1, **out)
        el = time.time() - t0
        print('  %d / %d  %.1f min,预计还需 %.1f min' % (i + 1, n, el / 60, el / (i + 1 - start) * (n - i - 1) / 60), flush=True)

ok = np.isfinite(out['B_s_d4'])
dev = np.abs(out['B_s_d4'][ok] - Bs_ref[ok])
print('自检:与 rebuild_points 的 B_surface 最大差 %.2e(应≈0,同为 8 向平均)' % dev.max(), flush=True)
np.savez_compressed(os.path.join(HERE, 'mc_d4_uncertainty.npz'), lon=lon, lat=lat, **out)
os.remove(PART)
print('已保存 mc_d4_uncertainty.npz,用时 %.1f h' % ((time.time() - t0) / 3600), flush=True)
