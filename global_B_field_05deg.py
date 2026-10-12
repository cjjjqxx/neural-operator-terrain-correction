# -*- coding: utf-8 -*-
"""用训练好的算子给全球每个陆地 0.5° 格点算地形传递因子 B。


算子真正做得到而有限元做不到的,是把校正从**逐点服务**变成**可查的场**:
  · 逐测点(已做)      36,392 点   FEM 22 h    算子 4.7 min
  · 全球陆地 0.5°(本脚本) 约 6.7 万格 FEM 40 h    算子 约 9 min
  · 0.1° 加密           约 167 万格 FEM 42 天   算子 3.6 h

产出的场是一份可直接发布的数据产品:任何人手上有热流测点,查这张表即可做
地形校正,不需要安装有限元环境、不需要自己划网格。这也是把本文从
"我们校正了一批测点"推进到"我们提供了一个人人可用的校正场"的那一步。

口径与逐测点流程完全一致:57 km 窗口、128×128 输入、取 41×41 输出场的中心像元,
归一化常量取自模型检查点。此处只给**地表**传递因子 B_s;
深度平均需要各测点自己的钻孔区间深度,由使用者按式(4)(5)施加。
"""
import io
import json
import os
import sys
import time

import numpy as np
import rasterio
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tile_reader import read_tile                            # noqa: E402
from step3_train_large import FNO3D                          # noqa: E402

DEM_PATH = os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif'))
FN = 40


def extract_dem(ds, lon, lat):
    try:
        dem = read_tile(ds, lon, lat)
    except Exception:
        return None
    return None if dem is None else dem.astype(np.float32)

CKPT = os.path.join(HERE, 'fno3d_large_model.pt')
OUT = os.path.join(HERE, 'global_B_field_05deg.npz')
BATCH = 128

# 陆地格点取自克里金场的有效掩膜 —— 与正文"只统计陆地格点"的口径一致
kr = np.load(os.path.join(HERE, 'kriged_heatflow_05deg_global.npz'))
lats, lons = kr['lats'], kr['lons']
land = np.isfinite(kr['krige_qm'])
jj, ii = np.nonzero(land)
print('陆地格点 %s 个' % format(len(jj), ','))

ck = torch.load(CKPT, map_location='cuda', weights_only=False)
model = FNO3D(modes=24, width=96, H_out=FN + 1, W_out=FN + 1, n_layers=6).cuda().eval()
model.load_state_dict(ck['state'])
from step3_train_large import D4Average                      # noqa: E402
model = D4Average(model).eval()
print('推理:8 朝向平均(D4Average)')
X_m, X_s, Y_m, Y_s = ck['X_mean'], ck['X_std'], ck['Y_mean'], ck['Y_std']

Bfield = np.full(land.shape, np.nan, dtype=np.float32)
relief = np.full(land.shape, np.nan, dtype=np.float32)
ds = rasterio.open(DEM_PATH)
c = FN // 2
buf_dem, buf_ij = [], []
t0 = time.time()
n_ok = 0


def flush():
    global n_ok
    if not buf_dem:
        return
    xb = (np.stack(buf_dem).reshape(-1, 1, 128, 128) - X_m) / X_s
    with torch.no_grad():
        pred = model(torch.tensor(xb, dtype=torch.float32).cuda())
    Bv = pred.cpu().numpy()[:, 0, c, c] * Y_s + Y_m
    for k, (j, i) in enumerate(buf_ij):
        Bfield[j, i] = Bv[k]
        relief[j, i] = buf_dem[k].max() - buf_dem[k].min()
    n_ok += len(buf_ij)
    buf_dem.clear()
    buf_ij.clear()


for n, (j, i) in enumerate(zip(jj, ii)):
    dem = extract_dem(ds, float(lons[i]), float(lats[j]))
    if dem is not None:
        buf_dem.append(dem)
        buf_ij.append((j, i))
    if len(buf_dem) >= BATCH:
        flush()
    if (n + 1) % 5000 == 0:
        el = time.time() - t0
        print('  %s/%s  已算 %s  用时 %.1f min  预计总计 %.1f min'
              % (format(n + 1, ','), format(len(jj), ','), format(n_ok, ','),
                 el / 60, el / 60 * len(jj) / (n + 1)))
flush()
ds.close()

np.savez_compressed(OUT, B=Bfield, relief=relief, lats=lats, lons=lons)
ok = np.isfinite(Bfield)
res = dict(
    n_land_cells=int(land.sum()), n_solved=int(ok.sum()),
    minutes=round((time.time() - t0) / 60, 1),
    B_median=round(float(np.median(Bfield[ok])), 4),
    B_mean=round(float(Bfield[ok].mean()), 4),
    B_p1=round(float(np.percentile(Bfield[ok], 1)), 4),
    B_p99=round(float(np.percentile(Bfield[ok], 99)), 4),
    frac_B_gt1_pct=round(100 * float((Bfield[ok] > 1).mean()), 2),
    frac_dev_gt2pct=round(100 * float((np.abs(Bfield[ok] - 1) > 0.02).mean()), 2),
    frac_dev_gt10pct=round(100 * float((np.abs(Bfield[ok] - 1) > 0.10).mean()), 2),
)
json.dump(res, open(os.path.join(HERE, 'global_B_field_05deg.json'), 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1)
print()
print('完成 %s / %s 格点,用时 %.1f 分钟'
      % (format(res['n_solved'], ','), format(res['n_land_cells'], ','), res['minutes']))
print('  B 中位 %.4f  均值 %.4f  1–99 分位 %.3f–%.3f'
      % (res['B_median'], res['B_mean'], res['B_p1'], res['B_p99']))
print('  B>1 占 %.2f%% | |B−1|>0.02 占 %.2f%% | >0.10 占 %.2f%%'
      % (res['frac_B_gt1_pct'], res['frac_dev_gt2pct'], res['frac_dev_gt10pct']))
print('已保存 global_B_field_05deg.npz / .json')
