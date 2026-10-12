# -*- coding: utf-8 -*-
"""发布脚本的回归测试:它算出来的必须就是论文里那个数。

为什么需要
----------
correct_heatflow.py 是随论文发布、供他人使用的入口。它与生产流水线
(rebuild_correction.py)是两份独立的代码,共享的只有模型权重、瓦片读取口径
与深度衰减常量。这三者中任何一处漂移,使用者算出的校正就不再是论文报告的
那个量 —— 而这种漂移不会报错,只会静静给出略有不同的数。

做法:从 rebuild_points.npz(无此文件时改读数据包中的 sites_terrain_corrected.csv)
抽 200 个测点,只把经纬度、实测热流与井深喂给发布脚本,再把它的输出与已发布的
B 和 q_cor 对齐。允许的差异只有
float32 批量推理的批次效应(约 1e-5 相对量级)。

    python test_correct_heatflow.py            # 需要 DEM 与 GPU/CPU 均可
"""
import io
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
DEM = os.environ.get(
    'HEATFLOW_DEM',
    os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif')))
N = int(os.environ.get('N_TEST', '200'))
TOL = 5e-5                      # 相对容差:float32 推理的批次效应约 1e-5

if not os.path.exists(DEM):
    print('跳过:未找到高程数据 %s(设 HEATFLOW_DEM 指向它)' % DEM)
    raise SystemExit(0)

NPZ = os.path.join(HERE, 'rebuild_points.npz')
CSV = os.environ.get('HEATFLOW_SITES', os.path.join(HERE, 'data', 'sites_terrain_corrected.csv'))
if os.path.exists(NPZ):
    d = dict(np.load(NPZ))
elif os.path.exists(CSV):
    s = pd.read_csv(CSV)
    d = dict(lon=s['lon'].values, lat=s['lat'].values, q_measured=s['q_measured_mW_m2'].values,
             borehole_depth=s['borehole_depth_m'].values, B_surface=s['B_s'].values, B=s['B_D'].values,
             q_corrected=s['q_corrected_mW_m2'].values)
else:
    print('跳过:未找到 rebuild_points.npz 或 %s(设 HEATFLOW_SITES 指向数据包中的 CSV)' % CSV)
    raise SystemExit(0)
rng = np.random.RandomState(7)
i = rng.choice(len(d['B']), N, replace=False)

tmp = tempfile.mkdtemp(prefix='hfcor_')
src = os.path.join(tmp, 'sites.csv')
out = os.path.join(tmp, 'out.csv')
pd.DataFrame({'lon': d['lon'][i], 'lat': d['lat'][i],
              'q': d['q_measured'][i], 'depth_m': d['borehole_depth'][i]}
             ).to_csv(src, index=False)

r = subprocess.run([sys.executable, os.path.join(HERE, 'correct_heatflow.py'),
                    '--sites', src, '--dem', DEM, '-o', out],
                   capture_output=True, text=True, encoding='utf-8', errors='replace')
if r.returncode != 0:
    print(r.stdout)
    print(r.stderr)
    raise SystemExit('✗ 发布脚本运行失败')

o = pd.read_csv(out)
bad = 0
for nm, got, ref in [('B_surface', o['B_surface'].values, d['B_surface'][i]),
                     ('B_effective', o['B_effective'].values, d['B'][i]),
                     ('q_corrected', o['q_corrected'].values, d['q_corrected'][i])]:
    rel = np.abs((got - ref) / ref)
    ok = np.nanmax(rel) < TOL
    bad += 0 if ok else 1
    print('  %s %-12s 最大相对差 %.2e  中位 %.2e'
          % ('✓' if ok else '✗', nm, np.nanmax(rel), np.nanmedian(rel)))

if o['B_surface'].isna().any():
    bad += 1
    print('  ✗ 有 %d 个测点没有算出 B —— 已发布的测点应当全部可算'
          % int(o['B_surface'].isna().sum()))

print()
if bad:
    raise SystemExit('✗ 发布脚本与论文产物不一致(%d 项)' % bad)
print('✓ 发布脚本在 %d 个测点上复现了论文的 B 与校正后热流' % N)
