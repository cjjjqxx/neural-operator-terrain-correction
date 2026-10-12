# -*- coding: utf-8 -*-
"""有限元求解器与算子的单瓦片耗时基准。


本脚本用**生产配置**(nx=ny=40, nz=10, 57 km 瓦片, depth=max(3000, 1.5×起伏))
在真实训练瓦片上重测,并把结果落成产物供 emit_numbers 读取。

口径说明
--------
分别报告 build(组装网格与刚度矩阵)与 solve(求解线性系统)两段,
因为读者会问"1.6 秒对一个 1.8 万自由度的稳态线性问题是不是太慢/太快"。
两段分开,这个问题就有据可查。
"""
import io
import json
import os
import sys
import time

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fem3d import Fem3D                                    # noqa: E402

FN = 40
TR_CELL_M = 57000.0 / 128.0          # 与 step2_parallel 一致
N_TILES = int(os.environ.get('N_TILES', '30'))

d = np.load(os.path.join(HERE, 'global_fem_dataset.npz'))
X = d['X']
rng = np.random.RandomState(20260908)
idx = rng.choice(len(X), N_TILES, replace=False)

t_build, t_solve, t_total, dofs = [], [], [], None
for n, i in enumerate(idx):
    dem = X[i, 0] if X.ndim == 4 else X[i]
    relief = float(dem.max() - dem.min())
    f = Fem3D(nx=FN, ny=FN, nz=10)
    t0 = time.perf_counter()
    f.build(dem, TR_CELL_M, depth=max(3000.0, relief * 1.5))
    t1 = time.perf_counter()
    f.solve_heat()
    t2 = time.perf_counter()
    t_build.append(t1 - t0)
    t_solve.append(t2 - t1)
    t_total.append(t2 - t0)
    if dofs is None:
        dofs = int((FN + 1) * (FN + 1) * 11)
    if (n + 1) % 10 == 0:
        print('  %2d/%d  累计中位 %.0f ms' % (n + 1, N_TILES, 1000 * np.median(t_total)))

# 三个量一律用**均值**上报,不用中位数。
# 中位数不可加:build 的中位 + solve 的中位 != total 的中位(实测 1.13 + 1.03 = 2.16,
# 而 total 的中位是 2.17)。正文与图注把这三个数并排给出,读者一加就发现对不上 ——
# 差 0.01 秒虽小,但它让人怀疑的是别的地方还有没有这种account不平。
# 均值天然可加,分布又很窄(P10-P90 只有 2145-2212 ms),换用均值不损失代表性。
res = dict(
    n_tiles=N_TILES, n_dof=dofs, nx=FN, ny=FN, nz=10,
    fem_build_ms=round(1000 * float(np.mean(t_build)), 1),
    fem_solve_ms=round(1000 * float(np.mean(t_solve)), 1),
    fem_total_ms=round(1000 * float(np.mean(t_total)), 1),
    fem_total_ms_median=round(1000 * float(np.median(t_total)), 1),
    fem_total_ms_p10=round(1000 * float(np.percentile(t_total, 10)), 1),
    fem_total_ms_p90=round(1000 * float(np.percentile(t_total, 90)), 1),
)
json.dump(res, open(os.path.join(HERE, 'fem_speed_benchmark.json'), 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1)

print()
print('生产配置 nx=ny=%d, nz=10 —— 自由度 %s' % (FN, format(dofs, ',')))
print('  组装 build   均值 %7.1f ms' % res['fem_build_ms'])
print('  求解 solve   均值 %7.1f ms' % res['fem_solve_ms'])
print('  合计         均值 %7.1f ms   (中位 %.1f, P10 %.1f, P90 %.1f)'
      % (res['fem_total_ms'], res['fem_total_ms_median'],
         res['fem_total_ms_p10'], res['fem_total_ms_p90']))
print('  自洽核对:组装 + 求解 = %.1f ms,合计 %.1f ms'
      % (res['fem_build_ms'] + res['fem_solve_ms'], res['fem_total_ms']))
print()
print('图 5a 单瓦片耗时 %.0f ms' % res['fem_total_ms'])
print('已保存 fem_speed_benchmark.json')
