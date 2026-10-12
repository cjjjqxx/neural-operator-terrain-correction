# -*- coding: utf-8 -*-
"""Davies 对照的裸地形基线:控制住地形本身之后,B 还剩多少独立贡献?


    如果把 B 换成一个**不经过有限元、也不经过算子**的朴素地形代理,
    同样的回归会不会也给出相似的结果?若会,这项检验证明的只是
    「地形与残差热流相关」,而不是「本文训练出的这个物理算子在起作用」。

本脚本用中心像元的相对高程做那个朴素代理 —— 它是 B 的主要驱动
(3.1:预测偏离量与它的相关为 r = −0.67),但完全不含求解器的信息:
读一块高程瓦片就能算出来,不需要解任何方程。

比什么
------
斜率依赖自变量的量纲,直接比大小没有意义。因此做两件事:

  1. **单变量对照**:两个自变量各自标准化后回归,比较标准化斜率与相关系数,
     看裸地形单独能解释多少。
  2. **多元回归**:resid ~ rel_B + rel_h,看控制住裸地形之后 rel_B 的偏系数
     是否仍然显著。这才是"B 是不是地形的冗余代理"的直接检验。

口径与 davies_validation.py 完全一致(同一份 Davies 表、同样的 1.5° 匹配半径、
同样的每格点至少 3 个测点、同样的 |B−1| 分档),只是多算一个自变量。
"""
import io
import json
import os
import sys
import time

import numpy as np
import rasterio
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tile_reader import read_tile, RES                       # noqa: E402

DAV = os.path.join(HERE, 'ggge20271-sup-0001-data_table1_eq_area_global_heat_flow.csv')
DEM = os.environ.get('HEATFLOW_DEM',
                     os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif')))
MAX_DEG = 1.5
DEFAULT_ERR = 41.5
MIN_PER_CELL = 3
BAND = float(os.environ.get('BAND', '0.05'))     # 主检验用 |B−1| >= 0.05


def load_davies():
    raw = open(DAV, 'rb').read().decode('utf-8', 'replace')
    lines = [l for l in raw.replace('\r\n', '\n').replace('\r', '\n').split('\n') if l.strip()]
    out = []
    for l in lines:
        p = l.split(',')
        try:
            out.append([float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])])
        except (ValueError, IndexError):
            continue
    return np.array(out)


D = load_davies()
dlon, dlat, dmean, derr = D[:, 0], D[:, 1], D[:, 2], D[:, 4]
d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
lon, lat, q, B = d['lon'], d['lat'], d['q_measured'], d['B']

idx = np.empty(len(q), dtype=np.int64)
dist = np.empty(len(q))
for i in range(len(q)):
    dla = dlat - lat[i]
    dlo = (dlon - lon[i] + 180) % 360 - 180
    dd = np.hypot(dla, dlo * np.cos(np.radians(lat[i])))
    j = int(np.argmin(dd))
    idx[i], dist[i] = j, dd[j]

ok = (dist <= MAX_DEG) & (derr[idx] != DEFAULT_ERR)
nb = len(dmean)
sumB = np.bincount(idx[ok], weights=B[ok], minlength=nb)
cnt = np.bincount(idx[ok], minlength=nb)
meanB = np.divide(sumB, cnt, out=np.ones(nb), where=cnt > 0)
ok &= (cnt[idx] >= MIN_PER_CELL)

qd = dmean[idx]
rel_B = B / meanB[idx]
resid = (q - qd) / qd
ab = np.abs(B - 1.0)
sel = ok & (ab >= BAND)
print('主检验样本 %d 个测点(|B−1| >= %.2f)' % (sel.sum(), BAND))

# ── 裸地形代理:中心像元相对高程,按瓦片起伏归一 ──────────────────────
# 不经过任何求解器:读一块瓦片,取中心像元与瓦片均值之差,除以瓦片起伏。
ds = rasterio.open(DEM)
c = RES // 2
rel_h = np.full(len(q), np.nan)
t0 = time.time()
todo = np.nonzero(sel)[0]
for n, i in enumerate(todo):
    try:
        dem = read_tile(ds, float(lon[i]), float(lat[i]), apply_taper=False)
    except Exception:
        dem = None
    if dem is None:
        continue
    rng = float(dem.max() - dem.min())
    if rng <= 0:
        continue
    rel_h[i] = (float(dem[c, c]) - float(dem.mean())) / rng
    if (n + 1) % 500 == 0:
        print('  %d/%d  用时 %.1f min' % (n + 1, len(todo), (time.time() - t0) / 60))
ds.close()

m = sel & np.isfinite(rel_h)
x_b = rel_B[m] - 1.0
x_h = rel_h[m]
y = resid[m]
print('可用 %d 个测点;rel_h 的标准差 %.3f,rel_B−1 的标准差 %.4f'
      % (m.sum(), x_h.std(), x_b.std()))


def z(v):
    return (v - v.mean()) / v.std()


rb = stats.linregress(z(x_b), z(y))
rh = stats.linregress(z(x_h), z(y))

# 多元:控制裸地形后 B 的偏系数
X = np.column_stack([np.ones(m.sum()), z(x_b), z(x_h)])
coef, *_ = np.linalg.lstsq(X, z(y), rcond=None)
resid_ols = z(y) - X @ coef
dof = m.sum() - 3
s2 = float(resid_ols @ resid_ols) / dof
cov = s2 * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(cov))
tvals = coef / se
pvals = 2 * stats.t.sf(np.abs(tvals), dof)

res = dict(
    band=BAND, n=int(m.sum()),
    # 单变量(自变量与因变量都标准化,故斜率即相关系数,可直接互比)
    beta_B=round(float(rb.slope), 4), p_B=float('%.3g' % rb.pvalue),
    beta_h=round(float(rh.slope), 4), p_h=float('%.3g' % rh.pvalue),
    # 多元:控制裸地形之后 B 的偏系数
    partial_B=round(float(coef[1]), 4), partial_B_p=float('%.3g' % pvals[1]),
    partial_h=round(float(coef[2]), 4), partial_h_p=float('%.3g' % pvals[2]),
    corr_B_h=round(float(np.corrcoef(x_b, x_h)[0, 1]), 3),
)
json.dump(res, open(os.path.join(HERE, 'davies_terrain_baseline.json'), 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1)

print()
print('=== 单变量(标准化)===')
print('  算子的 B          β = %+.4f   p = %s' % (res['beta_B'], res['p_B']))
print('  裸地形相对高程     β = %+.4f   p = %s' % (res['beta_h'], res['p_h']))
print('  两个自变量之间的相关 %.3f' % res['corr_B_h'])
print()
print('=== 多元:控制裸地形之后 ===')
print('  B 的偏系数        %+.4f   p = %s' % (res['partial_B'], res['partial_B_p']))
print('  裸地形的偏系数     %+.4f   p = %s' % (res['partial_h'], res['partial_h_p']))
print()
print('已保存 davies_terrain_baseline.json')
