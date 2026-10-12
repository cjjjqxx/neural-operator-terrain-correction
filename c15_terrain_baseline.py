# -*- coding: utf-8 -*-
"""第二项观测检验的裸地形基线(审稿 R1-M5、R2-M6)。

现场判断检验显示:原研究者判为「识别出但未校正」的测点,算子 |B−1| 高于
「存在但不显著」的测点(Cliff's δ 见 c15_ordinal_test.json)。替代解释:研究者只是看到了自己身处山区,
而算子 |B−1| 与起伏高度相关 —— 那样这一检验并不说明算子携带了超出「这里地形起伏大」的信息。

与第三项检验的裸地形基线同一思路:用不经有限元、不经算子的地形量(瓦片起伏、脊谷指数的
绝对值)做同一比较,再把两者一并放入逻辑回归(自变量取秩并标准化),看控制起伏之后
|B−1| 是否仍有独立贡献。只用前两类(二者都在应用集内,地形指标齐全)。
输出 c15_terrain_baseline.json。
"""
import collections
import json
import os
import sys

import numpy as np
from scipy import stats, optimize

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from c15_ordinal_test import ORDER, site_key, location_categories  # noqa: E402

LOW, HIGH = ORDER[0], ORDER[1]


def cliffs(x, y):
    u = stats.mannwhitneyu(y, x, alternative='two-sided').statistic
    return float(2 * u / (len(x) * len(y)) - 1)


def logit_fit(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])

    def nll(b):
        z = X1 @ b
        return np.sum(np.logaddexp(0, z) - y * z)
    r = optimize.minimize(nll, np.zeros(X1.shape[1]), method='BFGS')
    b = r.x
    p = 1 / (1 + np.exp(-(X1 @ b)))
    H = X1.T @ (X1 * (p * (1 - p))[:, None])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    z = b / se
    return [dict(coef=float(b[k]), se=float(se[k]), z=float(z[k]), p=float(2 * stats.norm.sf(abs(z[k]))))
            for k in range(1, len(b))]


def main():
    best, conflict = location_categories()          # 与 c15_ordinal_test 同一规则:矛盾位置整点剔除
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    t = np.load(os.path.join(HERE, 'site_terrain.npz'))
    assert np.allclose(t['lon'], d['lon'])
    ab = np.abs(d['B'] - 1)
    relief, rix = t['relief_m'].astype(float), np.abs(t['rel_elev'].astype(float))
    keys = [site_key(float(a), float(b)) for a, b in zip(d['lon'], d['lat'])]
    grp = np.array(['' if k in conflict else best.get(k, '') for k in keys])
    lo, hi = grp == LOW, grp == HIGH
    out = dict(n_low=int(lo.sum()), n_high=int(hi.sum()))
    for nm, v in (('absB1', ab), ('relief', relief), ('abs_ridge_valley_index', rix)):
        out['delta_' + nm] = cliffs(v[lo], v[hi])
        out['median_low_' + nm] = float(np.median(v[lo]))
        out['median_high_' + nm] = float(np.median(v[hi]))
    m = lo | hi
    y = hi[m].astype(float)

    def zr(v):
        r = stats.rankdata(v[m])
        return (r - r.mean()) / r.std()
    out['logit_absB1_only'] = logit_fit(zr(ab)[:, None], y)[0]
    out['logit_relief_only'] = logit_fit(zr(relief)[:, None], y)[0]
    both = logit_fit(np.column_stack([zr(ab), zr(relief)]), y)
    out['logit_both'] = dict(absB1=both[0], relief=both[1])
    out['spearman_absB1_relief'] = float(stats.spearmanr(ab[m], relief[m])[0])
    print('「存在但不显著」n=%d  vs  「识别未校正」n=%d' % (out['n_low'], out['n_high']))
    for nm in ('absB1', 'relief', 'abs_ridge_valley_index'):
        print('  %-24s 中位 %.4g → %.4g   Cliff δ = %+.3f' % (
            nm, out['median_low_' + nm], out['median_high_' + nm], out['delta_' + nm]))
    print('  |B−1| 与起伏的秩相关 %.3f' % out['spearman_absB1_relief'])
    for k in ('logit_absB1_only', 'logit_relief_only'):
        r = out[k]
        print('  %-18s 系数 %+.3f  p = %.2g' % (k, r['coef'], r['p']))
    for k, r in out['logit_both'].items():
        print('  联合回归 %-8s 系数 %+.3f  p = %.2g' % (k, r['coef'], r['p']))
    json.dump(out, open(os.path.join(HERE, 'c15_terrain_baseline.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('已保存 c15_terrain_baseline.json')


if __name__ == '__main__':
    main()
