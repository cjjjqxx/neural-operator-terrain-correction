# -*- coding: utf-8 -*-
"""谷地偏倚的连续趋势检验(审稿 R1-M7、R2-M4:10.9 个百分点取自事后分层的最大一层)。

主检验不再依赖起伏分层:对 1,993 组「测点—其 10 km 内随机局地参照」配对,
  d_i = 1[B_fem(测点) > 1] − 1[B_fem(参照) > 1]
对 x_i = log10(测点瓦片起伏) 做最小二乘回归,斜率即「起伏每增大 10 倍,测点相对局地参照
落在通量增强侧的比例增加多少个百分点」。置信区间与 p 值按测点所在 2° 块做块自助(与
valley_bias_matched.py 同一分块),单侧检验斜率 > 0。另报告在起伏 = 1,000 m 处的拟合超出量。

输入 valley_bias_matched.npz(同一估计器:有限元),输出 valley_trend_test.json。
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT, SEED = 5000, 20260930


def main():
    v = np.load(os.path.join(HERE, 'valley_bias_matched.npz'))
    S, L = v['kind'] == 'site', v['kind'] == 'local'
    assert np.array_equal(v['idx'][S], v['idx'][L]), '测点与局地参照未一一对应'
    d = (v['B_fem'][S] > 1).astype(float) - (v['B_fem'][L] > 1).astype(float)
    x = np.log10(np.clip(v['relief'][S], 10, None))
    blk = (np.floor(v['lon'][S] / 2) * 1000 + np.floor(v['lat'][S] / 2)).astype(int)

    def fit(dd, xx):
        A = np.column_stack([np.ones_like(xx), xx - 3.0])      # 截距 = 起伏 1,000 m 处的超出
        c, *_ = np.linalg.lstsq(A, dd, rcond=None)
        return c

    c0 = fit(d, x)
    ub = np.unique(blk)
    members = {b: np.where(blk == b)[0] for b in ub}
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(N_BOOT):
        pick = rng.choice(ub, len(ub))
        ii = np.concatenate([members[b] for b in pick])
        bs.append(fit(d[ii], x[ii]))
    bs = np.array(bs)
    slope, icpt = 100 * c0[1], 100 * c0[0]
    out = dict(n_pairs=int(S.sum()), n_blocks=int(len(ub)),
               slope_pp_per_decade=float(slope),
               slope_ci=[float(100 * np.percentile(bs[:, 1], 2.5)), float(100 * np.percentile(bs[:, 1], 97.5))],
               p_one_sided=float(max(np.mean(bs[:, 1] <= 0), 1.0 / N_BOOT)),
               excess_at_1000m_pp=float(icpt),
               excess_at_1000m_ci=[float(100 * np.percentile(bs[:, 0], 2.5)),
                                   float(100 * np.percentile(bs[:, 0], 97.5))],
               relief_median_m=float(10 ** np.median(x)))
    json.dump(out, open(os.path.join(HERE, 'valley_trend_test.json'), 'w', encoding='utf-8'), indent=2)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
