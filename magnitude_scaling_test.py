# -*- coding: utf-8 -*-
"""校正幅度:观测数据支持多大比例的几何校正?(对应审稿共识 C1)

两项观测检验都只证实了方向。本脚本问幅度,并把一个本文未建模、但方向确定的物理过程
单独拿出来量化:地表温度随高程递减。

地表温度递减的严格标度
----------------------
地表条件由 T = 0 换成 T = −γ·e(e 为高程,γ 为地表温度随高程的递减率)。令
T' = T + γ·e,则 T' 在地表为 0、仍满足拉普拉斯方程,底面通量变为 q0 − Kγ。
于是 T' 正是本文的几何问题,地表垂向通量 q_s = B(q0 − Kγ) + Kγ,即
    B_eff − 1 = (1 − γ/G)·(B − 1),   G = q0/K
对均匀 K 与按垂向通量定义的 B,这一关系不依赖线性化。γ > 0 时它只会按比例缩小校正幅度。
G 取 q_obs / K(K = 2.5 W m⁻¹ K⁻¹);γ 取 4.0 / 5.5 / 6.5 K km⁻¹ 三档。
回归检验另用区域背景 G = q_Davies / K(slope_lapse_bg_*),避免自变量与因变量共享 q_obs;正文取后者。

检验
----
1. 邻近测点配对:把校正按 α 缩放,B_α = 1 + α(B − 1),α 自 0 到 1.5;
   看配对差的离散度在哪个 α 最小(标准差口径同正文,另给对数比的稳健口径 MAD)。
2. Davies 场残差回归:原口径斜率;对数空间斜率;以地表温度递减缩放后的相对传递因子作
   自变量的斜率;按 Davies 格点做聚类自助的 95% 置信区间(测点在格点内不独立)。
输出 magnitude_scaling_test.json。
"""
import json
import os
import sys

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from davies_validation import load_davies, MAX_DEG, DEFAULT_ERR, MIN_PER_CELL  # noqa: E402

K_COND = 2.5
GAMMAS = (4.0, 5.5, 6.5)                   # K/km
ALPHAS = np.round(np.arange(0.0, 1.501, 0.05), 2)
EDGES = [0, 0.02, 0.05, 0.10, 0.20, np.inf]
LABELS = ['< 0.02', '0.02-0.05', '0.05-0.10', '0.10-0.20', '> 0.20']
N_BOOT, SEED = 2000, 20260929


def lapse_factor(q, gamma):
    G = q / K_COND                          # mW m^-2 / (W m^-1 K^-1) = K/km
    return np.clip(1.0 - gamma / G, 0.0, 1.0)


def mad(x):
    return float(np.median(np.abs(x - np.median(x))))


def paired(lon, lat, q, B):
    r = np.radians
    xyz = np.column_stack([np.cos(r(lat)) * np.cos(r(lon)), np.cos(r(lat)) * np.sin(r(lon)),
                           np.sin(r(lat))]) * 6371.0
    i, j = cKDTree(xyz).query_pairs(10.0, output_type='ndarray').T
    ab = np.abs(B - 1)
    band = np.digitize(np.minimum(ab[i], ab[j]), EDGES[1:-1])
    out = {}
    for s, lab in list(enumerate(LABELS)) + [(None, '全部')]:
        m = np.ones(len(i), bool) if s is None else band == s
        d0 = q[i[m]] - q[j[m]]
        l0 = np.log(q[i[m]] / q[j[m]])
        sd, rob = [], []
        for a in ALPHAS:
            Bi, Bj = 1 + a * (B[i[m]] - 1), 1 + a * (B[j[m]] - 1)
            sd.append(100 * (1 - (q[i[m]] / Bi - q[j[m]] / Bj).std() / d0.std()))
            rob.append(100 * (1 - mad(l0 - np.log(Bi) + np.log(Bj)) / mad(l0)))
        sd, rob = np.array(sd), np.array(rob)
        k1 = int(np.where(ALPHAS == 1.0)[0][0])
        out[lab] = dict(n=int(m.sum()), sd_reduction_at_1=float(sd[k1]), mad_reduction_at_1=float(rob[k1]),
                        alpha_best_sd=float(ALPHAS[np.argmax(sd)]), sd_reduction_best=float(sd.max()),
                        alpha_best_mad=float(ALPHAS[np.argmax(rob)]), mad_reduction_best=float(rob.max()))
    return out


def davies(lon, lat, q, B):
    D = load_davies()
    dlon, dlat, dmean, derr = D[:, 0], D[:, 1], D[:, 2], D[:, 4]
    tree = cKDTree(np.column_stack([dlat, dlon]))
    idx = np.empty(len(q), np.int64)
    dist = np.empty(len(q))
    for t in range(len(q)):                          # 与 davies_validation 同一最近格点规则
        dla = dlat - lat[t]
        dlo = (dlon - lon[t] + 180) % 360 - 180
        dd = dla * dla + (dlo * np.cos(np.radians(lat[t]))) ** 2
        k = int(np.argmin(dd))
        idx[t], dist[t] = k, np.sqrt(dd[k])
    ok = (dist <= MAX_DEG) & (derr[idx] != DEFAULT_ERR)
    nb = len(dmean)
    cnt = np.bincount(idx[ok], minlength=nb)
    ok &= cnt[idx] >= MIN_PER_CELL
    qd = dmean[idx]
    resid = (q - qd) / qd
    ab = np.abs(B - 1)
    rng = np.random.default_rng(SEED)

    def relB(Bx):
        s = np.bincount(idx[ok], weights=Bx[ok], minlength=nb)
        c = np.bincount(idx[ok], minlength=nb)
        mB = np.divide(s, c, out=np.ones(nb), where=c > 0)
        return Bx / mB[idx]

    def slope(x, y, m):
        return float(stats.linregress(x[m], y[m]).slope)

    def cluster_ci(x, y, m, rng=rng):
        cells = np.unique(idx[m])
        members = {c: np.where(m & (idx == c))[0] for c in cells}
        bs = []
        for _ in range(N_BOOT):
            pick = rng.choice(cells, len(cells), replace=True)
            ii = np.concatenate([members[c] for c in pick])
            bs.append(stats.linregress(x[ii], y[ii]).slope)
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    rb = relB(B)
    out = {}
    for lo, lab in ((0.0, '全部'), (0.05, '>= 0.05'), (0.10, '>= 0.10')):
        m = ok & (ab >= lo)
        r = stats.linregress(rb[m] - 1, resid[m])
        row = dict(n=int(m.sum()), n_cells=int(len(np.unique(idx[m]))),
                   slope=float(r.slope), ci_iid=[float(r.slope - 1.96 * r.stderr), float(r.slope + 1.96 * r.stderr)],
                   ci_cluster=cluster_ci(rb - 1, resid, m))
        pos = m & (q > 0) & (qd > 0)
        row['slope_log'] = slope(np.log(rb), np.log(q / qd), pos)
        row['slope_log_ci_cluster'] = cluster_ci(np.log(rb), np.log(np.where(pos, q / qd, 1.0)), pos)
        for g in GAMMAS:
            Be = 1 + lapse_factor(q, g) * (B - 1)
            rbe = relB(Be)
            row['slope_lapse_%.1f' % g] = slope(rbe - 1, resid, m)
            Bg = 1 + lapse_factor(qd, g) * (B - 1)
            rbg = relB(Bg)
            row['slope_lapse_bg_%.1f' % g] = slope(rbg - 1, resid, m)
            if g == 5.5:
                # 独立随机数流:不消耗主流程的 rng,原有各档的自助区间逐位不变
                row['slope_lapse_bg_5.5_ci_cluster'] = cluster_ci(rbg - 1, resid, m,
                                                                  rng=np.random.default_rng(SEED + 55))
                row['lapse_bg_factor_median'] = float(np.median(lapse_factor(qd[m], g)))
        out[lab] = row
    return out


def main():
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    lon, lat, q, B = d['lon'], d['lat'], d['q_measured'], d['B']
    res = dict(K=K_COND, gammas=list(GAMMAS))
    lf = {('%.1f' % g): dict(median=float(np.median(lapse_factor(q, g))),
                             p10=float(np.percentile(lapse_factor(q, g), 10)),
                             p90=float(np.percentile(lapse_factor(q, g), 90)),
                             median_ge010=float(np.median(lapse_factor(q, g)[np.abs(B - 1) >= 0.10])))
          for g in GAMMAS}
    res['lapse_factor'] = lf
    print('地表温度递减的缩放因子 1 − γ/G(G = q_obs / %.1f):' % K_COND)
    for g, v in lf.items():
        print('  γ = %s K/km  中位 %.3f(P10 %.3f,P90 %.3f);|B−1| ≥ 0.10 档中位 %.3f'
              % (g, v['median'], v['p10'], v['p90'], v['median_ge010']))

    res['paired'] = paired(lon, lat, q, B)
    print('\n邻近配对:校正按 α 缩放后配对差离散度的降幅(%)')
    print('%-10s %7s %9s %9s %7s %9s %7s %9s' % ('档', 'n', 'SD@α=1', 'MAD@α=1', 'α*SD', 'SD最佳', 'α*MAD', 'MAD最佳'))
    for lab, r in res['paired'].items():
        print('%-10s %7d %+9.2f %+9.2f %7.2f %+9.2f %7.2f %+9.2f' % (
            lab, r['n'], r['sd_reduction_at_1'], r['mad_reduction_at_1'], r['alpha_best_sd'],
            r['sd_reduction_best'], r['alpha_best_mad'], r['mad_reduction_best']))

    res['davies'] = davies(lon, lat, q, B)
    print('\nDavies 场残差回归斜率:')
    for lab, r in res['davies'].items():
        print('  %-8s n=%5d 格点 %4d  斜率 %.3f  独立同分布 CI [%.3f, %.3f]  格点聚类 CI [%.3f, %.3f]'
              % (lab, r['n'], r['n_cells'], r['slope'], *r['ci_iid'], *r['ci_cluster']))
        print('           对数空间 %.3f  聚类 CI [%.3f, %.3f];地表温度递减缩放后 %s'
              % (r['slope_log'], *r['slope_log_ci_cluster'],
                 ' / '.join('γ=%.1f: %.3f' % (g, r['slope_lapse_%.1f' % g]) for g in GAMMAS)))
    json.dump(res, open(os.path.join(HERE, 'magnitude_scaling_test.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('\n已保存 magnitude_scaling_test.json')


if __name__ == '__main__':
    main()
