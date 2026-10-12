# -*- coding: utf-8 -*-
"""外部验证:钻孔实测相对平滑区域模型的残差,是否跟随我们预测的地形因子。

参照场:Davies (2013) 2° 等面积全球热流图(AGU Supporting Information)。



已知局限(须在正文如实报告):
  Davies 部分由同一批钻孔建成,两者不完全独立。2° 格点内含多个钻孔,单点贡献被
  稀释,但循环性无法消除。Colgan 同样存在此局限。
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DAV = os.path.join(HERE, 'ggge20271-sup-0001-data_table1_eq_area_global_heat_flow.csv')
MAX_DEG = 1.5
DEFAULT_ERR = 41.5          # 无数据处的地质学缺省值标记
MIN_PER_CELL = 3            # 估计 <B> 所需的最少同格点测点数


def load_davies():
    raw = open(DAV, 'rb').read().decode('utf-8', 'replace')
    lines = [l for l in raw.replace('\r\n', '\n').replace('\r', '\n').split('\n') if l.strip()]
    return np.array([[float(x) for x in l.split(',')] for l in lines[1:]])


def main():
    D = load_davies()
    dlon, dlat, dmean, derr = D[:, 0], D[:, 1], D[:, 2], D[:, 4]
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    lon, lat, q, B = d['lon'], d['lat'], d['q_measured'], d['B']

    idx = np.empty(len(q), dtype=np.int64)
    dist = np.empty(len(q))
    for i in range(len(q)):
        dla = dlat - lat[i]
        dlo = (dlon - lon[i] + 180) % 360 - 180
        dd = dla * dla + (dlo * np.cos(np.radians(lat[i]))) ** 2
        j = int(np.argmin(dd))
        idx[i], dist[i] = j, np.sqrt(dd[j])

    ok = (dist <= MAX_DEG) & (derr[idx] != DEFAULT_ERR)

    # 每个 Davies 格点内的平均地形因子 <B>,用同格点的本文测点估计
    nb = len(dmean)
    sumB = np.bincount(idx[ok], weights=B[ok], minlength=nb)
    cnt = np.bincount(idx[ok], minlength=nb)
    meanB = np.divide(sumB, cnt, out=np.ones(nb), where=cnt > 0)
    ok &= (cnt[idx] >= MIN_PER_CELL)

    qd = dmean[idx]
    rel_B = B / meanB[idx]                      # 相对格点平均的地形因子
    resid = (q - qd) / qd                       # 实测相对区域模型的相对残差

    print('可用测点 %d(角距 <= %.1f°、非缺省格点、同格点 >= %d 点)'
          % (ok.sum(), MAX_DEG, MIN_PER_CELL))
    print('每格点测点数 中位 %d,最大 %d' % (int(np.median(cnt[cnt > 0])), int(cnt.max())))
    print('信噪比参考:残差 std %.1f%%,而 rel_B-1 的 std 仅 %.2f%%\n'
          % (100 * resid[ok].std(), 100 * (rel_B[ok] - 1).std()))

    print('=== 主检验:残差 vs 相对地形因子 ===')
    print('%-14s %7s %10s %10s %12s' % ('|B-1| 档', 'n', 'Pearson r', 'Spearman', 'p(Spearman)'))
    print('-' * 60)
    ab = np.abs(B - 1.0)
    for lo, hi, lbl in [(0, 0.02, '< 0.02'), (0.02, 0.05, '0.02-0.05'),
                        (0.05, 0.10, '0.05-0.10'), (0.10, 9, '>= 0.10'),
                        (0.05, 9, '>= 0.05'), (0, 9, '全部')]:
        m = ok & (ab >= lo) & (ab < hi)
        if m.sum() < 20:
            continue
        r = np.corrcoef(rel_B[m] - 1, resid[m])[0, 1]
        rho, p = stats.spearmanr(rel_B[m] - 1, resid[m])
        print('%-14s %7d %10.4f %10.4f %12.3g' % (lbl, m.sum(), r, rho, p))

    print('\n=== 辅助:按相对地形因子分位看残差中位数 ===')
    m = ok & (ab >= 0.05)
    v = rel_B[m] - 1
    qs = np.quantile(v, [0, .2, .4, .6, .8, 1])
    print('%-22s %7s %12s' % ('rel_B-1 分位', 'n', '残差中位(%)'))
    quints = []
    for i in range(5):
        s = m.copy()
        s[m] = (v >= qs[i]) & (v <= qs[i + 1])
        med = 100 * float(np.median(resid[s]))
        print('%-22s %7d %11.2f%%' % ('[%.3f, %.3f]' % (qs[i], qs[i + 1]),
                                      s.sum(), med))
        quints.append(dict(lo=round(float(qs[i]), 4), hi=round(float(qs[i + 1]), 4),
                           n=int(s.sum()), resid_median_pct=round(med, 2)))
    # 回归斜率:斜率=1 表示实测残差完全等于预测的相对地形因子。
    # 斜率<1 的已知成因:①回归稀释(rel_B 带误差,且 <B> 由有限样本估计);
    # ②Davies 在 2°(约 200 km)上平滑,并非干净的 q_深部;③地质异质性主导残差。
    print('\n=== 幅度:残差对 rel_B-1 的回归斜率 ===')
    print('%-10s %7s %9s %20s' % ('|B-1|档', 'n', '斜率', '95%CI'))
    print('-' * 52)
    res = {}
    for lo, hi, lbl in [(0.05, 9, '>= 0.05'), (0.10, 9, '>= 0.10'), (0, 9, '全部')]:
        m = ok & (ab >= lo) & (ab < hi)
        r = stats.linregress(rel_B[m] - 1, resid[m])
        ci = (r.slope - 1.96 * r.stderr, r.slope + 1.96 * r.stderr)
        print('%-10s %7d %9.3f  [%+.3f, %+.3f]' % (lbl, m.sum(), r.slope, *ci))
        res[lbl] = dict(n=int(m.sum()), slope=round(float(r.slope), 3),
                        ci_lo=round(ci[0], 3), ci_hi=round(ci[1], 3),
                        p=float('%.3g' % r.pvalue))
    res['quintiles_ge005'] = quints
    _mm = ok & (ab >= 0.05)
    _rho2, _p2 = stats.spearmanr(rel_B[_mm] - 1, resid[_mm])
    res['spearman_ge005'] = dict(rho=round(float(_rho2), 3), p=float('%.3g' % _p2),
                                 n=int(_mm.sum()))
    _rho3, _p3 = stats.spearmanr(rel_B[ok & (ab >= 0.10)] - 1,
                                 resid[ok & (ab >= 0.10)])
    res['spearman_ge010'] = dict(rho=round(float(_rho3), 3), p=float('%.3g' % _p3))
    # 正文那句「残差标准差 44.5%,传递因子仅 4.1%」讲的是**全样本**的量级对比
    # (说明地质异质性主导残差),不是 >=0.05 子集 —— 子集上两者分别为 55%/11%,
    # 口径混用会让这句话对不上。两个口径都存,键名区分。
    res['resid_sd_pct'] = round(100 * float(np.std(resid[ok])), 1)
    res['relB_sd_pct'] = round(100 * float(np.std(rel_B[ok] - 1)), 1)
    res['resid_sd_pct_ge005'] = round(100 * float(np.std(resid[_mm])), 1)
    res['relB_sd_pct_ge005'] = round(100 * float(np.std(rel_B[_mm] - 1)), 1)
    import json
    json.dump(res, open(os.path.join(HERE, 'davies_validation.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
    # 散点数据另存:图 9c 要画「相对残差 vs 相对传递因子」,汇总统计画不出
    # 「斜率显著大于 0 但显著小于 1」这个论点的证据。
    np.savez_compressed(os.path.join(HERE, 'davies_points.npz'),
                        rel_B_minus_1=(rel_B[_mm] - 1).astype(np.float32),
                        resid=resid[_mm].astype(np.float32),
                        absB1=ab[_mm].astype(np.float32))
    print('散点数据已存: davies_points.npz (%d 点)' % _mm.sum())
    print('\n斜率显著大于 0 = 地形信号确实存在于实测中(对照独立外部场);')
    print('斜率显著小于 1 = 幅度约为预测的四至五成,衰减来源见上方注释。')
    print('注:Davies 部分由同一批钻孔建成,两者不完全独立;须在正文如实说明。')


if __name__ == '__main__':
    main()
