# -*- coding: utf-8 -*-
"""经典高热流山区的表观异常里,有多少是地形假象。


做法:在若干有名有姓的构造省内,比较施加地形校正前后的区域均值,
并用自助法给区间;同时报告该省测点的谷地占比,以显示机制。

"""
import io
import os
import sys
import json

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 4000
SEED = 20260827

# 有名有姓的构造省(经度min, 经度max, 纬度min, 纬度max)。
# 选取标准:测点数 >= 25 且以山地为主,不按结果挑选。
PROVINCES = {
    'Canadian Cordillera':  (-126, -118, 49, 55),
    'Eastern Alps':         (10, 16, 45, 48),
    'Cascade Range':        (-123, -120, 42, 49),
    'Japan (Honshu)':       (136, 141, 34, 39),
    'Southern Andes':       (-73, -68, -40, -30),
    'Swiss/French Alps':    (5, 10, 44, 47),
    'Rocky Mountains (US)': (-114, -105, 37, 46),
    'Himalaya-Tibet':       (75, 100, 27, 36),
}


def main():
    rng = np.random.default_rng(SEED)
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    lon, lat = d['lon'], d['lat']
    q, qc, B = d['q_measured'], d['q_corrected'], d['B']

    print('%-22s %5s %8s %8s %9s %20s %8s'
          % ('构造省', 'n', '实测', '校正后', '变化%', '95%CI', '谷地%'))
    print('-' * 86)
    out = []
    for name, (a, b, c, e) in PROVINCES.items():
        m = (lon >= a) & (lon < b) & (lat >= c) & (lat < e)
        if m.sum() < 25:
            print('%-22s %5d  (测点不足,跳过)' % (name, m.sum()))
            continue
        q0, q1 = q[m], qc[m]
        rel = 100 * (q1.mean() - q0.mean()) / q0.mean()
        boot = np.empty(N_BOOT)
        for k in range(N_BOOT):
            i = rng.integers(0, m.sum(), m.sum())
            boot[k] = 100 * (q1[i].mean() - q0[i].mean()) / q0[i].mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        vfrac = 100 * (B[m] > 1).mean()
        print('%-22s %5d %8.1f %8.1f %8.1f%% [%+6.1f, %+6.1f]%8.0f%%'
              % (name, m.sum(), q0.mean(), q1.mean(), rel, lo, hi, vfrac))
        out.append(dict(province=name, n=int(m.sum()),
                        mean_measured=round(float(q0.mean()), 1),
                        mean_corrected=round(float(q1.mean()), 1),
                        change_pct=round(float(rel), 1),
                        ci_lo=round(float(lo), 1), ci_hi=round(float(hi), 1),
                        valley_frac_pct=round(float(vfrac), 0),
                        median_absB1=round(float(np.median(np.abs(B[m] - 1))), 3)))

    # 全球参照:绝大多数区域几乎不变,尾部才是发现所在
    gi = np.floor((lon + 180) / 2).astype(int)
    gj = np.floor((lat + 90) / 2).astype(int)
    key = gi * 1000 + gj
    uk, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv)
    sm = np.bincount(inv, weights=q)
    sc = np.bincount(inv, weights=qc)
    ok = n >= 25
    relc = 100 * (sc[ok] / n[ok] - sm[ok] / n[ok]) / (sm[ok] / n[ok])
    print('\n全球 2°x2° 格子(>=25 测点)共 %d 个:中位变化 %+.2f%%,'
          '5–95%% 分位 [%+.2f%%, %+.2f%%]'
          % (ok.sum(), np.median(relc), *np.percentile(relc, [5, 95])))
    print('即:绝大多数区域几乎不受影响,量级集中在少数以谷底钻孔为主的山区。')

    from scipy import stats as _st
    sv = np.bincount(inv, weights=(B > 1).astype(float))
    vfrac_c = 100.0 * sv[ok] / n[ok]                      # 每格 B>1 测点占比
    _rho, _prho = _st.spearmanr(vfrac_c, relc)
    _lr = _st.linregress(vfrac_c, relc)
    _lo, _hi = vfrac_c < 55.0, vfrac_c > 75.0
    print('\n2° 格子上「谷地占比 vs 校正后区域均值变化」:')
    print('  Spearman rho = %+.3f (p = %.3g);OLS 斜率 = %+.4f %%/pp'
          % (_rho, _prho, _lr.slope))
    print('  谷地占比 <55%%(n=%d)中位变化 %+.2f%% | >75%%(n=%d)中位 %+.2f%%'
          % (_lo.sum(), np.median(relc[_lo]), _hi.sum(), np.median(relc[_hi])))

    # 逐格数据另存,供图 10b 画散点 —— 只有汇总统计画不出「关系是单调的」。
    np.savez_compressed(os.path.join(HERE, 'province_cells_2deg.npz'),
                        valley_frac_pct=vfrac_c.astype(np.float32),
                        change_pct=relc.astype(np.float32),
                        n_sites=n[ok].astype(np.int32))
    print('  逐格数据已存: province_cells_2deg.npz (%d 格)' % ok.sum())

    json.dump(dict(provinces=out, n_cells=int(ok.sum()),
                   cell_median_change_pct=round(float(np.median(relc)), 2),
                   cell_p5=round(float(np.percentile(relc, 5)), 2),
                   cell_p95=round(float(np.percentile(relc, 95)), 2),
                   cell_valley_spearman=round(float(_rho), 3),
                   cell_valley_spearman_p=float('%.3g' % _prho),
                   cell_valley_slope_pct_per_pp=round(float(_lr.slope), 4),
                   cell_change_median_valley_lt55=round(float(np.median(relc[_lo])), 2),
                   cell_change_median_valley_gt75=round(float(np.median(relc[_hi])), 2),
                   cell_n_valley_lt55=int(_lo.sum()), cell_n_valley_gt75=int(_hi.sum()),
                   n_boot=N_BOOT, seed=SEED,
                   caveat='幅度在有限元尺度上换算;Davies 斜率是下界而非折算系数(正文3.4);'
                          '区域为经纬度框'),
              open(os.path.join(HERE, 'province_artifact_test.json'), 'w',
                   encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n已保存 province_artifact_test.json')


if __name__ == '__main__':
    main()
