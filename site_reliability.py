# -*- coding: utf-8 -*-
"""逐测点可信度分级 —— 给 36,392 个校正值各配一个方法误差与等级。


传播到通量:q_cor = q_obs / B,|dq| = q_cor·σ_B / B;校正量 Δq = q_cor − q_obs = q_cor(1 − B)。
信噪比 SNR = |Δq| / σ_q。分级沿用「2 倍误差」的通行约定:
    A  SNR ≥ 2      校正显著:校正后的值与原值在方法误差之外可区分
    B  1 ≤ SNR < 2  校正可分辨
    C  SNR < 1      校正处于方法误差之内,与零校正不可区分


另一项比较来自 IHFC 自身的质量体系:对报告了不确定度等级(U1–U4)的测点,看地形校正量
|Δq|/q_obs 是否超过该等级的变异系数上限(U1 5%、U2 15%、U3 25%;U4 无上限,取 25%)。
超过者,地形效应已大于这个测点本身的测量不确定度。

输出 site_reliability.npz(逐点)与 site_reliability.json(汇总)。
"""
import json
import os
import sys

import numpy as np

from error_model import BANDS, band_of, load_terms, per_site

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(HERE, f), encoding='utf-8'))

RELIEF_MIN_TRAIN = 25.0         # 训练集的起伏门槛(step2_parallel)
U_UPPER = {'U1': 5.0, 'U2': 15.0, 'U3': 25.0, 'U4': 25.0}


def main():
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    q, B, Bs = d['q_measured'], d['B'], d['B_surface']
    depth, qual = d['borehole_depth'], d['quality_score']
    t = np.load(os.path.join(HERE, 'site_terrain.npz'))
    assert np.allclose(t['lon'], d['lon']) and np.allclose(t['lat'], d['lat'])
    relief = t['relief_m'].astype(float)
    outside = relief < RELIEF_MIN_TRAIN

    qc = q / B
    dq = qc - q
    fallback = ~np.isfinite(depth)
    T = load_terms()
    E = per_site(B, Bs, fallback, T)
    band, sB = E['band'], E['sigma_B']
    sq = qc * sB / B
    bias_q = qc * E['bias_B'] / B
    snr = np.abs(dq) / sq
    grade = np.where(outside, 'C', np.where(snr >= 2, 'A', np.where(snr >= 1, 'B', 'C')))
    ab = np.abs(B - 1)

    n = len(q)
    summ = dict(n=n, terms=T, n_fallback=int(fallback.sum()),
                n_outside_training_relief=int(outside.sum()))
    print('算子(中心像元,对称化有限元参考,按预测 |B_s−1|):')
    for o_, v in T['op'].items():
        print('  %-10s n=%3d  RMSE %.4f  偏差 %+.5f' % (o_, v['n'], v['rmse'], v['bias']))
    print('有限元离散(单向) %.1f%%;回退深度 %.1f%%' % (100 * T['fe_bias'], 100 * T['dep_rel']))
    for nm in ('mesh', 'win'):
        print('  %s: ' % nm + '  '.join('%s 偏差 %+.1f%% 离散 %.1f%%' % (b, 100 * v['bias_rel'], 100 * v['scatter_rel'])
                                       for b, v in T[nm].items()))

    print('\n逐测点分级(n = %s):' % format(n, ','))
    for g, nm in (('A', '校正显著 SNR≥2'), ('B', '可分辨 1≤SNR<2'), ('C', '不可分辨 SNR<1')):
        m = grade == g
        summ['grade_' + g] = dict(
            n=int(m.sum()), pct=float(100 * m.mean()),
            median_abs_corr_pct=float(np.median(100 * np.abs(dq[m]) / q[m])) if m.any() else None,
            median_abs_corr_mW=float(np.median(np.abs(dq[m]))) if m.any() else None)
        print('  %s %-16s %6s 个 (%.1f%%)  |Δq/q| 中位 %.2f%%  |Δq| 中位 %.2f mW m⁻²'
              % (g, nm, format(int(m.sum()), ','), 100 * m.mean(),
                 summ['grade_' + g]['median_abs_corr_pct'], summ['grade_' + g]['median_abs_corr_mW']))
    summ['outside_training_relief_grades'] = {g: int((grade[outside] == g).sum()) for g in 'ABC'}
    print('  其中起伏 < 25 m(训练范围外)%d 个:A %d / B %d / C %d'
          % (outside.sum(), *[(grade[outside] == g).sum() for g in 'ABC']))

    print('\n按 |B−1| 档:')
    summ['by_band'] = {}
    for lo, hi, lab in BANDS:
        m = band == lab
        row = {g: float(100 * (grade[m] == g).mean()) for g in 'ABC'}
        row.update(n=int(m.sum()), median_snr=float(np.median(snr[m])),
                   median_sigma_mW=float(np.median(sq[m])),
                   median_bias_mW=float(np.median(bias_q[m])),
                   median_rel_sigma_pct=float(np.median(100 * sq[m] / qc[m])))
        summ['by_band'][lab] = row
        print('  %-10s n=%6s  A %5.1f%%  B %5.1f%%  C %5.1f%%  SNR 中位 %5.2f  σ_q 中位 %.3f mW m⁻²'
              % (lab, format(int(m.sum()), ','), row['A'], row['B'], row['C'],
                 row['median_snr'], row['median_sigma_mW']))
    ge = ab >= 0.02
    summ['ge002'] = dict(n=int(ge.sum()), pct_A=float(100 * (grade[ge] == 'A').mean()),
                         pct_AB=float(100 * (grade[ge] != 'C').mean()))
    print('  |B−1| ≥ 0.02 的 %s 个测点:A %.1f%%,A+B %.1f%%'
          % (format(int(ge.sum()), ','), summ['ge002']['pct_A'], summ['ge002']['pct_AB']))

    e10 = np.percentile(q, np.arange(10, 100, 10))
    cross = np.digitize(q, e10) != np.digitize(qc, e10)
    summ['decile_cross'] = dict(all_pct=float(100 * cross.mean()),
                                A_n=int((cross & (grade == 'A')).sum()),
                                A_pct_of_all=float(100 * (cross & (grade == 'A')).mean()),
                                AB_pct_of_all=float(100 * (cross & (grade != 'C')).mean()),
                                share_from_C=float(100 * (cross & (grade == 'C')).sum() / max(cross.sum(), 1)),
                                on_boundary_pct=float(100 * np.isin(q, e10).mean()))
    print('\n十分位跨越 %.1f%%;只计 A 级 %.1f%%、A+B 级 %.1f%%;跨越者中 C 级占 %.1f%%;实测值恰在分界上 %.1f%%'
          % (summ['decile_cross']['all_pct'], summ['decile_cross']['A_pct_of_all'],
             summ['decile_cross']['AB_pct_of_all'], summ['decile_cross']['share_from_C'],
             summ['decile_cross']['on_boundary_pct']))

    # 与测点自身的测量不确定度比较
    ucls = np.array([s.split('.')[0] if '.' in s else s for s in qual])
    rel = 100 * np.abs(dq) / q
    summ['vs_measurement'] = {}
    print('\n校正量与测点自报的测量不确定度(IHFC U 等级的变异系数上限):')
    has_u = np.isin(ucls, list(U_UPPER))
    for u, up in U_UPPER.items():
        m = ucls == u
        ex = m & (rel > up)
        summ['vs_measurement'][u] = dict(n=int(m.sum()), n_exceed=int(ex.sum()),
                                         pct_exceed=float(100 * ex.sum() / max(m.sum(), 1)),
                                         upper_pct=up)
        print('  %s(上限 %2.0f%%) n=%5s  校正量超过上限 %4d 个 (%.1f%%)'
              % (u, up, format(int(m.sum()), ','), ex.sum(), 100 * ex.sum() / max(m.sum(), 1)))
    ex_all = has_u & (rel > np.array([U_UPPER.get(u, np.inf) for u in ucls]))
    summ['vs_measurement']['all_rated'] = dict(n=int(has_u.sum()), n_exceed=int(ex_all.sum()),
                                               pct_exceed=float(100 * ex_all.sum() / has_u.sum()))
    summ['vs_measurement']['exceed_and_grade_A'] = int((ex_all & (grade == 'A')).sum())
    print('  合计:有 U 等级的 %s 个测点中,%d 个 (%.1f%%) 的地形校正超过其自身测量不确定度上限,'
          '其中 A 级 %d 个' % (format(int(has_u.sum()), ','), ex_all.sum(),
                              100 * ex_all.sum() / has_u.sum(),
                              summ['vs_measurement']['exceed_and_grade_A']))
    # 稳健口径:先从校正量里扣掉 2 倍方法误差,剩下的仍超过测量不确定度上限才算。
    # 只拿校正量的点估计去比是偏乐观的 —— 校正量本身也有误差。
    upper = np.array([U_UPPER.get(u, np.inf) for u in ucls])
    lo_rel = 100 * np.maximum(np.abs(dq) - 2 * sq, 0) / q
    ex_rob = has_u & (lo_rel > upper)
    u1 = ucls == 'U1'
    summ['vs_measurement']['robust'] = dict(
        n_exceed=int(ex_rob.sum()), pct_exceed=float(100 * ex_rob.sum() / has_u.sum()),
        u1_n_exceed=int((ex_rob & u1).sum()),
        u1_pct_exceed=float(100 * (ex_rob & u1).sum() / max(u1.sum(), 1)))
    print('  稳健口径(|Δq| − 2σ_q 仍超过上限):%d 个 (%.1f%%);其中 U1 级 %d 个 (%.1f%%)'
          % (ex_rob.sum(), 100 * ex_rob.sum() / has_u.sum(), (ex_rob & u1).sum(),
             100 * (ex_rob & u1).sum() / max(u1.sum(), 1)))

    np.savez_compressed(os.path.join(HERE, 'site_reliability.npz'),
                        lon=d['lon'], lat=d['lat'], q_measured=q, q_corrected=qc,
                        relief_m=relief.astype('float32'),
                        sigma_q=sq.astype('float32'), sigma_B=sB.astype('float32'),
                        bias_q=bias_q.astype('float32'),
                        snr=snr.astype('float32'), grade=grade.astype('<U1'),
                        depth_fallback=fallback, outside_training_relief=outside,
                        u_class=ucls.astype('<U2'), exceeds_own_uncertainty=ex_all,
                        exceeds_own_uncertainty_robust=ex_rob)
    json.dump(summ, open(os.path.join(HERE, 'site_reliability.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('\n已保存 site_reliability.npz / site_reliability.json')


if __name__ == '__main__':
    main()
