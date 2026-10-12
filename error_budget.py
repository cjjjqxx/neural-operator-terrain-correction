# -*- coding: utf-8 -*-
"""表 1 误差预算:逐测点误差(error_model.py)按 |B_D − 1| 五档取中位,以 mW m⁻² 表示。

"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from error_model import BANDS, load_terms, per_site  # noqa: E402

J = lambda f: json.load(open(os.path.join(HERE, f), encoding='utf-8'))
LAB = {'<0.02': '< 0.02', '0.02-0.05': '0.02–0.05', '0.05-0.10': '0.05–0.10',
       '0.10-0.20': '0.10–0.20', '>0.20': '> 0.20'}


def render_table1_rows(eb=None):
    """稿件表 1 的五行数据,由 error_budget.json 渲染;check_pipeline_consistency 用它逐字比对。"""
    eb = eb or J('error_budget.json')
    rows = []
    for key in (b[2] for b in BANDS):
        v = eb[key]
        rows.append('| %s | %s | %.2f | %.3f | %.3f | %.3f | %.3f | **%.2f** | %+.2f | %.1f |' % (
            LAB[key], format(v['n'], ','), v['correction_mW'], v['s_op_mW'], v['s_mesh_mW'],
            v['s_win_mW'], v['s_dep_mW'], v['sigma_mW'], v['bias_mW'], v['snr_median']))
    return rows


def main():
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    q, B, Bs, depth = d['q_measured'], d['B'], d['B_surface'], d['borehole_depth']
    fallback = ~np.isfinite(depth)
    T = load_terms()
    E = per_site(B, Bs, fallback, T)
    qc = q / B
    to_q = qc / B                                   # dB → dq
    corr = np.abs(qc - q)
    out = dict(terms=dict(fe_bias_rel=T['fe_bias'], dep_rel=T['dep_rel'],
                          op={k: v for k, v in T['op'].items()},
                          mesh=T['mesh'], win=T['win']))
    print('%-10s %6s %8s %7s %7s %7s %7s %7s %8s %6s' % ('|B_D−1|', 'n', '校正量', '算子', '网格', '窗口',
                                                    '深度', '随机σ', '系统偏差', 'SNR'))
    for lo, hi, lab in BANDS:
        m = E['band'] == lab
        med = lambda x: float(np.median(x[m]))
        rms = lambda x: float(np.sqrt(np.mean(x[m] ** 2)))
        avg = lambda x: float(np.mean(x[m]))
        row = dict(n=int(m.sum()), correction_mW=med(corr),
                   s_op_mW=rms(E['s_op'] * to_q), s_mesh_mW=rms(E['s_mesh'] * to_q),
                   s_win_mW=rms(E['s_win'] * to_q), s_dep_mW=rms(E['s_dep'] * to_q),
                   sigma_mW=rms(E['sigma_B'] * to_q),
                   bias_fe_mW=avg(E['b_fe'] * to_q), bias_mesh_mW=avg(E['b_mesh'] * to_q),
                   bias_win_mW=avg(E['b_win'] * to_q), bias_mW=avg(E['bias_B'] * to_q),
                   correction_mean_mW=avg(corr),
                   snr_median=med(corr / (E['sigma_B'] * to_q)),
                   frac_fallback=float(fallback[m].mean()))
        row['sigma_over_corr_pct'] = 100 * row['sigma_mW'] / row['correction_mean_mW']
        row['bias_over_corr_pct'] = 100 * row['bias_mW'] / row['correction_mean_mW']
        out[lab] = row
        print('%-10s %6d %8.3f %7.3f %7.3f %7.3f %7.3f %7.3f %+8.3f %6.2f' % (
            lab, row['n'], row['correction_mW'], row['s_op_mW'], row['s_mesh_mW'], row['s_win_mW'],
            row['s_dep_mW'], row['sigma_mW'], row['bias_mW'], row['snr_median']))
    print('\n随机误差占校正量:' + '  '.join('%s %.0f%%' % (b[2], out[b[2]]['sigma_over_corr_pct']) for b in BANDS))
    print('系统偏差占校正量:' + '  '.join('%s %+.0f%%' % (b[2], out[b[2]]['bias_over_corr_pct']) for b in BANDS))
    json.dump(out, open(os.path.join(HERE, 'error_budget.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('\n已保存 error_budget.json')


if __name__ == '__main__':
    main()
