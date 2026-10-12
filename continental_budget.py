# -*- coding: utf-8 -*-
"""大陆热流收支:校正前后网格化场的面积加权平均,以及点汇编上的带符号变化。


网格场按 cos(纬度) 做面积加权;克里金结果在非陆地格点为 NaN,自然构成陆地掩膜。
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    k = np.load(os.path.join(HERE, 'kriged_heatflow_05deg_global.npz'))
    qm, qc, lats = k['krige_qm'], k['krige_qc'], k['lats']
    w = np.cos(np.radians(lats))[:, None] * np.ones_like(qm)
    m = np.isfinite(qm) & np.isfinite(qc)
    a = float(np.average(qm[m], weights=w[m]))
    b = float(np.average(qc[m], weights=w[m]))
    print(f'网格化场(面积加权,{m.sum():,} 个陆地格点)')
    print(f'  实测 {a:.2f} → 校正后 {b:.2f} mW m⁻²   '
          f'变化 {b - a:+.3f} mW m⁻² ({100 * (b - a) / a:+.2f}%)')

    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    sg = d['q_corrected'] - d['q_measured']
    B = d['B']
    n = len(sg)
    se = sg.std(ddof=1) / np.sqrt(n)
    p = 2 * stats.norm.sf(abs(sg.mean() / se))
    print(f'\n点汇编({n:,} 个测点)')
    print(f'  带符号变化均值 {sg.mean():+.3f} mW m⁻²  '
          f'95% CI {sg.mean() - 1.96 * se:+.3f} 至 {sg.mean() + 1.96 * se:+.3f}   p = {p:.0e}')
    print(f'  B > 1(谷地,下调) {100 * (B > 1).mean():.1f}%   '
          f'B < 1(山脊,上调) {100 * (B < 1).mean():.1f}%')

    json.dump(dict(grid_measured=a, grid_corrected=b, grid_delta=b - a,
                   grid_delta_pct=100 * (b - a) / a, n_land_cells=int(m.sum()),
                   point_mean=float(sg.mean()), point_ci=[float(sg.mean() - 1.96 * se),
                                                          float(sg.mean() + 1.96 * se)],
                   point_p=float(p), n_sites=n,
                   frac_B_gt1=float(100 * (B > 1).mean())),
              open(os.path.join(HERE, 'continental_budget.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('\n已保存 continental_budget.json')


if __name__ == '__main__':
    main()
