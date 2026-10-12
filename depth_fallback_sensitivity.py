# -*- coding: utf-8 -*-
"""回退钻孔深度的敏感性检验。


B = 1 + f(D) · (B_surface − 1),因此只需替换缺深度点的 f 即可重算。
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# 与 rebuild_correction.py 中 f_depth() 完全一致的深度衰减曲线
_F_D = np.array([0., 100., 146., 250., 390., 500., 1000., 1450., 2000.])
_F_V = np.array([1.000, 0.987, 0.970, 0.929, 0.873, 0.833, 0.677, 0.574, 0.476])
f_of = lambda D: float(np.interp(D, _F_D, _F_V))
MEDIAN_DEPTH_M = 338.0   # 与 rebuild_correction.py 一致


def stats(qm, B):
    """给定 B 场,返回校正量与重分类统计"""
    qc = qm / B
    rel = 100.0 * np.abs(qc - qm) / qm
    dec_m = np.digitize(qm, np.percentile(qm, np.arange(10, 100, 10)))
    dec_c = np.digitize(qc, np.percentile(qm, np.arange(10, 100, 10)))
    return dict(median=float(np.median(rel)), p90=float(np.percentile(rel, 90)),
                p99=float(np.percentile(rel, 99)),
                moved=float(100.0 * (dec_m != dec_c).mean()))


def main():
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    qm, B, Bs, D = d['q_measured'], d['B'], d['B_surface'], d['borehole_depth']
    miss = ~np.isfinite(D)
    print(f'测点 {len(qm):,}  |  缺钻孔深度 {miss.sum():,} ({100 * miss.mean():.1f}%)')
    print(f'实测深度四分位: {np.nanpercentile(D, 25):.0f} / {np.nanmedian(D):.0f} / '
          f'{np.nanpercentile(D, 75):.0f} m\n')

    base = stats(qm, B)
    print(f'{"回退深度":<20}{"f":>7}{"中位校正%":>11}{"P90":>8}{"P99":>8}{"跨十分位%":>11}')
    out = {}
    for lab, Dfb in [('130 m (Q1,浅)', 130.0), ('338 m (中位,本文采用)', 338.0),
                     ('1,250 m (Q3,深)', 1250.0)]:
        f = f_of(Dfb)
        Bv = B.copy()
        Bv[miss] = 1.0 + (B[miss] - 1.0) * (f / f_of(MEDIAN_DEPTH_M))
        s = stats(qm, Bv)
        out[lab] = dict(depth=Dfb, f=f, **s)
        print(f'{lab:<20}{f:>7.3f}{s["median"]:>11.3f}{s["p90"]:>8.2f}'
              f'{s["p99"]:>8.2f}{s["moved"]:>11.2f}')

    # 口径自检:采用档必须与主流水线的 B 完全一致(见上方注释)
    _adopt = out['338 m (中位,本文采用)']['median']
    assert abs(_adopt - base['median']) < 1e-9, (
        '采用档 %.4f 与主流水线 %.4f 不一致 —— 深度换算口径又漂了'
        % (_adopt, base['median']))
    print('  [自检] 采用档与主流水线一致: %.4f' % _adopt)

    lo, hi = out['130 m (Q1,浅)'], out['1,250 m (Q3,深)']
    print(f'\n跨越整个四分位范围,中位校正量变动 '
          f'{abs(hi["median"] - lo["median"]):.3f} 个百分点 '
          f'({100 * abs(hi["median"] - lo["median"]) / base["median"]:.1f}% 相对变化);')
    print(f'跨十分位比例变动 {abs(hi["moved"] - lo["moved"]):.2f} 个百分点。')

    json.dump(out, open(os.path.join(HERE, 'depth_fallback_sensitivity.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('\n已保存 depth_fallback_sensitivity.json')


if __name__ == '__main__':
    main()
