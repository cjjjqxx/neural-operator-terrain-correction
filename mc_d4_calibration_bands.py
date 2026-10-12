# -*- coding: utf-8 -*-
"""逐点不确定度(8 朝向 + dropout 合成)的**分档**校准检验。

mc_d4_calibration.py 只检验了整体覆盖率(两集迁移后 ±1σ 68–69%)。误差模型若改用逐点值,
近平坦档的信噪比会大幅上升、改变「近平坦档不可分辨」的结论,因此必须先确认每一档都校准良好。

判据(先定、后看):按预测的 |B_s − 1| 分四档(与误差模型 OP_BANDS 相同);缩放系数 c 取整体口径
(另一集上使 ±1σ 覆盖 68.3% 的值,即生产上会用的那个),看每档的 ±1cσ / ±2cσ 覆盖率是否接近
68.3% / 95.4%。另报告每档「按档单独定 c」所需的系数,作为若需分档缩放时的依据。
同时保存逐瓦片数组(mc_d4_calibration_tiles.npz),以后不必再跑 GPU。
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from error_model import OP_BANDS, op_site_scale  # noqa: E402

C = 20                                   # 41×41 输出场的中心像元


def scale_for(sd, e, q=0.683):
    """与 mc_d4_calibration.scale_for 相同:使 ±1σ 覆盖 q 的缩放系数。"""
    return float(np.quantile(e / np.maximum(sd, 1e-9), q))


def main():
    fo = np.load(os.path.join(HERE, 'fem_orientation_test.npz'))
    v = np.load(os.path.join(HERE, 'validation_independent_500.npz'))
    ex = np.load(os.path.join(HERE, 'validation_independent_excl.npz'))
    sets = {'原独立集': (v['dems'], fo['F'][:, :, C, C].mean(1)),
            '排除距离集': (ex['dems'], ex['B_fem'][:, C, C])}
    _tp = os.path.join(HERE, 'mc_d4_calibration_tiles.npz')
    cache = {}
    if os.path.exists(_tp) and not os.environ.get('RERUN_GPU'):
        _t = np.load(_tp)
        for n, name in enumerate(sets):
            cache[name] = {k: _t['%s_%d' % (k, n)] for k in ('P', 'F', 'e', 'sd')}
    else:
        from mc_d4_calibration import run   # 导入即加载模型权重
        for name, (dems, F) in sets.items():
            P, o_sd, d_sd = run(dems)
            cache[name] = dict(P=P, F=F, e=np.abs(P - F), sd=np.hypot(o_sd, d_sd))
        np.savez_compressed(_tp, **{f'{k}_{n}': v2 for n, (k, d) in enumerate(cache.items()) for k, v2 in d.items()})
    names = list(cache)
    c_all = {n: scale_for(cache[n]['sd'], cache[n]['e']) for n in names}
    out = dict(c_overall=c_all, bands={})
    for a, b in ((names[0], names[1]), (names[1], names[0])):
        d = cache[b]
        ab = np.abs(d['P'] - 1)
        for lo, hi, lab in OP_BANDS:
            m = (ab >= lo) & (ab < hi)
            e, sd = d['e'][m], d['sd'][m]
            c = c_all[a]
            row = dict(n=int(m.sum()), c_from_other=c,
                       cov1=float(np.mean(e <= c * sd)), cov2=float(np.mean(e <= 2 * c * sd)),
                       c_band_own=scale_for(sd, e) if m.sum() >= 10 else None,
                       rmse=float(np.sqrt(np.mean(e ** 2))), median_sd_scaled=float(np.median(c * sd)),
                       rms_sd_scaled=float(np.sqrt(np.mean((c * sd) ** 2))))
            out['bands'].setdefault(b, {})[lab] = row
    cm = op_site_scale()
    P = np.concatenate([cache[n]['P'] for n in names]); e = np.concatenate([cache[n]['e'] for n in names])
    sd = np.concatenate([cache[n]['sd'] for n in names]); ab = np.abs(P - 1)
    out['pooled'] = {'c': cm}
    for lo, hi, lab in OP_BANDS:
        m = (ab >= lo) & (ab < hi)
        out['pooled'][lab] = dict(n=int(m.sum()), cov1=float(np.mean(e[m] <= cm * sd[m])),
                                  cov2=float(np.mean(e[m] <= 2 * cm * sd[m])),
                                  c_band_own=scale_for(sd[m], e[m]),
                                  rmse=float(np.sqrt(np.mean(e[m] ** 2))),
                                  rms_sd_scaled=float(np.sqrt(np.mean((cm * sd[m]) ** 2))))
    json.dump(out, open(os.path.join(HERE, 'mc_d4_calibration_bands.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('整体缩放系数 c:', {k: round(v2, 3) for k, v2 in c_all.items()})
    for b, rows in out['bands'].items():
        print('== %s(用另一集的 c)' % b)
        for lab, r in rows.items():
            print('  %-8s n=%4d  ±1σ %5.1f%%  ±2σ %5.1f%%  本档自定 c=%s  RMSE %.4f  缩放后 RMS σ %.4f'
                  % (lab, r['n'], 100 * r['cov1'], 100 * r['cov2'],
                     'n/a' if r['c_band_own'] is None else '%.2f' % r['c_band_own'], r['rmse'], r['rms_sd_scaled']))
    print('== 两集合并(c = %.3f)' % cm)
    for lab, r in out['pooled'].items():
        if lab == 'c':
            continue
        print('  %-8s n=%4d  ±1σ %5.1f%%  ±2σ %5.1f%%  本档自定 c=%.2f  RMSE %.4f  缩放后 RMS σ %.4f'
              % (lab, r['n'], 100 * r['cov1'], 100 * r['cov2'], r['c_band_own'], r['rmse'], r['rms_sd_scaled']))


if __name__ == '__main__':
    main()
