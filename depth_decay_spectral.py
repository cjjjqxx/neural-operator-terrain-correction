# -*- coding: utf-8 -*-
"""刻画 40 条深度衰减剖面的波长内容与 e 折深度,并检验二者是否相关。


不需要重跑有限元:depth_decay_profiles.json 已存完整 B(z) 剖面,
global_fem_dataset.npz 按瓦片索引可取回同一块 DEM。

用法
----
    python depth_decay_spectral.py
"""
import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
CELL_M = 57000.0 / 128          # 与 depth_decay_characterize.py 一致


def centroid_wavelength_m(dem, cell_m):
    """DEM 的功率谱质心波长。

    定义:对二维功率谱做径向合并,以功率为权重对波长求加权平均,
        lambda_c = sum(P_k * lambda_k) / sum(P_k),  lambda_k = 1 / |k|
    直流分量(k = 0)不参与,因为它只反映平均高程而非起伏。
    """
    a = np.asarray(dem, dtype=float)
    a = a - a.mean()
    n = a.shape[0]
    # 加窗抑制瓦片边界的截断泄漏,否则谱在长波端被人为抬高
    w = np.hanning(n)
    a = a * w[:, None] * w[None, :]
    P = np.abs(np.fft.fft2(a)) ** 2
    fx = np.fft.fftfreq(n, d=cell_m)
    KX, KY = np.meshgrid(fx, fx, indexing='ij')
    K = np.sqrt(KX ** 2 + KY ** 2)
    m = K > 0
    lam = 1.0 / K[m]
    p = P[m]
    return float((p * lam).sum() / p.sum())


def efold_depth_m(depths, Bs, B_surf):
    """扰动衰减到地表值 1/e 处的深度(线性内插);未衰减到则返回 None。"""
    z = np.asarray(depths, dtype=float)
    dev = (np.asarray(Bs, dtype=float) - 1.0) / (B_surf - 1.0)
    tgt = 1.0 / np.e
    if dev[0] < tgt or not (dev < tgt).any():
        return None
    k = int(np.argmax(dev < tgt))
    if k == 0:
        return None
    z0, z1, d0, d1 = z[k - 1], z[k], dev[k - 1], dev[k]
    return float(z0 + (d0 - tgt) / (d0 - d1) * (z1 - z0))


def spearman(x, y):
    """Spearman 秩相关与双侧 p 值(t 近似,n = 40 时足够)。"""
    from scipy import stats
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def main():
    prof = json.load(open(os.path.join(HERE, 'depth_decay_profiles.json'),
                          encoding='utf-8'))
    ds = np.load(os.path.join(HERE, 'global_fem_dataset.npz'), mmap_mode='r')
    X = ds['X']

    rows = []
    for r in prof:
        ef = efold_depth_m(r['depths'], r['Bs'], r['B_surf'])
        if ef is None:
            continue
        dem = np.asarray(X[r['idx']]).squeeze()
        rows.append(dict(idx=int(r['idx']), band=r['band'],
                         relief_m=float(r['relief']),
                         lambda_c_m=centroid_wavelength_m(dem, CELL_M),
                         efold_m=ef))

    # 分档一致性 —— 这才是"单条 f(D) 曲线够用"的直接证据。
    # 原稿绕道"e 折深度跨度小"来论证,而实际跨度是 168–3,720 m,并不小;
    # 档间极差则确实小,且正落在钻孔实际采样的深度范围内。
    spread = {}
    for D in ('250', '390', '500', '1000'):
        by_band = {}
        for r in prof:
            if D in r['f']:
                by_band.setdefault(r['band'], []).append(r['f'][D])
        if len(by_band) >= 2:
            med = [float(np.median(v)) for v in by_band.values()]
            spread[D] = round(max(med) - min(med), 3)

    _abp = os.path.join(HERE, 'analytic_benchmark_wavelength.json')
    if os.path.exists(_abp):
        _ab = json.load(open(_abp, encoding='utf-8'))
        _x = np.array([r['domain_over_efold'] for r in _ab['rows']])
        _y = np.array([abs(r['err_vs_depthavg_pct']) for r in _ab['rows']])
        _o = np.argsort(_x)
        # 与流水线同一口径(参照深度 = D_0 = 域深/10)的离散误差;同一波长两个振幅取均值。
        # 超出基准扫描上界的剖面取端点值(np.interp 在端点外取常数),不外推。
        _lam = sorted({r['lambda_km'] for r in _ab['rows']})
        _xd = np.array([np.mean([r['domain_over_efold'] for r in _ab['rows'] if r['lambda_km'] == l]) for l in _lam])
        _yd = np.array([np.mean([abs(r['err_vs_D0_pct']) for r in _ab['rows'] if r['lambda_km'] == l]) for l in _lam])
        _od = np.argsort(_xd)
        for r in rows:
            _dom = max(3000.0, 1.5 * r['relief_m'])
            r['domain_over_efold'] = _dom / r['efold_m']
            r['err_est_pct'] = float(np.interp(r['domain_over_efold'], _x[_o], _y[_o]))
            r['err_D0_pct'] = float(np.interp(r['domain_over_efold'], _xd[_od], _yd[_od]))

    lam = np.array([r['lambda_c_m'] for r in rows]) / 1000.0    # km
    ef = np.array([r['efold_m'] for r in rows])
    r_s, p_s = spearman(lam, ef)

    out = dict(
        n=len(rows),
        lambda_c_km_min=round(float(lam.min()), 1),
        lambda_c_km_max=round(float(lam.max()), 1),
        lambda_c_km_median=round(float(np.median(lam)), 1),
        efold_m_min=int(round(ef.min())),
        efold_m_max=int(round(ef.max())),
        efold_m_median=int(round(np.median(ef))),
        # 口径说明:解析基准只扫到 domain/efold = 2.65,超出者无法插值,
        # 只能报至少已达基准末端的水平。不要对超出区间的点强行插值再取中位,
        # 那会把 23/40 个点钳在端点上,得到一个既非实测也非外推的数。
        domain_over_efold_median=round(float(np.median(
            [r['domain_over_efold'] for r in rows])), 1),
        n_beyond_benchmark=int(sum(r['domain_over_efold'] > 2.65 for r in rows)),
        err_at_benchmark_end_pct=round(float(min(
            abs(q['err_vs_depthavg_pct']) for q in json.load(
                open(os.path.join(HERE, 'analytic_benchmark_wavelength.json'),
                     encoding='utf-8'))['rows'])), 1),
        n_domain_lt15=int(sum(r['domain_over_efold'] < 1.5 for r in rows)),
        frac_domain_lt15_pct=round(100.0 * float(np.mean(
            [r['domain_over_efold'] < 1.5 for r in rows])), 0),
        err_lt15_lo_pct=round(float(min(
            r['err_D0_pct'] for r in rows if r['domain_over_efold'] < 1.5)), 0),
        err_lt15_hi_pct=round(float(max(
            r['err_D0_pct'] for r in rows if r['domain_over_efold'] < 1.5)), 0),
        band_spread=spread,
        # 有限元离散误差(对 |B−1| 的单向低估),参照深度与流水线的 D_0 一致;误差预算取中位
        fe_D0_median_pct=round(float(np.median([r['err_D0_pct'] for r in rows])), 1),
        fe_D0_p90_pct=round(float(np.percentile([r['err_D0_pct'] for r in rows], 90)), 1),
        fe_D0_max_pct=round(float(max(r['err_D0_pct'] for r in rows)), 1),
        n_efold_above_360=int((ef > 360).sum()),
        efold_two_shallowest=[int(round(x)) for x in np.sort(ef)[:2]],
        spearman_r=round(r_s, 2),
        spearman_p=round(p_s, 2),
        rows=rows,
    )
    json.dump(out, open(os.path.join(HERE, 'depth_decay_spectral.json'), 'w',
                        encoding='utf-8'), indent=2, ensure_ascii=False)

    print('剖面 %d 条' % out['n'])
    print('  谱质心波长   %.1f – %.1f km(中位 %.1f）'
          % (out['lambda_c_km_min'], out['lambda_c_km_max'],
             out['lambda_c_km_median']))
    print('  e 折深度     %d – %d m(中位 %d)'
          % (out['efold_m_min'], out['efold_m_max'], out['efold_m_median']))
    print('  Spearman     r = %.2f, p = %.2f' % (out['spearman_r'], out['spearman_p']))
    print('  档间极差     ' + '  '.join('%s m: %.3f' % (k, v)
                                     for k, v in sorted(spread.items(), key=lambda x: int(x[0]))))
    print('  e 折 > 360 m  %d / %d' % (out['n_efold_above_360'], out['n']))
    print('  域深覆盖     中位 %.1f 个 e 折;其中 %d/%d 条超出解析基准扫描上界(2.65)'
          % (out['domain_over_efold_median'], out['n_beyond_benchmark'], out['n']))
    print('  基准末端误差 %.1f%%(域深更大只会更小)' % out['err_at_benchmark_end_pct'])
    print('  域深 < 1.5 个 e 折 %d 条(%.0f%%),预期误差 %.0f%%–%.0f%%'
          % (out['n_domain_lt15'], out['frac_domain_lt15_pct'],
             out['err_lt15_lo_pct'], out['err_lt15_hi_pct']))
    print('\n已保存 depth_decay_spectral.json')


if __name__ == '__main__':
    main()
