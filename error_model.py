# -*- coding: utf-8 -*-
"""唯一的误差模型 —— 逐测点可信度(site_reliability.py)与误差预算表 1(error_budget.py)共用。


每个测点的方法误差分两类,分开报告、不混加:


系统项(单向,生产配置低估 |B − 1| 的相对量)——不进入 σ_B,单列:
  b_FE    有限元离散:解析基准在流水线自身的 D_0(域深/10)口径下的残余误差,取水平分辨率
          充足且域深不构成约束的两个波长(7.1、9.5 km)的均值(analytic_benchmark_wavelength)
  b_mesh  水平网格截断:n_x 40 → 128 变化量的中位,除以该档中位 |B − 1|
  b_win   窗口截断:57 → 160 km,同上

分档变量一律是深度平均后的 |B_D − 1|(即实际施加的校正所用的 B)。算子误差表的查表键是
预测的 |B_s − 1| —— 算子输出的是地表值,应用时手里只有它。
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(HERE, f), encoding='utf-8'))

BANDS = [(0.0, 0.02, '<0.02'), (0.02, 0.05, '0.02-0.05'), (0.05, 0.10, '0.05-0.10'),
         (0.10, 0.20, '0.10-0.20'), (0.20, 9.0, '>0.20')]
OP_BANDS = [(0.0, 0.02, '<0.02'), (0.02, 0.05, '0.02-0.05'), (0.05, 0.10, '0.05-0.10'),
            (0.10, 9.0, '>=0.10')]
MESH_KEY = {'<0.02': '0.0-0.02', '0.02-0.05': '0.02-0.05', '0.05-0.10': '0.05-0.1',
            '0.10-0.20': '0.1-0.2', '>0.20': '0.2-9.0'}
WIN_KEY = {'<0.02': '0.05-0.1', '0.02-0.05': '0.05-0.1', '0.05-0.10': '0.05-0.1',
           '0.10-0.20': '0.1-0.2', '>0.20': '0.2-9.0'}
FALLBACK_DEPTH_M = 338.0


def band_of(x, bands=BANDS):
    out = np.empty(np.shape(x), dtype=object)
    for lo, hi, lab in bands:
        out[(x >= lo) & (x < hi)] = lab
    return out


def _robust_sd(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def operator_table():
    """中心像元、对称化有限元参考;按预测 |B_s − 1| 分档。"""
    v = np.load(os.path.join(HERE, 'fem_orientation_test.npz'))
    F, P = v['F'].astype(float), v['P'].astype(float)
    c = F.shape[-1] // 2
    Fs, P0 = F[:, :, c, c].mean(1), P[:, :, c, c].mean(1)
    ab = np.abs(P0 - 1)
    tab = {}
    for lo, hi, lab in OP_BANDS:
        m = (ab >= lo) & (ab < hi)
        e = P0[m] - Fs[m]
        tab[lab] = dict(n=int(m.sum()), rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)))
    return tab


def truncation_terms(fname, key_new, key_ref, summ_key_map):
    """逐点变化量 δ = |B_new − 1| − |B_ref − 1|;偏差 = 中位 δ / 中位 |B_ref − 1|,离散 = 稳健 SD / 同一分母。"""
    rows = J(fname)['rows']
    out = {}
    for lab, key in summ_key_map.items():
        rr = [r for r in rows if r['band'] == key]
        ref = np.array([abs(r['B'][key_ref] - 1) for r in rr])
        new = np.array([abs(r['B'][key_new] - 1) for r in rr])
        d = new - ref
        den = float(np.median(ref))
        out[lab] = dict(n=len(rr), bias_rel=float(np.median(d)) / den, scatter_rel=_robust_sd(d) / den,
                        ref_median=den)
    return out


def fe_bias():
    rows = J('analytic_benchmark_wavelength.json')['rows']
    v = [abs(r['err_vs_D0_pct']) for r in rows if round(r['lambda_km'], 1) in (7.1, 9.5)]
    return float(np.mean(v)) / 100.0


def depth_rel():
    s = J('depth_fallback_sensitivity.json')
    f = {round(v['depth']): v['f'] for v in s.values()}
    return (f[130] - f[1250]) / 2.0 / f[338]



_SITE_SD = None


def op_site_scale():
    """保守缩放系数:两独立集合并后使 ±2σ 覆盖 95.4% 的 c(即 quantile(|e|/sd, 0.954) / 2)。"""
    t = np.load(os.path.join(HERE, 'mc_d4_calibration_tiles.npz'))
    e = np.concatenate([t['e_0'], t['e_1']])
    sd = np.concatenate([t['sd_0'], t['sd_1']])
    return float(np.quantile(e / sd, 0.954) / 2.0)


def site_op_sd(Bs):
    """逐测点算子离散度(地表 B 单位,未缩放);与输入的 B_s 逐点核对对齐,对不上返回 None。"""
    global _SITE_SD
    if _SITE_SD is None:
        p = os.path.join(HERE, 'mc_d4_uncertainty.npz')
        _SITE_SD = np.load(p) if os.path.exists(p) else False
    if _SITE_SD is False or len(_SITE_SD['combined_sd']) != len(Bs):
        return None
    if np.max(np.abs(_SITE_SD['B_s_d4'] - Bs)) > 1e-3:
        return None
    return _SITE_SD['combined_sd']

def load_terms():
    return dict(op=operator_table(), op_site_c=op_site_scale(),
                mesh=truncation_terms('mesh_convergence_large.json', '128', '40', MESH_KEY),
                win=truncation_terms('footprint_fixed_resolution_test.json', '160', '57', WIN_KEY),
                fe_bias=fe_bias(), dep_rel=depth_rel())


def per_site(B, Bs, fallback, T=None):
    """返回逐测点的各分项(B 的单位)与合成量。B 为深度平均 B_D,Bs 为地表值。"""
    T = T or load_terms()
    ab = np.abs(B - 1)
    band = band_of(ab)
    nz = np.abs(Bs - 1) > 1e-6
    k = np.abs(np.where(nz, (B - 1) / np.where(nz, Bs - 1, 1), 1.0))
    ob = band_of(np.abs(Bs - 1), OP_BANDS)
    sd = site_op_sd(Bs)
    if sd is not None:
        s_op = T['op_site_c'] * sd * k
        src = 'site'
    else:
        print('⚠ error_model:逐点算子离散度缺失或未对齐,退回分档 RMSE')
        s_op = np.array([T['op'][o]['rmse'] for o in ob]) * k
        src = 'band'
    s_mesh = np.array([T['mesh'][b]['scatter_rel'] for b in band]) * ab
    s_win = np.array([T['win'][b]['scatter_rel'] for b in band]) * ab
    s_dep = np.where(fallback, T['dep_rel'] * ab, 0.0)
    sB = np.sqrt(s_op ** 2 + s_mesh ** 2 + s_win ** 2 + s_dep ** 2)
    b_fe = T['fe_bias'] * ab
    b_mesh = np.array([T['mesh'][b]['bias_rel'] for b in band]) * ab
    b_win = np.array([T['win'][b]['bias_rel'] for b in band]) * ab
    bias = b_fe + b_mesh + b_win                    # 带符号:正 = 生产配置低估 |B − 1|
    return dict(band=band, s_op_source=src, s_op=s_op, s_mesh=s_mesh, s_win=s_win, s_dep=s_dep, sigma_B=sB,
                b_fe=b_fe, b_mesh=b_mesh, b_win=b_win, bias_B=bias)
