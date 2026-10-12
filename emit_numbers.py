# -*- coding: utf-8 -*-
"""从流水线产物生成 numbers.json —— 稿件中每一个数字的唯一来源。


新增或修改稿件中的数字时,先在这里加一个键,再在正文使用它。
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda n: json.load(open(os.path.join(HERE, n), encoding='utf-8'))


def main():
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    qm, qc, B, Bs, D = (d['q_measured'], d['q_corrected'], d['B'],
                        d['B_surface'], d['borehole_depth'])
    qual = np.array([str(x) for x in d['quality_score']])
    n = len(qm)
    rel = 100.0 * np.abs(qc - qm) / qm
    ab = np.abs(B - 1.0)
    N = {}

    N['n_sites'] = n
    N['median_correction_pct'] = round(float(np.median(rel)), 3)
    for p in (90, 95, 99):
        N[f'p{p}_correction_pct'] = round(float(np.percentile(rel, p)), 2)
    N['frac_gt10pct'] = round(float(100 * (rel > 10).mean()), 2)
    N['median_absB1'] = round(float(np.median(ab)), 4)
    N['frac_B_gt1_pct'] = round(float(100 * (B > 1).mean()), 2)

    # 分档计数
    for lo, hi, k in [(0, .02, 'lt002'), (.02, .05, '002_005'), (.05, .10, '005_010'),
                      (.10, .20, '010_020'), (.20, 9, 'gt020')]:
        N[f'n_band_{k}'] = int(((ab >= lo) & (ab < hi)).sum())

    # 重分类
    for nq, lab in [(10, 'decile'), (4, 'quartile')]:
        e = np.percentile(qm, np.arange(100 / nq, 100, 100 / nq))
        N[f'frac_change_{lab}_pct'] = round(
            float(100 * (np.digitize(qm, e) != np.digitize(qc, e)).mean()), 2)

    # 地表 vs 深度平均
    relS = 100.0 * np.abs(qm / Bs - qm) / qm
    N['surface_median_pct'] = round(float(np.median(relS)), 3)
    N['surface_p99_pct'] = round(float(np.percentile(relS, 99)), 2)

    # 缺深度比例
    N['frac_no_depth_pct'] = round(float(100 * (~np.isfinite(D)).mean()), 1)
    N['n_no_depth'] = int((~np.isfinite(D)).sum())
    N['median_reported_depth_m'] = int(round(float(np.nanmedian(D))))

    # 质量子集
    sub = ~np.char.startswith(qual, 'Ux')
    N['quality_subset_pct'] = round(float(100 * sub.mean()), 1)
    N['n_quality_subset'] = int(sub.sum())
    for p, k in [(90, 'p90'), (95, 'p95'), (99, 'p99')]:
        N[f'quality_{k}_subset'] = round(float(np.percentile(rel[sub], p)), 1)
        N[f'quality_{k}_all'] = round(float(np.percentile(rel, p)), 1)
    N['quality_median_subset'] = round(float(np.median(rel[sub])), 1)
    N['quality_median_all'] = round(float(np.median(rel)), 1)
    N['quality_gt10_subset'] = round(100 * float((rel[sub] > 10).mean()), 1)
    N['quality_gt10_all'] = round(100 * float((rel > 10).mean()), 1)
    N['quality_mean_subset'] = round(float(rel[sub].mean()), 1)
    N['quality_mean_all'] = round(float(rel.mean()), 1)

    import re as _re_cfg

    def _const(path, name, cast=float):
        with open(os.path.join(HERE, path), encoding='utf-8') as _fh:
            _m = _re_cfg.search(r'^%s\s*=\s*([\d.]+)' % name, _fh.read(), _re_cfg.M)
        return cast(_m.group(1)) if _m else None

    _res = _const('tile_reader.py', 'RES', int)
    _tile_km = _const('tile_reader.py', 'TILE_KM')
    _fn = _const('step2_parallel.py', 'FN', int)
    if _res and _tile_km and _fn:
        N['cfg_tile_km'] = int(_tile_km)                    # 瓦片边长 57 km
        N['cfg_input_px'] = _res                            # 输入 128×128
        N['cfg_fem_cells'] = _fn                            # 每边 40 个单元
        N['cfg_fem_nodes'] = _fn + 1                        # 输出 41×41
        N['cfg_cell_m'] = int(round(_tile_km * 1000 / _fn))  # 水平单元 1,425 m
        N['cfg_dem_cell_m'] = int(round(_tile_km * 1000 / _res))  # 重采样像元 445 m
    with open(os.path.join(HERE, 'step2_parallel.py'), encoding='utf-8') as _fh:
        _s2 = _fh.read()
    _mz = _re_cfg.search(r'Fem3D\(nx=FN, ny=FN, nz=(\d+)\)', _s2)
    if _mz:
        N['cfg_fem_layers'] = int(_mz.group(1))             # 垂向 10 层
    _md = _re_cfg.search(r'depth=max\((\d+), relief \* ([\d.]+)\)', _s2)
    if _md:
        N['cfg_depth_min_m'] = int(_md.group(1))            # 域深下限 3,000 m
        N['cfg_depth_relief_factor'] = float(_md.group(2))  # 1.5 倍起伏
    _mt = _re_cfg.search(r'int\(h \* ([\d.]+)\)', _s2)
    if _mt:
        N['cfg_taper_pct'] = int(round(100 * float(_mt.group(1))))  # 锥化 12%

    try:
        from sklearn.cluster import KMeans as _KM
        _fd = np.load(os.path.join(HERE, 'global_fem_dataset.npz'))
        _lab = _KM(n_clusters=12, random_state=42, n_init='auto').fit_predict(
            np.column_stack([_fd['lon'], _fd['lat']]))
        _ist = np.isin(_lab, [0, 4, 8])
        _rl = _fd['relief']
        N['relief_train_median_m'] = int(round(float(np.median(_rl[~_ist]))))
        N['relief_holdout_median_m'] = int(round(float(np.median(_rl[_ist]))))
    except Exception:
        pass

    # 窗口试验的总范围:窄端与宽端分别存在两个产物里,正文合起来说
    _wins = []
    for _f in ('footprint_narrow_test.json', 'footprint_fixed_resolution_test.json'):
        _p = os.path.join(HERE, _f)
        if os.path.exists(_p):
            with open(_p, encoding='utf-8') as _fh:
                _wins += list(json.load(_fh).get('windows_km', []))
    if _wins:
        N['win_km_min'] = int(min(_wins))
        N['win_km_max'] = int(max(_wins))

    # 网格场上的相对校正量:图 8 图注引其极值与集中程度
    _kg = np.load(os.path.join(HERE, 'kriged_heatflow_05deg_global.npz'))
    _a, _b = _kg['krige_qm'], _kg['krige_qc']
    _mk = np.isfinite(_a) & np.isfinite(_b) & (_a != 0)
    _r = 100.0 * (_b[_mk] - _a[_mk]) / _a[_mk]
    N['grid_corr_min_pct'] = round(abs(float(_r.min())), 1)  # 正文写「−29.7%」,负号在捕获组外
    N['grid_corr_max_pct'] = round(float(_r.max()), 1)
    N['grid_corr_within_05_pct'] = round(100.0 * float((np.abs(_r) < 0.5).mean()), 1)

    # 图 7a 画的是逐测点值,不是网格场 —— 图注一度混用了上面三个网格量,
    # 把最大校正低报了三分之一(+61.0 对 +90.4)。测点侧单独出键。
    _pt = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    _pr = 100.0 * (_pt['q_corrected'] - _pt['q_measured']) / _pt['q_measured']
    N['point_corr_min_pct'] = round(abs(float(_pr.min())), 1)   # 正文写「−30.8%」
    N['point_corr_max_pct'] = round(float(_pr.max()), 1)
    N['point_corr_within_05_pct'] = round(100.0 * float((np.abs(_pr) < 0.5).mean()), 1)
    N['point_corr_gt30_n'] = int((_pr > 30.0).sum())

    # 高起伏地带的观测覆盖(3.2 用):图 7a 的带状分布同时反映地形与钻孔分布,
    # 要说清这一点需要「高起伏格点里有多大比例根本没有测点」。
    # 覆盖判据与附图 S5 一致:格内含至少一个实测点。
    _rl = np.load(os.path.join(HERE, 'global_relief_05deg.npz'))['relief']
    _kq = np.load(os.path.join(HERE, 'kriged_heatflow_05deg_global.npz'))['krige_qm']
    _land = np.isfinite(_kq)
    _ptc = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    _iy = np.clip(((_ptc['lat'] + 90) / 0.5).astype(int), 0, _rl.shape[0] - 1)
    _ix = np.clip(((_ptc['lon'] + 180) / 0.5).astype(int), 0, _rl.shape[1] - 1)
    _has = np.zeros(_rl.shape, bool)
    _has[_iy, _ix] = True
    _hi = _land & np.isfinite(_rl) & (_rl > 500)
    N['relief_gt500_land_pct'] = round(100.0 * _hi.sum() / _land.sum(), 1)
    N['relief_gt500_noobs_pct'] = round(100.0 * (1 - (_hi & _has).sum() / _hi.sum()), 1)

    _sb = os.path.join(HERE, 'spatial_split_relief_bins.json')
    if os.path.exists(_sb):
        with open(_sb, encoding='utf-8') as _fh:
            _sbd = json.load(_fh)
        N['split_r2_flat'] = round(_sbd['r2_by_relief'][0], 3)
        N['split_r2_rugged'] = round(_sbd['r2_by_relief'][-1], 3)

    _dc = os.path.join(HERE, 'dropout_calibration_test.json')
    if os.path.exists(_dc):
        with open(_dc, encoding='utf-8') as _fh:
            _dcd = json.load(_fh)
        N['dropout_cov_1sigma'] = _dcd['coverage_1sigma']
        N['dropout_cov_2sigma'] = _dcd['coverage_2sigma']

    # 大陆收支
    k = np.load(os.path.join(HERE, 'kriged_heatflow_05deg_global.npz'))
    gm, gc, lats = k['krige_qm'], k['krige_qc'], k['lats']
    w = np.cos(np.radians(lats))[:, None] * np.ones_like(gm)
    m = np.isfinite(gm) & np.isfinite(gc)
    a_, b_ = float(np.average(gm[m], weights=w[m])), float(np.average(gc[m], weights=w[m]))
    N['grid_mean_measured'] = round(a_, 2)
    N['grid_mean_corrected'] = round(b_, 2)
    N['grid_delta'] = round(b_ - a_, 3)
    N['grid_delta_pct'] = round(100 * (b_ - a_) / a_, 2)

    _fb = os.path.join(HERE, 'fem_speed_benchmark.json')
    if os.path.exists(_fb):
        with open(_fb, encoding='utf-8') as _fh:
            _f = json.load(_fh)
        N['fem_tile_ms'] = _f['fem_total_ms']
        N['fem_build_ms'] = _f['fem_build_ms']
        N['fem_solve_ms'] = _f['fem_solve_ms']
        N['fem_n_dof'] = _f['n_dof']
        # 正文以秒叙述、产物以毫秒记录。单位换算放在这里,不放进正则 ——
        # 正则里做换算,读者改一个量纲就会失配,而失配读起来像数字错了。
        N['fem_tile_s'] = round(_f['fem_total_ms'] / 1000.0, 2)
        N['fem_build_s'] = round(_f['fem_build_ms'] / 1000.0, 2)
        N['fem_solve_s'] = round(_f['fem_solve_ms'] / 1000.0, 2)
    _mc = os.path.join(HERE, 'mesh_convergence_large.json')
    if os.path.exists(_mc):
        _m = json.load(open(_mc, encoding='utf-8'))['timing_s']
        N['fem_tile_s_nx40_mc'] = round(_m['40']['median'], 1)
        N['fem_tile_s_nx128'] = round(_m['128']['median'], 0)
        N['fem_days_sites_nx128'] = round(_m['128']['median'] * 36392 / 86400.0, 0)
    _nb = os.path.join(HERE, 'fno_speed_benchmark.json')
    if os.path.exists(_nb):
        with open(_nb, encoding='utf-8') as _fh:
            _n = json.load(_fh)
        N['fno_single_ms'] = _n['fno_single_ms']      # 单次调用,与 FEM 单瓦片对等
        N['fno_batch_ms'] = _n['fno_batch_ms']        # 批量摊薄,全球场的实际成本
        N['fno_cpu_single_ms'] = _n.get('cpu_batch1_per_tile_ms')
        N['fno_cpu_batch_ms'] = _n.get('cpu_batch32_per_tile_ms')
        # 正文以秒叙述 CPU 那一行,换算放这里而不是正则里
        if _n.get('cpu_batch1_per_tile_ms'):
            N['fno_cpu_single_s'] = round(_n['cpu_batch1_per_tile_ms'] / 1000.0, 2)
    # 全球陆地 B 场
    _gb = os.path.join(HERE, 'global_B_field_05deg.json')
    if os.path.exists(_gb):
        with open(_gb, encoding='utf-8') as _fh:
            _g = json.load(_fh)
        N['gB_cells'] = _g['n_solved']
        N['gB_minutes'] = _g['minutes']
        N['gB_frac_dev_gt2pct'] = _g['frac_dev_gt2pct']
        N['gB_frac_dev_gt10pct'] = _g['frac_dev_gt10pct']
        N['gB_median'] = _g['B_median']
        N['gB_p1'] = _g['B_p1']
        N['gB_p99'] = _g['B_p99']
        # 墙钟 2.8 h 与纯推理 8.5 min 是两个口径:前者含逐格点读高程,
        # 后者不含。正文与图注各引一个,两处都要能核。
        N['gB_wall_hours'] = round(_g['minutes'] / 60.0, 1)
    _gs = os.path.join(HERE, 'global_B_field_stats.json')
    if os.path.exists(_gs):
        with open(_gs, encoding='utf-8') as _fh:
            _s = json.load(_fh)
        N['gB_n_01deg'] = _s['n_cells_01']
        N['gB_dev10_relief_gt500'] = _s['dev_gt10_relief_gt500_pct']
        N['gB_dev10_relief_med'] = _s['dev_gt10_relief_median_m']
        N['gB_land_relief_med'] = _s['land_relief_median_m']
        N['gB_area_mean'] = _s.get('B_area_mean')
        # 三个评估规模,正文逐档并列给出
        N['gB_fem_h_sites'] = _s['fem_h_sites']
        N['gB_fno_min_sites'] = _s['fno_min_sites']
        N['gB_fem_h_05'] = _s['fem_h_05']
        N['gB_fno_min_05'] = _s['fno_min_05']
        N['gB_fem_d_01'] = _s['fem_d_01']
        N['gB_fno_h_01'] = _s['fno_h_01']
        N['gB_dem_gb'] = _s['dem_gb']
        # 0.5° 场能不能当查表用 —— 开放研究节据此给出使用限制
        N['cellvar_n_cells'] = _s['cellvar_n_cells']
        N['cellvar_median'] = _s['cellvar_median']
        N['cellvar_frac_gt2pct'] = _s['cellvar_frac_gt2pct']
        N['cellvar_frac_gt10pct'] = _s['cellvar_frac_gt10pct']
        N['cellvar_ref_absdev'] = _s['cellvar_ref_absdev']
        for _k in ('lookup_n', 'lookup_err_median', 'lookup_sign_agree_pct', 'lookup_absdev_rho',
                   'cell_mixed_sign_pct'):
            if _k in _s:
                N[_k] = _s[_k]

    _gc = os.path.join(HERE, 'geometry_consistency_test.json')
    if os.path.exists(_gc):
        with open(_gc, encoding='utf-8') as _fh:
            _g2 = json.load(_fh)
        for _k in ('pred_med_ridge', 'pred_med_valley', 'fem_med_ridge',
                   'fem_med_valley', 'sign_agree_pct',
                   'sign_from_relh_fem_pct', 'sign_from_relh_fno_pct'):
            N['geom_' + _k] = _g2[_k]
        N['geom_corr_abs'] = abs(_g2['corr_dev_relh'])   # 正文写「r = −0.675」
    _tb = os.path.join(HERE, 'davies_terrain_baseline.json')
    if os.path.exists(_tb):
        with open(_tb, encoding='utf-8') as _fh:
            _t = json.load(_fh)
        N['tb_n'] = _t['n']
        N['tb_beta_B'] = _t['beta_B']
        N['tb_beta_h'] = abs(_t['beta_h'])          # 正文写「−0.051」,负号在捕获组外
        N['tb_partial_B'] = _t['partial_B']
        N['tb_partial_h'] = abs(_t['partial_h'])
        N['tb_partial_h_p'] = _t['partial_h_p']
        N['tb_corr_B_h'] = abs(_t['corr_B_h'])
    _ob = os.path.join(HERE, 'official_grid_benchmark.json')
    if os.path.exists(_ob):
        with open(_ob, encoding='utf-8') as _fh:
            _o = json.load(_fh)
        for _k in ('official_land_area_weighted', 'official_global_arithmetic',
                   'official_corrected', 'official_delta', 'official_delta_pct'):
            N['ghfdb_' + _k.replace('official_', '')] = _o[_k]
    else:
        print('  ⚠ 缺 official_grid_benchmark.json,官方场对照键跳过')
    sg = qc - qm
    se = sg.std(ddof=1) / np.sqrt(n)
    N['point_mean_shift'] = round(float(sg.mean()), 3)
    N['point_ci_lo'] = round(float(sg.mean() - 1.96 * se), 3)
    N['point_ci_hi'] = round(float(sg.mean() + 1.96 * se), 3)

    # 外部检验(读各自的 json,保持单一来源)
    for src, keys in [
        ('paired_null_fairness_test.json', None),
        ('expert_judgement_validation.json', ['n_not_significant', 'n_not_corrected',
                                              'median_ns', 'median_nc', 'frac_gt5_ns',
                                              'frac_gt5_nc', 'p', 'cliffs_delta']),
        ('random_land_baseline.json', ['n', 'frac_B_gt1']),
        ('mesh_resolution_test.json', None),
        ('footprint_fixed_resolution_test.json', None),
        ('vertical_mesh_test.json', ['equivalent_depth_m', 'ratio_nz10_to_finest']),
        ('analytic_benchmark.json', None),
        ('quality_stratified_test.json', None),
        ('truncation_bias_test.json', None),
        ('dropout_calibration_test.json', None),
    ]:
        pth = os.path.join(HERE, src)
        if not os.path.exists(pth):
            print(f'  ⚠ 缺 {src},相关键跳过')
            continue
        obj = J(src)
        tag = src.replace('.json', '')
        if keys is None:
            N[tag] = obj.get('summary', obj)
        else:
            N[tag] = {k2: obj[k2] for k2 in keys if k2 in obj}

    # 随机陆地参照的显著性
    if 'random_land_baseline' in N:
        p0 = N['random_land_baseline']['frac_B_gt1'] / 100
        n0 = N['random_land_baseline']['n']
        p1 = N['frac_B_gt1_pct'] / 100
        se_ = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n)
        z = (p1 - p0) / se_
        N['valley_excess_pp'] = round(100 * (p1 - p0), 2)
        N['valley_p'] = float(f'{2 * stats.norm.sf(abs(z)):.3g}')

    N['b_min'] = round(float(B.min()), 3)
    N['b_max'] = round(float(B.max()), 3)


    # 这是从 b_min 一步就能算出的推导值,却因为无人管而错了。

    N['b_min_flux_pct'] = round(100 * (1.0 / float(B.min()) - 1), 1)

    N['b_max_flux_pct'] = round(abs(100 * (1.0 / float(B.max()) - 1)), 1)  # 正文写「−31%」,负号在捕获组外
    N['b_mean'] = round(float(B.mean()), 3)
    N['b_revision_up_pct'] = int(round(100 * (1.0 / float(B.min()) - 1)))
    N['b_revision_down_pct'] = int(round(100 * (1 - 1.0 / float(B.max()))))
    N['n_b_gt_1p3'] = int((B > 1.3).sum())
    N['n_b_lt_0p6'] = int((B < 0.6).sum())
    N['frac_b_extreme_pct'] = round(float(100 * ((B < 0.6) | (B > 1.3)).mean()), 2)
    N['frac_nearflat_pct'] = round(float(100 * (ab < 0.02).mean()), 1)
    N['nearflat_median_pct'] = round(float(np.median(rel[ab < 0.02])), 2)

    mp = os.path.join(HERE, 'mc_dropout_uncertainty.npz')
    if os.path.exists(mp):
        mc = np.load(mp)
        Bo, Bm, Bsd = mc['B_original'], mc['B_mc_mean'], mc['B_mc_std']
        _rp = np.load(os.path.join(HERE, 'rebuild_points.npz'))
        assert np.allclose(_rp['lon'], mc['lon']) and np.allclose(_rp['lat'], mc['lat']),             'mc_dropout 与 rebuild_points 的测点顺序不一致,不能按位置共用分档'
        mab = np.abs(_rp['B'] - 1.0)
        N['mc_r_mean_vs_single'] = round(float(np.corrcoef(Bm, Bo)[0, 1]), 3)
        N['mc_corr_std_absB1'] = round(float(np.corrcoef(Bsd, mab)[0, 1]), 2)
        for lo, hi, k in [(0, .02, 'flat'), (.02, .10, 'mod'), (.10, 9, 'high')]:
            s = Bsd[(mab >= lo) & (mab < hi)]
            N[f'mc_n_{k}'] = int(s.size)
            N[f'mc_std_mean_{k}'] = round(float(s.mean()), 4)
            N[f'mc_std_median_{k}'] = round(float(np.median(s)), 4)
        N['mc_std_growth'] = round(float(N['mc_std_mean_high'] / N['mc_std_mean_flat']), 1)
    else:
        print('  ⚠ 缺 mc_dropout_uncertainty.npz,MC dropout 键跳过')

    vp2 = os.path.join(HERE, 'validation_independent_500.npz')
    if os.path.exists(vp2):
        v = np.load(vp2)
        Bf, Bn, rl = v['B_fem'], v['B_fno'], v['reliefs']
        w = float(stats.wasserstein_distance(Bf.ravel(), Bn.ravel()))
        # 5 位小数:该量在 1e-3 量级,4 位精度不足以表达(0.00085 会被截成 0.0008)
        N['indep_wasserstein'] = round(w, 5)
        N['indep_wasserstein_ratio'] = int(round(float(Bf.ravel().std()) / w))
        for lo, hi, k in [(0, 50, 'lt50m'), (500, 1e9, 'gt500m')]:
            m2 = (rl >= lo) & (rl < hi)
            a2, b2 = Bf[m2].ravel(), Bn[m2].ravel()
            N[f'indep_r2_{k}'] = round(
                float(1 - ((a2 - b2) ** 2).sum() / ((a2 - a2.mean()) ** 2).sum()), 2)
            N[f'indep_mae_{k}'] = round(float(np.abs(a2 - b2).mean()), 3)

    if os.path.exists(os.path.join(HERE, 'fno3d_large_metrics.json')):
        m = J('fno3d_large_metrics.json')
        N['n_tiles_total'] = int(m['n_total'])
        N['n_tiles_train'] = int(m['n_train'])
        N['n_tiles_test'] = int(m['n_test'])
        N['n_params'] = int(m['n_params'])
        N['r2_spatial_split'] = round(float(m['r2_test']), 4)
        N['train_time_h'] = round(float(m['train_time_s']) / 3600.0, 1)
    else:
        print('  ⚠ 缺 fno3d_large_metrics.json,模型指标键跳过')

    vp = os.path.join(HERE, 'validation_independent_500.npz')
    if os.path.exists(vp):
        N['r2_independent'] = round(float(np.load(vp)['r2_total']), 4)
    else:
        print('  ⚠ 缺 validation_independent_500.npz,独立集 R² 跳过')

    # B 场集合统计量(训练标签全体输出像元) —— 正文"该场几乎处处接近1"一句的来源
    gp = os.path.join(HERE, 'global_fem_dataset.npz')
    if os.path.exists(gp):
        Y = np.load(gp)['Y'].ravel()
        N['b_ens_n_pixels'] = int(Y.size)
        N['b_ens_mean'] = round(float(Y.mean()), 3)
        N['b_ens_std'] = round(float(Y.std()), 3)
        N['b_ens_min'] = round(float(Y.min()), 2)
        N['b_ens_max'] = round(float(Y.max()), 2)
        lo, hi = np.percentile(Y, [5, 95])
        N['b_ens_p5'] = round(float(lo), 2)
        N['b_ens_p95'] = round(float(hi), 2)
    else:
        print('  ⚠ 缺 global_fem_dataset.npz,B 场集合统计量跳过')

    vcp = os.path.join(HERE, 'valley_clustering_test.json')
    if os.path.exists(vcp):
        vc = json.load(open(vcp, encoding='utf-8'))
        ps = [r['p'] for r in vc['blocks']]
        N['valley_p_block_min'] = float(f'{min(ps):.3g}')
        N['valley_p_block_max'] = float(f'{max(ps):.3g}')
        N['valley_baseline_se_pp'] = round(
            100 * (vc['baseline_pct'] / 100 * (1 - vc['baseline_pct'] / 100)
                   / vc['baseline_n']) ** 0.5, 1)
    else:
        print('  ⚠ 缺 valley_clustering_test.json,空间聚集 p 值跳过')

    pnp = os.path.join(HERE, 'paired_null_fairness_test.json')
    if os.path.exists(pnp):
        pn = json.load(open(pnp, encoding='utf-8'))
        allb = pn.get('全部')
        if allb:
            N['pair_n_total'] = int(allb['n'])
            N['pair_real_pct'] = round(float(allb['real']), 2)
            N['pair_margin_b'] = round(float(allb['marginB']), 2)
            N['pair_margin_a'] = round(float(allb['marginA']), 2)

            N['pair_null_b_pct'] = round(float(allb['nullB']), 2)
            N['pair_null_b_rise_pct'] = abs(N['pair_null_b_pct'])
            N['pair_null_a_pct'] = round(float(allb['nullA']), 2)
        for k, tag in (('0.10-0.20', 'hi1'), ('> 0.20', 'hi2')):
            if k in pn:
                N[f'pair_n_{tag}'] = int(pn[k]['n'])
    else:
        print('  ⚠ 缺 paired_null_fairness_test.json,配对检验键跳过')

    for src, tag in (('footprint_fixed_resolution_test.json', 'fp'),
                     ('mesh_convergence_large.json', 'mesh')):
        pth = os.path.join(HERE, src)
        if not os.path.exists(pth):
            print(f'  ⚠ 缺 {src},收敛汇总键跳过')
            continue
        obj = json.load(open(pth, encoding='utf-8'))
        summ = obj.get('summary', {})
        for band, v in summ.items():
            if not isinstance(v, dict):
                continue
            key = band.replace('.', '').replace('-', '_').replace(' ', '').replace('>', 'gt').replace('<', 'lt')
            # 两个脚本的字段名不同:窗口用 pct_change,网格用 pct_change_40_to_128
            for fld in ('pct_change', 'pct_change_40_to_128'):
                if fld in v:
                    N[f'{tag}_{key}_pct'] = round(float(v[fld]), 1)
                    break

    mpth = os.path.join(HERE, 'mesh_convergence_large.json')
    if os.path.exists(mpth):
        mm = json.load(open(mpth, encoding='utf-8'))
        nxs = mm['nx_ladder']
        for band, s in mm['summary'].items():
            if not isinstance(s, dict) or 'median_absB1' not in s:
                continue
            key = band.replace('.', '').replace('-', '_')
            md = s['median_absB1']
            for a, b in zip(nxs[:-1], nxs[1:]):
                va, vb = md[str(a)], md[str(b)]
                N['mesh_%s_step_%d_%d_pct' % (key, a, b)] = round(100 * (vb - va) / va, 1)
        if 'mesh_00_002_pct' in N:                       # 正文写「下降 38.8%」
            N['mesh_00_002_drop_pct'] = abs(N['mesh_00_002_pct'])
        # 「B 本身仅变 x%」:|B−1| ≥ 0.02 的测点上逐点 |B_40 − B_128| / B_128 的中位数
        # (近平坦档校正量处于噪声区,正文 S1.2 的这句话讲的是有实际校正量的测点)
        _a, _b = str(nxs[0]), str(nxs[-1])
        # 扩大试验的逐档偏差(逐点变化中位 ÷ 档内中位,与误差模型同一口径)、置信区间、离散、变号数
        _lab = {'<0.02': 'lt002', '0.02-0.05': '002_005', '0.05-0.10': '005_010', '0.10-0.20': '010_020', '>0.20': 'gt020'}
        for _bk, _v in mm['bands'].items():
            _t = _lab[_bk]
            N[f'mc_mesh_bias_{_t}'] = round(_v['bias_40_to_128_pct'], 1)
            N[f'mc_mesh_ci_lo_{_t}'] = round(_v['bias_ci'][0], 1)
            N[f'mc_mesh_ci_hi_{_t}'] = round(_v['bias_ci'][1], 1)
            N[f'mc_mesh_scatter_{_t}'] = round(_v['scatter_40_to_128_pct'], 1)
            N[f'mc_mesh_step96_{_t}'] = round(_v['step_96_to_128_pct'], 1)
            N[f'mc_mesh_aa_bias_{_t}'] = round(_v['bias_40aa_to_128_pct'], 1)
            N[f'mc_mesh_flip_{_t}'] = int(_v['sign_flip_40'])
        N['mc_mesh_n_sites'] = len(mm['rows'])
        N['mc_mesh_flip_total'] = int(sum(v['sign_flip_40'] for v in mm['bands'].values()))
        N['mc_mesh_step96_max'] = round(max(abs(v['step_96_to_128_pct']) for v in mm['bands'].values()), 1)
        _hi = [r for r in mm['rows'] if r['absB1'] >= 0.05]
        _ratio = np.array([abs(r['B']['128'] - 1) / abs(r['B']['40'] - 1) for r in _hi])
        N['mc_mesh_n_ge005'] = len(_hi)
        N['mc_mesh_ratio_med_ge005'] = round(float(np.median(_ratio)), 2)
        N['mc_mesh_frac_lt1_ge005_pct'] = round(100 * float(np.mean(_ratio < 1)), 0)
        N['mc_mesh_t96_s'] = round(mm['timing_s']['96']['median'], 1)
        N['mc_mesh_t40_s'] = round(mm['timing_s']['40']['median'], 1)
        N['mesh_B_self_pct'] = round(100 * float(np.median(
            [abs(r['B'][_b] - r['B'][_a]) / r['B'][_b]
             for r in mm['rows'] if r['absB1'] >= 0.02])), 1)


    rpth = os.path.join(HERE, 'global_relief_05deg.npz')
    if os.path.exists(rpth):
        import cartopy.feature as _cf
        from rasterio.features import rasterize as _rz
        from rasterio.transform import from_origin as _fo
        _rf = np.load(rpth)
        _rel = _rf['relief']
        _lnd = _rz([(g, 1) for g in _cf.LAND.geometries()], out_shape=_rel.shape,
                   transform=_fo(-180, 90, 0.5, 0.5), fill=0,
                   all_touched=True).astype(bool)[::-1]
        # 口径必须与图 8b 一致:那里画的是"陆地 ∩ 克里金有覆盖"的格点,
        # 只按纯陆地统计会得到 216 m,与图上那张场的中位(305 m)对不上。
        _kg = np.load(os.path.join(HERE, 'kriged_heatflow_05deg_global.npz'))
        _v = _rel[_lnd & np.isfinite(_rel) & np.isfinite(_kg['krige_qm'])]
        N['relief_land_median_m'] = int(round(float(np.median(_v))))
        N['relief_land_p95_m'] = int(round(float(np.percentile(_v, 95))))
        N['relief_land_max_m'] = int(round(float(_v.max())))
        print('  relief_land_median_m           %d' % N['relief_land_median_m'])

    vpth = os.path.join(HERE, 'vertical_mesh_test.json')
    if os.path.exists(vpth):
        vm = json.load(open(vpth, encoding='utf-8'))
        nzs = vm['nz_ladder']
        s = vm['summary']
        N['vert_nz_lo'] = int(nzs[0])
        N['vert_nz_hi'] = int(nzs[-1])
        N['vert_med_lo'] = round(s[str(nzs[0])]['median_absB1'], 3)
        N['vert_med_hi'] = round(s[str(nzs[-1])]['median_absB1'], 3)
        N['vert_dz_lo'] = round(s[str(nzs[0])]['median_dz_top'])
        N['vert_dz_hi'] = round(s[str(nzs[-1])]['median_dz_top'])
        # 生产配置相对最细解的低估幅度(正文引用的那个数,写成正值)
        N['vert_underest_pct'] = round(abs(s[str(nzs[0])]['rel_to_finest']), 1)
        # 逐档增量:相对前一档
        for a, b in zip(nzs[:-1], nzs[1:]):
            ma, mb = s[str(a)]['median_absB1'], s[str(b)]['median_absB1']
            N['vert_step_%d_%d_pct' % (a, b)] = round(100 * (mb - ma) / ma, 1)
        # 「B 本身仅变 x%」:逐测点 |B_粗 − B_细| / B_细 的中位数(正文 S1.2)
        _lo, _hi = str(nzs[0]), str(nzs[-1])
        N['vert_B_self_pct'] = round(100 * float(np.median(
            [abs(r['B'][_hi] - r['B'][_lo]) / r['B'][_hi] for r in vm['rows']])), 1)
    else:
        print('  ⚠ 缺 vertical_mesh_test.json,垂向收敛键跳过')


    npth = os.path.join(HERE, 'footprint_narrow_test.json')
    wpth = os.path.join(HERE, 'footprint_fixed_resolution_test.json')
    if os.path.exists(npth) and os.path.exists(wpth):
        nb = json.load(open(npth, encoding='utf-8'))
        wb = json.load(open(wpth, encoding='utf-8'))
        for band in nb['summary']:
            if band not in wb['summary']:
                continue
            mn = nb['summary'][band]['median_absB1']
            mw = wb['summary'][band]['median_absB1']
            # 57 km 是两组共有的锚点,同一批测点同一配置,数值必须逐档相等;
            # 不等说明两次跑的不是同一批点,拼接出来的跨度就没有意义。
            assert abs(mn['57'] - mw['57']) < 1e-9, (band, mn['57'], mw['57'])
            lo, hi = mn['20'], mw['160']
            key = band.replace('.', '').replace('-', '_')
            N[f'fp_full_{key}_pct'] = round(100 * (hi - lo) / lo, 1)
            N[f'fp_full_{key}_lo'] = round(lo, 4)
            N[f'fp_full_{key}_hi'] = round(hi, 4)
    else:
        print('  ⚠ 缺窄窗口或宽窗口产物,fp_full_* 键跳过')


    pap = os.path.join(HERE, 'province_artifact_test.json')
    if os.path.exists(pap):
        pa = json.load(open(pap, encoding='utf-8'))
        N['cells_n'] = pa['n_cells']
        N['cells_median_change_pct'] = pa['cell_median_change_pct']
        N['cells_p5'] = pa['cell_p5']
        N['cells_p95'] = pa['cell_p95']
        for r in pa['provinces']:
            key = r['province'].split()[0].lower().strip('(')
            N[f'prov_{key}_change'] = r['change_pct']
            N[f'prov_{key}_change_abs'] = abs(r['change_pct'])
            N[f'prov_{key}_valley'] = r['valley_frac_pct']
            N[f'prov_{key}_raw'] = r['mean_measured']
            N[f'prov_{key}_cor'] = r['mean_corrected']
            N[f'prov_{key}_n'] = r['n']
            N[f'prov_{key}_ci_lo'] = abs(r['ci_lo'])
            N[f'prov_{key}_ci_hi'] = abs(r['ci_hi'])
            # 稳态传导下热厚度与地表热流成反比,故热流下调 x% 对应厚度上调
            # q_raw/q_cor − 1。4.1 用它把「区域均值变了」推到「由它导出的量跟着变」。
            if r['mean_corrected']:
                N[f'prov_{key}_thick_pct'] = int(round(
                    100 * (r['mean_measured'] / r['mean_corrected'] - 1)))
                N[f'prov_{key}_dq'] = round(r['mean_measured'] - r['mean_corrected'], 1)
                N[f'prov_{key}_dT40'] = int(round((r['mean_measured'] - r['mean_corrected']) * 40 / 2.5))
        top4 = sorted(pa['provinces'], key=lambda r: -r['valley_frac_pct'])[:4]
        dq4 = [r['mean_measured'] - r['mean_corrected'] for r in top4]
        N['prov_top4_dq_min'] = round(min(dq4), 1)
        N['prov_top4_dq_max'] = round(max(dq4), 1)
        N['prov_top4_dT40_min'] = int(round(min(dq4) * 40 / 2.5))
        N['prov_top4_dT40_max'] = int(round(max(dq4) * 40 / 2.5))
        for src, dst in (('cell_valley_spearman', 'cell_valley_rho'),
                         ('cell_valley_slope_pct_per_pp', 'cell_valley_slope'),
                         ('cell_change_median_valley_lt55', 'cell_chg_lt55'),
                         ('cell_change_median_valley_gt75', 'cell_chg_gt75'),
                         ('n_cells', 'cell_n_2deg'),
                         ('cell_n_valley_lt55', 'cell_n_lt55'),
                         ('cell_n_valley_gt75', 'cell_n_gt75')):
            if src in pa:
                N[dst] = pa[src]
    else:
        print('  ⚠ 缺 province_artifact_test.json,构造省键跳过')

    dvp = os.path.join(HERE, 'davies_validation.json')
    if os.path.exists(dvp):
        dv = json.load(open(dvp, encoding='utf-8'))
        for key, tag in (('>= 0.05', 'b05'), ('>= 0.10', 'b10')):
            if key in dv:
                N[f'davies_slope_{tag}'] = dv[key]['slope']
                N[f'davies_ci_lo_{tag}'] = dv[key]['ci_lo']
                N[f'davies_ci_hi_{tag}'] = dv[key]['ci_hi']
                N[f'davies_n_{tag}'] = dv[key]['n']
        if '全部' in dv:
            N['davies_slope_all'] = dv['全部']['slope']
        q = dv.get('quintiles_ge005')
        if q:
            N['davies_quint_first_pct'] = q[0]['resid_median_pct']
            N['davies_quint_last_pct'] = q[-1]['resid_median_pct']
            N['davies_quint_mid_min_pct'] = min(x['resid_median_pct'] for x in q[1:4])
            N['davies_quint_mid_max_pct'] = max(x['resid_median_pct'] for x in q[1:4])
        for k in ('resid_sd_pct', 'relB_sd_pct'):
            if k in dv:
                N['davies_' + k] = dv[k]
        if 'spearman_ge010' in dv:
            N['davies_rho_b10'] = dv['spearman_ge010']['rho']
    else:
        print('  ⚠ 缺 davies_validation.json,外部验证键跳过')

    cop = os.path.join(HERE, 'c15_ordinal_test.json')
    if os.path.exists(cop):
        co = json.load(open(cop, encoding='utf-8'))
        tagmap = {'[Present not significant]': 'ns',
                  '[Present and not corrected]': 'nc',
                  '[Present and corrected]': 'pc'}
        for g in co['groups']:
            t = tagmap.get(g['category'])
            if t:
                N[f'c15_n_{t}'] = g['n']
                N[f'c15_med_{t}'] = round(g['median_absB1'], 4)
                N[f'c15_gt5_{t}'] = round(g['frac_gt5pct'], 1)
        if 'jonckheere_z' in co:
            N['c15_jt_z'] = round(co['jonckheere_z'], 2)
        if 'step1_mw_p' in co:
            N['c15_step1_p'] = float('%.2g' % co['step1_mw_p'])
            N['c15_n_conflict'] = co['n_conflict_sites']
        for a in co.get('adjacent_deltas', []):
            if a['frm'] == '存在但不显著':
                N['c15_delta_step1'] = round(a['cliffs_delta'], 2)
            elif a['frm'] == '识别未校正':
                N['c15_delta_step2'] = round(a['cliffs_delta'], 2)
    else:
        print('  ⚠ 缺 c15_ordinal_test.json,C15 有序检验键跳过')

    vcp = os.path.join(HERE, 'valley_clustering_test.json')
    if os.path.exists(vcp):
        vc = json.load(open(vcp, encoding='utf-8'))
        ps = [b['p'] for b in vc['blocks']]
        N['valley_p_block_min'] = float(f'{min(ps):.3g}')
        N['valley_p_block_max'] = float(f'{max(ps):.3g}')
        N['valley_baseline_se_pp'] = round(
            100 * (vc['baseline_pct'] / 100 * (1 - vc['baseline_pct'] / 100)
                   / vc['baseline_n']) ** 0.5, 1)
    else:
        print('  ⚠ 缺 valley_clustering_test.json,块自助键跳过')

    ebp = os.path.join(HERE, 'error_budget.json')
    if os.path.exists(ebp):
        eb = json.load(open(ebp, encoding='utf-8'))
        for cls, tag in (('<0.02', 'b0'), ('0.02-0.05', 'b1'), ('0.05-0.10', 'b2'),
                         ('0.10-0.20', 'b3'), ('>0.20', 'b4')):
            row = eb[cls]
            N[f'eb_n_{tag}'] = int(row['n'])
            N[f'eb_corr_{tag}'] = round(float(row['correction_mW']), 2)
            N[f'eb_sigma_{tag}'] = round(float(row['sigma_mW']), 2)
            N[f'eb_bias_{tag}'] = round(float(row['bias_mW']), 2)
            N[f'eb_snr_{tag}'] = round(float(row['snr_median']), 2)
            N[f'eb_sigma_pct_{tag}'] = int(round(row['sigma_over_corr_pct']))
            N[f'eb_bias_pct_{tag}'] = int(round(row['bias_over_corr_pct']))
            for _c in ('op', 'mesh', 'win', 'dep'):
                N[f'eb_{_c}_{tag}'] = round(float(row[f's_{_c}_mW']), 2)
        T = eb['terms']
        N['fe_bias_pct'] = round(100 * T['fe_bias_rel'], 1)
        N['dep_rel_pct'] = int(round(100 * T['dep_rel']))
        for cls, tag in (('0.05-0.10', 'b2'), ('0.10-0.20', 'b3'), ('>0.20', 'b4')):
            N[f'mesh_bias_pct_{tag}'] = round(100 * T['mesh'][cls]['bias_rel'], 1)
            N[f'win_bias_pct_{tag}'] = round(100 * T['win'][cls]['bias_rel'], 1)
        for cls, tag in (('<0.02', 'b0'), ('0.02-0.05', 'b1'), ('0.05-0.10', 'b2'), ('>=0.10', 'b3')):
            N[f'op_c_rmse_{tag}'] = round(T['op'][cls]['rmse'], 4)
            N[f'op_c_n_{tag}'] = T['op'][cls]['n']
    else:
        print('  ⚠ 缺 error_budget.json,误差预算键跳过')

    # 4 折交叉验证各折的训练集规模 —— 供 check_numbers.py 判定是否与主模型同代
    # 读取 fno3d_foldB_metrics.json、fno3d_foldC_metrics.json、
    # fno3d_foldD_metrics.json(名字下面用 f-string 拼,写在这里是为了让
    # audit_narrative_binding.py 的整名匹配能认出它们)
    folds = {}
    for f in ('B', 'C', 'D'):
        fp = os.path.join(HERE, f'fno3d_fold{f}_metrics.json')
        if os.path.exists(fp):
            fm = json.load(open(fp, encoding='utf-8'))
            folds[f] = {k2: fm[k2] for k2 in ('n_total', 'n_train', 'n_test', 'r2_test')
                        if k2 in fm}
    if folds:
        N['cv_folds'] = folds


    # 4 折汇总 —— 附表 S3 的逐折值与正文的均值/标准差/范围
    if folds and 'r2_spatial_split' in N:
        _r2 = [N['r2_spatial_split']] + [v['r2_test'] for v in folds.values()
                                         if 'r2_test' in v]
        if len(_r2) == 4:
            import statistics as _st
            N['cv_r2_mean'] = round(_st.mean(_r2), 3)
            N['cv_r2_std'] = round(_st.stdev(_r2), 3)
            N['cv_r2_min'] = round(min(_r2), 3)
            N['cv_r2_max'] = round(max(_r2), 3)
        for _k, _v in folds.items():
            if 'r2_test' in _v:
                N['cv_r2_' + _k] = round(_v['r2_test'], 4)
                N['cv_ntrain_' + _k] = int(_v['n_train'])
                N['cv_ntest_' + _k] = int(_v['n_test'])

    import re as _re_md
    with open(os.path.join(HERE, 'rebuild_correction.py'), encoding='utf-8') as _fh:
        N['fallback_depth_m'] = float(
            _re_md.search(r'MEDIAN_DEPTH_M = ([\d.]+)', _fh.read()).group(1))
    _fp = os.path.join(HERE, 'depth_fallback_sensitivity.json')
    if os.path.exists(_fp):
        _dfs = json.load(open(_fp, encoding='utf-8'))
        for _lab, _tag in (('130 m (Q1,浅)', 'lo'),
                           ('360 m (中位,本文采用)', 'mid'),
                           ('1,250 m (Q3,深)', 'hi')):
            if _lab in _dfs:
                N['df_%s_depth' % _tag] = int(_dfs[_lab]['depth'])
                N['df_%s_median' % _tag] = round(_dfs[_lab]['median'], 2)
                N['df_%s_moved' % _tag] = round(_dfs[_lab]['moved'], 1)

    # 测点筛选链(图 1a 与引言的 54,552 条陆地记录)
    _cp = os.path.join(HERE, 'filter_chain.json')
    if os.path.exists(_cp):
        for _row in json.load(open(_cp, encoding='utf-8')):
            _key = _row['label'].lower().replace(' ', '_').replace('-', '_')
            N['fc_' + _key] = int(_row['n'])

    _pf = os.path.join(HERE, 'depth_decay_profiles.json')
    if os.path.exists(_pf):
        _prof = json.load(open(_pf, encoding='utf-8'))
        for _D in ('250', '390', '500', '1000'):
            _v = [r['f'][_D] for r in _prof if _D in r.get('f', {})]
            if _v:
                N['fD_' + _D] = round(float(np.median(_v)), 3)

    _vp = os.path.join(HERE, 'validation_independent_500.npz')
    if os.path.exists(_vp):
        _v = np.load(_vp)
        _Bf, _Bn, _rl = _v['B_fem'], _v['B_fno'], _v['reliefs']
        if _Bf.ndim == 4:
            _Bf, _Bn = _Bf[:, 0], _Bn[:, 0]
        for _lo, _hi, _tag in [(0, 100, 'flat'), (100, 500, 'mid'), (500, 9e9, 'high')]:
            _m = (_rl >= _lo) & (_rl < _hi)
            if _m.any():
                N['op_rmse_' + _tag] = round(
                    float(np.sqrt(((_Bf[_m] - _Bn[_m]) ** 2).mean())), 4)

    _rp = os.path.join(HERE, 'rebuild_points.npz')
    _srp = os.path.join(HERE, 'site_reliability.npz')
    if os.path.exists(_rp) and os.path.exists(_srp):
        _d = np.load(_rp)
        _i = int(np.argmin((_d['lon'] - (-69.93)) ** 2 + (_d['lat'] - (-32.68)) ** 2))
        _qm = float(_d['q_measured'][_i]); _qc = float(_d['q_corrected'][_i])
        _B, _Bs = float(_d['B'][_i]), float(_d['B_surface'][_i])
        _k = (_B - 1) / (_Bs - 1)
        _s = np.load(_srp)
        _sq = float(_s['sigma_q'][_i])
        N['andes_q_measured'] = round(_qm, 0)
        N['andes_q_corrected'] = round(_qc, 1)
        N['andes_B'] = round(_B, 3)
        N['andes_B_surface'] = round(_Bs, 3)
        N['andes_k'] = round(_k, 2)
        N['andes_correction_mW'] = round(_qc - _qm, 1)
        N['andes_up_pct'] = int(round(100 * (_qc / _qm - 1)))
        N['andes_sigma_q'] = round(_sq, 1)
        N['andes_q_lo2'] = int(round(_qc - 2 * _sq))
        N['andes_q_hi2'] = int(round(_qc + 2 * _sq))
        N['andes_fallback'] = bool(_s['depth_fallback'][_i])
        _rel = float(np.load(os.path.join(HERE, 'site_terrain.npz'))['relief_m'][_i])
        _mr = [r for r in json.load(open(os.path.join(HERE, 'mesh_resolution_test.json'), encoding='utf-8'))['rows']
               if abs(r['relief'] - _rel) < 2.0]
        if _mr:
            for _nx in ('40', '128'):
                _BD = 1 + _k * (_mr[0]['B'][_nx] - 1)
                N[f'andes_fem{_nx}_Bs'] = round(_mr[0]['B'][_nx], 3)
                N[f'andes_fem{_nx}_q'] = round(_qm / _BD, 1)

    _srn = os.path.join(HERE, 'site_reliability.npz')
    if os.path.exists(_srn):
        _s = np.load(_srn, allow_pickle=True)
        _o = _s['outside_training_relief'].astype(bool)
        N['out_relief_n'] = int(_o.sum())
        N['out_relief_med_pct'] = round(100 * float(np.median(np.abs(_s['q_corrected'][_o] / _s['q_measured'][_o] - 1))), 2)
        N['out_relief_C_n'] = int((_s['grade'][_o] == 'C').sum())

    if 'andes_B' in N:
        N['andes_dev_s'] = round(abs(1 - N['andes_B_surface']), 3)
        N['andes_dev_D'] = round(abs(1 - N['andes_B']), 3)
    _rp2 = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    N['site_mean_qobs'] = round(float(_rp2['q_measured'].mean()), 1)
    _kk = np.abs((_rp2['B'] - 1) / np.where(np.abs(_rp2['B_surface'] - 1) > 1e-6, _rp2['B_surface'] - 1, 1))
    N['frac_k_gt1_pct'] = round(100 * float(np.mean(_kk > 1)), 1)
    _c = 100 * (_rp2['q_corrected'] / _rp2['q_measured'] - 1)
    N['point_corr_lt_m30_n'] = int((_c < -30).sum())
    N['point_corr_beyond30_pct'] = round(100 * float(np.mean(np.abs(_c) > 30)), 2)
    if 'andes_k' in N:
        N['andes_k_pct'] = int(round(100 * (N['andes_k'] - 1)))
    _vi = np.load(os.path.join(HERE, 'validation_independent_500.npz'))
    _rel = _vi['reliefs']; _bf = _vi['B_fem']; _bp = _vi['B_fno']
    _lo, _hi = _rel < 100, _rel > 500
    N['indep_mae_lt100m'] = round(float(np.abs(_bp[_lo] - _bf[_lo]).mean()), 4)
    N['indep_mae_gt500m_px'] = round(float(np.abs(_bp[_hi] - _bf[_hi]).mean()), 4)
    N['indep_var_ratio_500_100'] = int(round(float(_bf[_hi].var() / _bf[_lo].var()), -1))
    _l50 = _rel < 50
    N['indep_mae_lt50m_px'] = round(float(np.abs(_bp[_l50] - _bf[_l50]).mean()), 4)
    N['depthavg_median_pct'] = N.get('median_correction_pct')          # S2.1 用的三位小数口径(与正文 0.7% 分开断言)

    _tb = os.path.join(HERE, 'truncation_bias_test.json')
    if os.path.exists(_tb):
        _t = json.load(open(_tb, encoding='utf-8'))
        N['tb_iqr_out_n'] = _t['因 3xIQR 剔除']['n']
        N['tb_iqr_out_med'] = round(_t['因 3xIQR 剔除']['median_absB1'], 3)
        N['tb_kept_med'] = round(_t['最终保留']['median_absB1'], 4)
        N['tb_n_differ'] = _t['iqr']['n_differ']
        _rep = _t['representative']
        for _k, _tag in (('iqr_uncorrected_kept', 'unc'), ('iqr_corrected_kept', 'cor'), ('no_iqr', 'none')):
            N[f'tb_{_tag}_median'] = round(_rep[_k]['median'], 3)
            N[f'tb_{_tag}_p99'] = round(_rep[_k]['p99'], 2)

    # 逐测点算子不确定度的校准(补充材料 S2.3)
    _ct = os.path.join(HERE, 'mc_d4_calibration_tiles.npz')
    if os.path.exists(_ct):
        from scipy import stats as _st
        _c = np.load(_ct)
        _e = np.concatenate([_c['e_0'], _c['e_1']]); _sd = np.concatenate([_c['sd_0'], _c['sd_1']])
        _P = np.concatenate([_c['P_0'], _c['P_1']]); _ab = np.abs(_P - 1)
        _c95 = float(np.quantile(_e / _sd, 0.954) / 2); _c68 = float(np.quantile(_e / _sd, 0.683))
        N['opsite_c95'] = round(_c95, 2); N['opsite_c68'] = round(_c68, 2)
        _cov1, _cov2 = [], []
        for _lo, _hi in ((0, .02), (.02, .05), (.05, .10), (.10, 9)):
            _m = (_ab >= _lo) & (_ab < _hi)
            _cov1.append(100 * np.mean(_e[_m] <= _c95 * _sd[_m])); _cov2.append(100 * np.mean(_e[_m] <= 2 * _c95 * _sd[_m]))
        N['opsite_cov1_min'] = int(round(min(_cov1))); N['opsite_cov1_max'] = int(round(max(_cov1)))
        N['opsite_cov2_min'] = int(round(min(_cov2))); N['opsite_cov2_max'] = int(round(max(_cov2)))
        _cj = json.load(open(os.path.join(HERE, 'mc_d4_calibration.json'), encoding='utf-8'))
        N['opsite_rho_indep'] = round(_cj['原独立集']['combined']['spearman'], 2)
        N['opsite_rho_excl'] = round(_cj['排除距离集']['combined']['spearman'], 2)
        N['opsite_ratio_indep'] = round(_cj['原独立集']['combined']['ratio_rmse_to_rms_sd'], 2)
        N['opsite_ratio_excl'] = round(_cj['排除距离集']['combined']['ratio_rmse_to_rms_sd'], 2)
        _u = np.load(os.path.join(HERE, 'mc_d4_uncertainty.npz'))
        N['opsite_r_orient_dropout'] = round(float(np.corrcoef(_u['orient_sd'], _u['dropout_sd'])[0, 1]), 2)
        _rp = np.load(os.path.join(HERE, 'rebuild_points.npz'))
        N['opsite_rho_absB'] = round(float(_st.spearmanr(_u['combined_sd'], np.abs(_rp['B_surface'] - 1)).correlation), 2)

    _pr = os.path.join(HERE, 'paired_null_robust_test.json')
    if os.path.exists(_pr):
        _r = json.load(open(_pr, encoding='utf-8'))
        _t = {'< 0.02': 'b0', '0.02-0.05': 'b1', '0.05-0.10': 'b2', '0.10-0.20': 'b3', '> 0.20': 'b4', '全部': 'all'}
        for _b, _k in _t.items():
            N[f'prr_margin_{_k}'] = round(_r[_b]['margin'], 2)
            N[f'prr_p_{_k}'] = round(_r[_b]['p'], 4)
        N['prr_n_sig'] = int(sum(_r[_b]['p'] < 0.05 for _b in list(_t)[:5]))

    _vt = os.path.join(HERE, 'valley_trend_test.json')
    if os.path.exists(_vt):
        _v = json.load(open(_vt, encoding='utf-8'))
        N['vt_slope'] = round(_v['slope_pp_per_decade'], 1)
        N['vt_slope_lo'] = round(_v['slope_ci'][0], 1)
        N['vt_slope_hi'] = round(_v['slope_ci'][1], 1)
        N['vt_p'] = round(_v['p_one_sided'], 3)
        N['vt_1000'] = round(_v['excess_at_1000m_pp'], 1)
        N['vt_1000_lo'] = round(_v['excess_at_1000m_ci'][0], 1)
        N['vt_1000_hi'] = round(_v['excess_at_1000m_ci'][1], 1)

    _tc = os.path.join(HERE, 'terrain_control_stats.json')
    if os.path.exists(_tc):
        _t = json.load(open(_tc, encoding='utf-8'))
        _rb = _t['relief_bins']
        N['tc_med_lowest'] = round(_rb[0]['med'], 2)
        N['tc_med_highest'] = round(_rb[-1]['med'], 1)
        N['tc_gt5_lowest'] = round(_rb[0]['gt5_pct'], 1)
        N['tc_gt5_highest'] = int(round(_rb[-1]['gt5_pct']))
        N['tc_rho_relief'] = round(_t['spearman'], 2)
        N['tc_exp'] = round(_t['power_exponent_binned'], 2)
        N['tc_r2_binned'] = round(_t['r2_binned'], 2)
        N['tc_r2_site'] = round(_t['r2_site'], 2)
        N['tc_r2_ridge'] = round(_t['r2_site_ridge'], 2)
        if 'power_median_1000m_pct' in _t:
            N['tc_med_1000m'] = round(_t['power_median_1000m_pct'], 1)
            N['tc_fold10'] = round(_t['power_fold_per_decade'], 1)
            N['tc_fit_lo_m'] = int(round(_t['power_relief_range_m'][0], -1))
            N['tc_fit_hi_m'] = int(round(_t['power_relief_range_m'][1], -2))
            N['tc_top_med'] = round(_t['power_top_bin_median_pct'], 1)
            N['tc_top_fit'] = round(_t['power_top_bin_fit_pct'], 1)
        if 'calibration' in _t:
            _cv = _t['calibration']['coverage']
            for _n, _nm in (('0', 'ind'), ('1', 'excl')):
                N['cal_cov1_' + _nm] = int(round(_cv[_n]['1']))
                N['cal_cov2_' + _nm] = int(round(_cv[_n]['2']))
                N['cal_cov2_1d_' + _nm] = round(_cv[_n]['2'], 1)   # 交叉检验,补充材料 S2.3 引一位小数
        # 图 6f:逐瓦片预测误差与实际误差的秩相关
            if 'spearman' in _t['calibration']:
                N['cal_rho_ind'] = round(_t['calibration']['spearman']['0'], 2)
                N['cal_rho_excl'] = round(_t['calibration']['spearman']['1'], 2)
        N['tc_r2_valley'] = round(_t['r2_site_valley'], 2)

    _sp = os.path.join(HERE, 'depth_decay_spectral.json')
    if os.path.exists(_sp):
        _dd = json.load(open(_sp, encoding='utf-8'))
        for _k in ('lambda_c_km_min', 'lambda_c_km_max', 'efold_m_min', 'efold_m_max',
                   'domain_over_efold_median', 'n_beyond_benchmark',
                   'err_at_benchmark_end_pct', 'n_domain_lt15', 'frac_domain_lt15_pct',
                   'err_lt15_lo_pct', 'err_lt15_hi_pct',
                   'efold_m_median', 'spearman_r', 'spearman_p', 'n_efold_above_360',
                   'fe_D0_median_pct'):
            N['dds_' + _k] = _dd[_k]
        for _D, _v in _dd.get('band_spread', {}).items():
            N['dds_spread_' + _D] = _v

    _sb = os.path.join(HERE, 'analytic_benchmark.json')
    if os.path.exists(_sb):
        _bm = json.load(open(_sb, encoding='utf-8'))
        _cs = _bm['cases']
        for _k, _tag in (('nx40_nz10', 'nz10'), ('nx40_nz40', 'nz40'),
                         ('nx40_nz80', 'nz80'), ('nx80_nz40', 'nx80')):
            _v = [abs(c['fem'][_k]['rel_err_pct']) for c in _cs if _k in c['fem']]
            if _v:
                N['sb_%s_pct' % _tag] = round(float(np.median(_v)), 1)
        _a = [abs(c['fem']['nx40_nz40']['rel_err_pct']) for c in _cs]
        _b = [abs(c['fem']['nx80_nz40']['rel_err_pct']) for c in _cs]
        N['sb_nx_diff_pp'] = round(max(abs(x - y) for x, y in zip(_a, _b)), 2)
        _pr = [abs(c['production']['rel_err_pct']) for c in _cs]
        N['sb_prod_lo_pct'] = round(min(_pr), 0)
        N['sb_prod_hi_pct'] = round(max(_pr), 0)
        N['sb_n_amp'] = len(_cs)
        N['sb_lambda_km'] = round(_bm['lambda_m'] / 1000.0, 1)

    # 解析基准的波长扫描(方法 2.1)
    _ap = os.path.join(HERE, 'analytic_benchmark_wavelength.json')
    if os.path.exists(_ap):
        _ab = json.load(open(_ap, encoding='utf-8'))
        N['ab_equiv_depth_m'] = int(round(_ab['equiv_depth_m']))
        for _lam, _tag in ((28.5, '285'), (14.25, '1425'),
                           (9.5, '95'), (7.125, '7125')):
            _v = [r['err_vs_D0_pct'] for r in _ab['rows']
                  if abs(r['lambda_km'] - _lam) < 1e-6]
            if _v:
                N['ab_err_%s_lo' % _tag] = round(min(_v), 1)
                N['ab_err_%s_hi' % _tag] = round(max(_v), 1)

    # 附图 S2 的观测支撑度量:图注引用了全球中位距离与欧洲/南美/非洲三档
    _sj = os.path.join(HERE, 'figures', 'figs_v2', 'figED2_support.json')
    if os.path.exists(_sj):
        _s = json.load(open(_sj, encoding='utf-8'))
        N['support_median_dist_km'] = int(round(_s['median_dist_km_global']))
        for _k, _tag in (('Europe', 'eu'), ('S. America', 'sa'), ('Africa', 'af')):
            if _k in _s['per_region']:
                N['support_dist_km_' + _tag] = int(round(
                    _s['per_region'][_k]['median_dist_km']))

    # Fig. 6 的盈亏平衡瓦片数与加速比 —— 由绘图脚本落盘,正文三处引用都要跟它对齐
    _bej = os.path.join(HERE, 'figures', 'figs_v2', 'fig6_breakeven.json')
    if os.path.exists(_bej):
        _be = json.load(open(_bej, encoding='utf-8'))
        N['breakeven_tiles'] = int(round(_be['breakeven_tiles']))
        N['fno_speedup'] = int(round(_be['speedup']))
        if 'fig4b' in _be:
            N['fig4b_n_gt20'] = _be['fig4b']['n_gt20']
            N['fig4b_n_lt20'] = _be['fig4b']['n_lt20']
            N['fig4b_pct_up'] = int(round(_be['fig4b']['pct_raised_gt2']))
            N['fig4b_pct_dn'] = int(round(_be['fig4b']['pct_lowered_gt2']))

    _f1 = os.path.join(HERE, 'figures', 'figs_v2', 'fig1_example.json')
    if os.path.exists(_f1):
        _e1 = json.load(open(_f1, encoding='utf-8'))
        N['fig1_relief_m'] = int(round(_e1['relief_m']))
        N['fig1_Bs_fem'] = round(_e1['B_s_fem'], 2)
        N['fig1_Bs_op'] = round(_e1['B_s_operator'], 2)
        N['fig1_ve'] = _e1['vertical_exaggeration']
        N['fig1_iso_step'] = int(_e1['isotherm_step_C'])

    _rj = os.path.join(HERE, 'site_reliability.json')
    if os.path.exists(_rj):
        _r = json.load(open(_rj, encoding='utf-8'))
        for _g in 'ABC':
            N['rel_%s_n' % _g] = _r['grade_' + _g]['n']
            N['rel_%s_pct' % _g] = round(_r['grade_' + _g]['pct'], 1)
        N['rel_C_med_corr_mW'] = round(_r['grade_C']['median_abs_corr_mW'], 2)
        N['rel_C_med_corr_pct'] = round(_r['grade_C']['median_abs_corr_pct'], 2)
        _fl = _r['by_band']['<0.02']
        N['rel_C_flat_share_pct'] = round(100 * round(_fl['C'] / 100 * _fl['n'])
                                          / _r['grade_C']['n'], 1)
        N['rel_flat_sigma_mW'] = round(_fl['median_sigma_mW'], 2)
        N['rel_gt020_n'] = _r['by_band']['>0.20']['n']
        N['rel_ge002_n'] = _r['ge002']['n']
        N['rel_ge002_AB_pct'] = round(_r['ge002']['pct_AB'], 1)
        N['rel_ge002_share_pct'] = round(100 * _r['ge002']['n'] / _r['n'], 1)
        _vm = _r['vs_measurement']
        N['rel_u_rated_n'] = _vm['all_rated']['n']
        N['rel_u_exceed_n'] = _vm['all_rated']['n_exceed']
        N['rel_u_exceed_A_n'] = _vm['exceed_and_grade_A']
        N['rel_u1_n'] = _vm['U1']['n']
        N['rel_u1_exceed_n'] = _vm['U1']['n_exceed']
        N['rel_u1_exceed_pct'] = round(_vm['U1']['pct_exceed'], 1)
        # 稳健口径:扣除 2 倍方法误差后仍超过测量不确定度上限
        N['rel_rob_n'] = _vm['robust']['n_exceed']
        N['rel_rob_u1_n'] = _vm['robust']['u1_n_exceed']
        N['rel_rob_u1_pct'] = round(_vm['robust']['u1_pct_exceed'], 1)
        N['rel_flat_share_pct'] = round(100 * _fl['n'] / _r['n'], 1)
        N['rel_flat_snr'] = round(_fl['median_snr'], 2)
        N['rel_snr_02_05'] = round(_r['by_band']['0.02-0.05']['median_snr'], 2)
    _sj2 = os.path.join(HERE, 'evaluation_scorecard.json')
    if os.path.exists(_sj2):
        _sc = json.load(open(_sj2, encoding='utf-8'))
        N['sc_n_rows'] = len(_sc['rows'])
        N['sc_n_pass'] = _sc['summary']['通过']
        N['sc_n_part'] = _sc['summary']['部分通过']
        N['sc_n_fail'] = _sc['summary']['未通过']

    try:
        _fs = json.load(open(os.path.join(HERE, 'fem_speed_benchmark.json'), encoding='utf-8'))
        _fn = json.load(open(os.path.join(HERE, 'fno_speed_benchmark.json'), encoding='utf-8'))
        _tm = json.load(open(os.path.join(HERE, 'fno3d_large_metrics.json'), encoding='utf-8'))
        _fem_h = _fs['fem_total_ms'] / 1000 * _tm['n_total'] / 3600       # 生成训练集的有限元耗时
        N['onetime_fem_h'] = round(_fem_h, 1)
        N['onetime_total_h'] = int(round(_fem_h + _tm['train_time_s'] / 3600))
        N['fno_batch128_ms'] = round(_fn['batch128_per_tile_ms'], 1)
        # 8 朝向推理(生产口径):单瓦片一次调用 = 8 个朝向一批;批处理 = 8 × 批量摊薄
        N['fno_d4_single_ms'] = round(_fn['batch8_call_ms'], 1)
        N['fno_d4_batch_ms'] = round(8 * _fn['batch128_per_tile_ms'], 1)
        _gbj = json.load(open(os.path.join(HERE, 'global_B_field_05deg.json'), encoding='utf-8'))
        _nsite = int(np.load(os.path.join(HERE, 'rebuild_points.npz'))['B'].shape[0])
        _d4s = 8 * _fn['batch128_per_tile_ms'] / 1000
        N['fno_d4_sites_min'] = round(_nsite * _d4s / 60, 1)
        N['fno_d4_grid05_min'] = round(_gbj['n_solved'] * _d4s / 60, 0)
        N['fno_d4_grid01_h'] = round(_gbj['n_solved'] * 25 * _d4s / 3600, 1)
        _fp = json.load(open(os.path.join(HERE, 'footprint_fixed_resolution_test.json'), encoding='utf-8'))
        N['fp_005_01_lo57'] = round(_fp['summary']['0.05-0.1']['median_absB1']['57'], 4)
        _rel = [b['ratio'] for b in json.load(open(os.path.join(HERE, 'dropout_calibration_test.json'),
                                                   encoding='utf-8'))['reliability']]
        N['dropout_ratio_min'] = round(min(_rel), 1)
        N['dropout_ratio_max'] = round(max(_rel), 1)
    except Exception as _e:
        print('  ⚠ 预审口径键跳过:', _e)

    def _j(f):
        pth = os.path.join(HERE, f)
        return json.load(open(pth, encoding='utf-8')) if os.path.exists(pth) else None
    _sr = _j('site_reliability.json')
    if _sr:
        _dc = _sr['decile_cross']
        N['dec_all_pct'] = round(_dc['all_pct'], 1)
        N['dec_A_pct'] = round(_dc['A_pct_of_all'], 1)
        N['dec_AB_pct'] = round(_dc['AB_pct_of_all'], 1)
        N['dec_C_share_pct'] = round(_dc['share_from_C'], 1)
        N['dec_boundary_pct'] = round(_dc['on_boundary_pct'], 1)
        N['rel_ge002_A_pct'] = round(_sr['ge002']['pct_A'], 1)
        N['rel_rob_pct'] = round(_sr['vs_measurement']['robust']['pct_exceed'], 1)
        for _b, _t in (('<0.02', 'b0'), ('0.02-0.05', 'b1'), ('0.05-0.10', 'b2'), ('0.10-0.20', 'b3'), ('>0.20', 'b4')):
            N[f'rel_A_pct_{_t}'] = round(_sr['by_band'][_b]['A'], 1)
            N[f'rel_relsig_{_t}'] = round(_sr['by_band'][_b]['median_rel_sigma_pct'], 2)
    _vm = _j('valley_bias_matched.json')
    if _vm:
        N['vm_n_pairs'] = _vm['n_pairs']
        N['vm_n_rand'] = _vm['n_rand']
        N['vm_site_fem_pct'] = round(_vm['测点']['fem'], 1)
        N['vm_local_fem_pct'] = round(_vm['局地随机点']['fem'], 1)
        N['vm_rand_fem_pct'] = round(_vm['全球随机陆地']['fem'], 1)
        _ex = _vm['excess']['fem']
        N['vm_global_pp'] = round(_ex['global_pp'], 1)
        N['vm_global_lo'] = round(_ex['global_ci'][0], 1)
        N['vm_global_hi'] = round(_ex['global_ci'][1], 1)
        N['vm_local_pp'] = round(_ex['local_pp'], 1)
        N['vm_local_lo'] = round(_ex['local_ci'][0], 1)
        N['vm_local_hi'] = round(_ex['local_ci'][1], 1)
        N['vm_op_global_pp'] = round(_vm['excess']['op']['global_pp'], 1)
        for _b, _t in (('<100 m', 'r0'), ('100-500 m', 'r1'), ('500-1000 m', 'r2'), ('>1000 m', 'r3')):
            _r = _vm['by_relief'][_b]
            N[f'vm_{_t}_n'] = _r['n_pairs']
            N[f'vm_{_t}_pp'] = round(_r['local_pp'], 1)
            N[f'vm_{_t}_lo'] = round(_r['local_ci'][0], 1)
            N[f'vm_{_t}_hi'] = round(_r['local_ci'][1], 1)
            N[f'vm_{_t}_site_fem'] = round(_r['site_fem'], 1)
            N[f'vm_{_t}_site_op'] = round(_r['site_op'], 1)
        N['vm_pos_site'] = round(_vm['topo_position_below_mean']['site'], 1)
        N['vm_pos_local'] = round(_vm['topo_position_below_mean']['local'], 1)
        N['vm_pos_pp'] = round(_vm['topo_position_below_mean']['local_pp'], 1)
        N['vm_pos_lo'] = round(_vm['topo_position_below_mean']['local_ci'][0], 1)
        N['vm_pos_hi'] = round(_vm['topo_position_below_mean']['local_ci'][1], 1)
    _ms = _j('magnitude_scaling_test.json')
    if _ms:
        for _g in ('4.0', '5.5', '6.5'):
            N['lapse_%s_med' % _g.replace('.', '')] = round(_ms['lapse_factor'][_g]['median'], 2)
        _dv = _ms['davies']
        for _b, _t in (('全部', 'all'), ('>= 0.05', 'ge005'), ('>= 0.10', 'ge010')):
            N[f'dvc_{_t}_lo'] = round(_dv[_b]['ci_cluster'][0], 2)
            N[f'dvc_{_t}_hi'] = round(_dv[_b]['ci_cluster'][1], 2)
            N[f'dvc_{_t}_lapse55'] = round(_dv[_b]['slope_lapse_bg_5.5'], 2)
            N[f'dvc_{_t}_lapse55_qobs'] = round(_dv[_b]['slope_lapse_5.5'], 2)
            N[f'dvc_{_t}_lapse55_lo'] = round(_dv[_b]['slope_lapse_bg_5.5_ci_cluster'][0], 2)
            N[f'dvc_{_t}_lapse55_hi'] = round(_dv[_b]['slope_lapse_bg_5.5_ci_cluster'][1], 2)
            N[f'dvc_{_t}_log'] = round(_dv[_b]['slope_log'], 2)
        N['dvc_ncells_ge005'] = _dv['>= 0.05']['n_cells']
        _pb = _ms['paired']
        for _b, _t in (('< 0.02', 'b0'), ('0.02-0.05', 'b1'), ('0.05-0.10', 'b2'), ('0.10-0.20', 'b3'), ('> 0.20', 'b4')):
            N[f'pair_net_mad_{_t}'] = round(_pb[_b]['mad_reduction_at_1'], 2)
            N[f'pair_net_sd_{_t}'] = round(_pb[_b]['sd_reduction_at_1'], 2)
            N[f'pair_alpha_mad_{_t}'] = _pb[_b]['alpha_best_mad']
            N[f'pair_sd_increase_{_t}'] = round(-_pb[_b]['sd_reduction_at_1'], 1)
            N[f'pair_alpha_sd_{_t}'] = _pb[_b]['alpha_best_sd']
        N['pair_mad_all_increase'] = round(-_pb['全部']['mad_reduction_at_1'], 2)
        _al = [_pb[b]['alpha_best_sd'] for b in _pb if b != '全部']
        N['pair_alpha_sd_min'], N['pair_alpha_sd_max'] = min(_al), max(_al)
        _al = [_pb[b]['alpha_best_mad'] for b in _pb if b != '全部']
        N['pair_alpha_mad_min'], N['pair_alpha_mad_max'] = min(_al), max(_al)
    _pn = _j('paired_null_fairness_test.json')
    if _pn:
        for _b, _t in (('< 0.02', 'b0'), ('0.02-0.05', 'b1'), ('0.05-0.10', 'b2'), ('0.10-0.20', 'b3'), ('> 0.20', 'b4')):
            N[f'pair_marginB_{_t}'] = round(_pn[_b]['marginB'], 2)
            N[f'pair_pB_{_t}'] = round(_pn[_b]['p_B'], 4)
    _dv2 = _j('davies_validation.json')
    if _dv2:
        N['davies_rho_b10_p'] = float('%.2g' % _dv2['spearman_ge010']['p'])
    _dp = os.path.join(HERE, 'davies_points.npz')
    if os.path.exists(_dp):
        _r = np.load(_dp)['resid']
        N['davies_resid_median_pct'] = round(100 * float(np.median(_r)), 1)
        N['davies_resid_mean_pct'] = round(100 * float(np.mean(_r)), 1)
    _lb = _j('linear_spectral_baseline.json')
    if _lb:
        N['lin_r2_center'] = round(_lb['center']['linear41']['r2'], 3)
        N['op_r2_center'] = round(_lb['center']['operator']['r2'], 3)
        N['lin_r2_field'] = round(_lb['field']['linear41']['r2'], 3)
        N['lin_rmse_center'] = round(_lb['center']['linear41']['rmse'], 4)
        N['op_rmse_center'] = round(_lb['center']['operator']['rmse'], 4)
        for _b, _t in (('<0.02', 'b0'), ('0.02-0.05', 'b1'), ('0.05-0.10', 'b2'), ('>=0.10', 'b3')):
            N[f'lin_rmse_{_t}'] = round(_lb['center_by_band'][_b]['linear41']['rmse'], 4)
            N[f'opl_rmse_{_t}'] = round(_lb['center_by_band'][_b]['operator']['rmse'], 4)
            N[f'lin_bias_{_t}'] = round(_lb['center_by_band'][_b]['linear41']['bias'], 4)
            N[f'opl_bias_{_t}'] = round(_lb['center_by_band'][_b]['operator']['bias'], 4)
            N[f'lin_n_{_t}'] = _lb['center_by_band'][_b]['n']
        N['lin_amp_ratio'] = round(_lb['center']['linear41']['amp_ratio_median'], 2)
        if 'ms_per_tile_cpu' in _lb:
            N['lin_ms_per_tile'] = round(_lb['ms_per_tile_cpu'], 2)
        N['op_amp_ratio'] = round(_lb['center']['operator']['amp_ratio_median'], 2)
    _co = _j('c15_ordinal_test.json')
    if _co:
        N['c15_n_na'] = [g['n'] for g in _co['groups'] if not g['in_ordinal_scale']][0]   # 「未评估」对照组
    _ct = _j('c15_terrain_baseline.json')
    if _ct:
        N['c15tb_delta_relief'] = round(_ct['delta_relief'], 2)
        N['c15tb_delta_absB1'] = round(_ct['delta_absB1'], 2)
        N['c15tb_coef_B'] = round(_ct['logit_both']['absB1']['coef'], 2)
        N['c15tb_coef_B_only'] = round(_ct['logit_absB1_only']['coef'], 2)
        N['c15tb_p_B'] = float('%.2g' % _ct['logit_both']['absB1']['p'])
        N['c15tb_coef_relief'] = round(_ct['logit_both']['relief']['coef'], 2)
        N['c15tb_relief_lo'] = int(round(_ct['median_low_relief']))
        N['c15tb_relief_hi'] = int(round(_ct['median_high_relief']))
    _fo = _j('fem_orientation_test.json')
    if _fo:
        N['fo_fem_rmse_center'] = round(_fo['overall']['rmse_center_fem1_vs_sym'], 4)
        N['fo_op8_r2'] = round(_fo['overall']['r2_field_op8_vs_sym'], 3)
        N['fo_op8_bias_r0'] = round(_fo['relief_0']['op8_bias_vs_sym'], 5)      # 8 向平均后近平坦地形的正偏差
    _ds = _j('depth_decay_spectral.json')
    if _ds:
        N['fe_D0_median_pct'] = _ds['fe_D0_median_pct']
        N['fe_D0_max_pct'] = _ds['fe_D0_max_pct']
    # 观测格点数与陆地覆盖率:与 krige_global_ok.bin_medians 同一分箱口径(floor 到 0.5°)
    _kp = os.path.join(HERE, 'kriged_heatflow_05deg_global.npz')
    _rp2 = os.path.join(HERE, 'rebuild_points.npz')
    if os.path.exists(_kp) and os.path.exists(_rp2):
        _k = np.load(_kp); _d2 = np.load(_rp2)
        _jj = np.clip(((_d2['lon'] + 180) / 0.5).astype(int), 0, 719)
        _ii = np.clip(((_d2['lat'] + 90) / 0.5).astype(int), 0, 359)
        _cells = np.unique(_ii.astype(np.int64) * 720 + _jj)
        N['krige_obs_cells'] = int(len(_cells))
        _land = np.isfinite(_k['krige_qm'])
        _lat_c = -89.75 + (_cells // 720) * 0.5
        _lon_c = -179.75 + (_cells % 720) * 0.5
        _li = np.array([int(np.argmin(np.abs(_k['lats'] - v))) for v in _lat_c])
        _lj = np.array([int(np.argmin(np.abs(_k['lons'] - v))) for v in _lon_c])
        N['obs_cell_coverage_pct'] = round(100 * float(_land[_li, _lj].sum()) / float(_land.sum()), 1)
    if 'point_mean_shift' in N and 'grid_delta' in N and N['grid_delta']:
        N['point_grid_ratio'] = round(float(N['point_mean_shift']) / float(N['grid_delta']), 1)
    _hd = _j('holdout_d4_r2.json')
    if _hd:
        N['r2_holdout_d4'] = round(_hd['r2_d4'], 3)
        N['r2_holdout_center_d4'] = round(_hd['r2_center_d4'], 3)
        N['mae_holdout_d4'] = round(_hd['mae_d4'], 4)
        N['r2_holdout_single'] = round(_hd['r2_single'], 3)
    _ex = _j('validation_independent_excl.json')
    if _ex:
        N['excl_km'] = int(round(_ex['excl_km']))
        N['excl_r2_field'] = round(_ex['r2_field'], 3)
        N['excl_r2_center'] = round(_ex['r2_center'], 3)
        N['excl_rmse_center'] = round(_ex['rmse_center'], 4)
        N['excl_lin_r2_center'] = round(_ex['linear_r2_center'], 3)
        N['excl_median_dist_km'] = int(round(_ex['median_dist_km']))
        for _b, _t in (('0.02-0.10', 'mid'), ('>=0.10', 'high')):
            N[f'excl_op_rmse_{_t}'] = round(_ex['by_band'][_b]['rmse_center'], 4)
            N[f'excl_lin_rmse_{_t}'] = round(_ex['by_band'][_b]['linear_rmse_center'], 4)

    _lb2 = _j('linear_spectral_baseline.json')
    if _lb2:
        for _k, _b in enumerate(('<0.02', '0.02-0.05', '0.05-0.10', '>=0.10')):
            N['ind_op_r2_b%d' % _k] = round(_lb2['center_by_band'][_b]['operator']['r2'], 3)
            N['ind_lin_r2_b%d' % _k] = round(_lb2['center_by_band'][_b]['linear41']['r2'], 3)
    _exz = os.path.join(HERE, 'validation_independent_excl.npz')
    if os.path.exists(_exz):
        _z = np.load(_exz)
        _t, _p, _l = _z['B_fem'][:, 20, 20], _z['B_fno'][:, 20, 20], _z['B_linear_center']
        _a = np.abs(_t - 1.0)

        def _r2(t_, p_):
            return 1.0 - float(((p_ - t_) ** 2).sum() / ((t_ - t_.mean()) ** 2).sum())
        for _k, (_lo, _hi) in enumerate(((0, 0.02), (0.02, 0.10), (0.10, np.inf))):
            _m = (_a >= _lo) & (_a < _hi)
            N['excl_op_r2_b%d' % _k] = round(_r2(_t[_m], _p[_m]), 3)
            N['excl_lin_r2_b%d' % _k] = round(_r2(_t[_m], _l[_m]), 3)
            N['excl_n_b%d' % _k] = int(_m.sum())

    _ap2 = _j('analytic_benchmark_wavelength.json')
    if _ap2:   # 以山脊测点所在列实际顶层厚度为参照时,9.5 与 7.1 km 两档的离散误差(S1.1)
        _v = [r['err_vs_actual_layer_pct'] for r in _ap2['rows'] if r['lambda_km'] in (9.5, 7.125)]
        N['ab_err_actual_lo'] = round(min(_v), 1)
        N['ab_err_actual_hi'] = round(max(_v), 1)
    _vm = _j('vertical_mesh_test.json')
    if _vm:    # 垂向加密试验的测点数(2.1.1、附图 S2c)
        N['vmesh_n'] = len(_vm['rows'])
    _ms = _j('magnitude_scaling_test.json')
    if _ms:    # 按区域地温梯度缩放后 |B−1| 的中位缩放(3.3.3)
        _dv = _ms['davies']
        N['lapse_bg_med_all'] = round(list(_dv.values())[0]['lapse_bg_factor_median'], 2)
        N['lapse_bg_med_ge005'] = round(_dv['>= 0.05']['lapse_bg_factor_median'], 2)
    _kz = os.path.join(HERE, 'kriged_heatflow_05deg_global.npz')
    if os.path.exists(_kz):   # 克里金标准差的四分位距(S2.2)
        _k = np.load(_kz)
        for _f, _t in (('std_qm', 'qm'), ('std_dq', 'dq')):
            _x = _k[_f][np.isfinite(_k[_f])]
            N['krige_std_%s_q25' % _t] = round(float(np.percentile(_x, 25)), 2)
            N['krige_std_%s_q75' % _t] = round(float(np.percentile(_x, 75)), 2)

    json.dump(N, open(os.path.join(HERE, 'numbers.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'已生成 numbers.json,共 {len(N)} 个键')
    for k2 in ['n_sites', 'median_correction_pct', 'p99_correction_pct', 'frac_gt10pct',
               'frac_change_decile_pct', 'valley_excess_pp', 'valley_p']:
        print(f'  {k2:<28} {N.get(k2)}')


if __name__ == '__main__':
    main()
