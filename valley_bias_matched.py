# -*- coding: utf-8 -*-
"""谷地选址偏倚:同一估计器、局地匹配零假设、分层 —— 一次回答三个问题。


本脚本对三类位置各取瓦片,**同时**求有限元中心 B 与算子中心 B:
  site    从 36,392 个测点中均匀随机抽 N_SITE 个
  local   每个被抽中的测点,在其周围 R_LOCAL_KM 内均匀随机取一个陆地点(局地零假设)
  rand    南极以外陆地上按面积均匀随机取 N_RAND 个点(全球零假设)

口径与应用集一致:瓦片读取、锥化、域深规则、网格与生产流程相同;不剔除沿海瓦片
(应用集也不剔除);点本身必须落在陆地上(Natural Earth 110m 陆地且高程 > 0 m)。
格陵兰冰盖内部(北纬 59° 以北、西经 75°–10°、高程 > 1,000 m)不属于"无冰陆地",剔除。

另记录每块瓦片的地形位置 rel_elev =(中心高程 − 瓦片均值)/ 未锥化起伏:
"钻孔打在谷底"说的是地形位置,而 B > 1 是通量增强,二者不是一回事,平行报告。

输出 valley_bias_matched.npz(逐瓦片)与 valley_bias_matched.json(汇总,含空间块自助区间)。
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
DEM_DEFAULT = os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif'))
SEED = 20260922
R_LOCAL_KM = 10.0
RELIEF_BANDS = [(0, 100, '<100 m'), (100, 500, '100-500 m'), (500, 1000, '500-1000 m'),
                (1000, 1e9, '>1000 m')]
_DS = None


def _init(dem_path):
    global _DS
    import rasterio
    sys.path.insert(0, HERE)
    _DS = rasterio.open(dem_path)


def _elev(lon, lat):
    """点位高程(原生像元);越界返回 None。"""
    try:
        r, c = _DS.index(lon, lat)
        if not (0 <= r < _DS.height):
            return None
        return float(_DS.read(1, window=((r, r + 1), (c % _DS.width, c % _DS.width + 1)))[0, 0])
    except Exception:
        return None


def _greenland_ice(lon, lat, z):
    return lat > 59 and -75 < lon < -10 and z > 1000


def work(task):
    kind, idx, lon, lat, seed = task
    from tile_reader import read_tile, taper_dem, CELL_M, RES
    from fem3d import Fem3D
    try:
        if kind == 'local':                      # 在测点周围 R_LOCAL_KM 内找一个陆地点
            rng = np.random.default_rng(seed)
            lon0, lat0 = lon, lat
            for _ in range(30):
                r = R_LOCAL_KM * np.sqrt(rng.random())
                th = 2 * np.pi * rng.random()
                la = lat0 + r * np.cos(th) / 111.32
                lo = lon0 + r * np.sin(th) / (111.32 * max(np.cos(np.radians(lat0)), 1e-3))
                z = _elev(lo, la)
                if z is not None and z > 0:
                    lon, lat = float(lo), float(la)
                    break
            else:
                return None
        elif kind == 'rand':
            z = _elev(lon, lat)
            if z is None or z <= 0 or _greenland_ice(lon, lat, z):
                return None
        raw = read_tile(_DS, lon, lat, apply_taper=False)
        if raw is None:
            return None
        dem = taper_dem(raw)
        c = RES // 2
        rr = float(raw.max() - raw.min())
        relief = float(dem.max() - dem.min())
        f = Fem3D(nx=40, ny=40, nz=10)
        f.build(dem, CELL_M, depth=max(3000.0, 1.5 * relief))
        f.solve_heat()
        f.compute_B()
        G = f.B_grid
        if not np.all(np.isfinite(G)):
            return None
        return dict(kind=kind, idx=idx, lon=lon, lat=lat,
                    B_fem=float(G[G.shape[0] // 2, G.shape[1] // 2]), relief=relief,
                    rel_elev=(float(raw[c, c]) - float(raw.mean())) / rr if rr > 0 else 0.0,
                    dem=dem.astype(np.float32))
    except Exception:
        return None


def random_land(n, rng):
    """南极以外陆地上按面积均匀的候选点(Natural Earth 110m 陆地)。"""
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from shapely.prepared import prep
    land = prep(unary_union(list(shpreader.Reader(shpreader.natural_earth(
        resolution='110m', category='physical', name='land')).geometries())))
    s0, s1 = np.sin(np.radians(-60)), np.sin(np.radians(84))
    out = []
    while len(out) < n:
        lo = rng.uniform(-180, 180)
        la = float(np.degrees(np.arcsin(rng.uniform(s0, s1))))
        if land.contains(Point(lo, la)):
            out.append((lo, la))
    return out


def predict(dems):
    import torch
    sys.path.insert(0, HERE)
    from step3_train_large import FNO3D
    ck = torch.load(os.path.join(HERE, 'fno3d_large_model.pt'), map_location='cuda',
                    weights_only=False)
    m = FNO3D(modes=24, width=96, H_out=41, W_out=41, n_layers=6).cuda().eval()
    m.load_state_dict(ck['state'])
    out = []
    with torch.no_grad():
        for i in range(0, len(dems), 64):
            x = (dems[i:i + 64, None] - ck['X_mean']) / ck['X_std']
            y = m(torch.from_numpy(x.astype(np.float32)).cuda()).cpu().numpy()
            out.append(y[:, 0, 20, 20] * ck['Y_std'] + ck['Y_mean'])
    return np.concatenate(out).astype(float)


def block_boot(values_by_group, groups, stat, n_boot=2000, seed=SEED):
    """按空间块(2° 格)重采样的自助区间。values_by_group 为逐单位数组的字典。"""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    members = {g: np.nonzero(groups == g)[0] for g in ug}
    res = []
    for _ in range(n_boot):
        pick = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
        res.append(stat({k: v[pick] for k, v in values_by_group.items()}))
    return float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dem', default=DEM_DEFAULT)
    ap.add_argument('--n-site', type=int, default=2000)
    ap.add_argument('--n-rand', type=int, default=2000)
    ap.add_argument('--procs', type=int, default=10)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    pick = np.sort(rng.choice(len(d['lon']), a.n_site, replace=False))
    tasks = [('site', int(i), float(d['lon'][i]), float(d['lat'][i]), None) for i in pick]
    tasks += [('local', int(i), float(d['lon'][i]), float(d['lat'][i]), SEED + int(i)) for i in pick]
    cand = random_land(int(a.n_rand * 1.3), rng)          # 多备候选,剔除海面/冰盖后取前 n_rand
    tasks += [('rand', k, lo, la, None) for k, (lo, la) in enumerate(cand)]
    print('任务 %d 个(测点 %d、局地点 %d、随机陆地候选 %d),进程 %d'
          % (len(tasks), a.n_site, a.n_site, len(cand), a.procs), flush=True)

    t0 = time.time()
    res = []
    with Pool(a.procs, initializer=_init, initargs=(a.dem,)) as p:
        for k, r in enumerate(p.imap_unordered(work, tasks, chunksize=4), 1):
            if r is not None:
                res.append(r)
            if k % 500 == 0:
                el = time.time() - t0
                print('  %d/%d  %.1f min,预计还需 %.1f min'
                      % (k, len(tasks), el / 60, el / k * (len(tasks) - k) / 60), flush=True)

    rand = sorted([r for r in res if r['kind'] == 'rand'], key=lambda r: r['idx'])[:a.n_rand]
    site = {r['idx']: r for r in res if r['kind'] == 'site'}
    local = {r['idx']: r for r in res if r['kind'] == 'local'}
    both = sorted(set(site) & set(local))                  # 只保留测点与局地点都成功的对
    rows = [site[i] for i in both] + [local[i] for i in both] + rand
    kind = np.array([r['kind'] for r in rows])
    dems = np.stack([r['dem'] for r in rows])
    print('有效:测点—局地点 %d 对,随机陆地 %d 个;有限元用时 %.1f min'
          % (len(both), len(rand), (time.time() - t0) / 60), flush=True)
    B_op = predict(dems)

    Bf = np.array([r['B_fem'] for r in rows])
    rel = np.array([r['relief'] for r in rows])
    rpos = np.array([r['rel_elev'] for r in rows])
    lon = np.array([r['lon'] for r in rows])
    lat = np.array([r['lat'] for r in rows])
    idx = np.array([r['idx'] for r in rows])
    np.savez_compressed(os.path.join(HERE, 'valley_bias_matched.npz'), kind=kind, idx=idx,
                        lon=lon, lat=lat, B_fem=Bf, B_op=B_op, relief=rel, rel_elev=rpos)

    S, L, R = kind == 'site', kind == 'local', kind == 'rand'
    # 核对:同一流程下,测点上的算子值应与 rebuild_points 的 B_surface 一致
    chk = np.abs(B_op[S] - d['B_surface'][idx[S]]).max()
    print('核对:测点算子值与 rebuild_points.B_surface 最大差 %.2e' % chk)

    blk = (np.floor(lon[S] / 2) * 1000 + np.floor(lat[S] / 2)).astype(int)
    out = dict(n_pairs=int(S.sum()), n_rand=int(R.sum()), r_local_km=R_LOCAL_KM,
               check_op_vs_rebuild_maxdiff=float(chk),
               full_sites_op_pct=float(100 * (d['B_surface'] > 1).mean()))

    def frac(x):
        return float(100 * np.mean(x))

    print('\n══ B > 1 占比(%) ══')
    print('%-20s %10s %10s' % ('', '有限元', '算子'))
    for nm, m in (('测点', S), ('局地随机点', L), ('全球随机陆地', R)):
        out[nm] = dict(fem=frac(Bf[m] > 1), op=frac(B_op[m] > 1), n=int(m.sum()))
        print('%-20s %10.1f %10.1f   n=%d' % (nm, out[nm]['fem'], out[nm]['op'], m.sum()))
    print('  (全部 36,392 测点的算子值:%.1f%%)' % out['full_sites_op_pct'])

    # 超出量与区间:测点侧按 2° 块自助;局地比较是配对的,测点与其局地点同进同出
    sF, lF, sO, lO = (Bf[S] > 1), (Bf[L] > 1), (B_op[S] > 1), (B_op[L] > 1)
    rF, rO = (Bf[R] > 1), (B_op[R] > 1)
    se_r = lambda x: 100 * np.sqrt(x.mean() * (1 - x.mean()) / len(x))
    ex = {}
    for nm, est, s_, l_, r_ in (('fem', '有限元', sF, lF, rF), ('op', '算子', sO, lO, rO)):
        loc = frac(s_) - frac(l_)
        ci_loc = block_boot({'s': s_.astype(float), 'l': l_.astype(float)}, blk,
                            lambda v: 100 * (v['s'].mean() - v['l'].mean()))
        glo = frac(s_) - frac(r_)
        ci_s = block_boot({'s': s_.astype(float)}, blk, lambda v: 100 * v['s'].mean())
        half = np.sqrt(((ci_s[1] - ci_s[0]) / 3.92) ** 2 + se_r(r_) ** 2) * 1.96
        ex[nm] = dict(local_pp=loc, local_ci=ci_loc, global_pp=glo,
                      global_ci=(glo - half, glo + half))
        print('%s:测点 − 局地点 %+.1f pp(95%% 块自助 %+.1f 至 %+.1f);测点 − 全球随机 %+.1f pp(±%.1f)'
              % (est, loc, ci_loc[0], ci_loc[1], glo, half))
    out['excess'] = ex

    print('\n══ 按测点瓦片起伏分层(有限元;局地点随其测点归档,全球随机按自身起伏归档)══')
    strat = {}
    for lo_, hi_, lab in RELIEF_BANDS:
        ms = (rel[S] >= lo_) & (rel[S] < hi_)
        mr = (rel[R] >= lo_) & (rel[R] < hi_)
        if ms.sum() < 20:
            continue
        row = dict(n_pairs=int(ms.sum()), n_rand=int(mr.sum()),
                   site_fem=frac(sF[ms]), local_fem=frac(lF[ms]),
                   rand_fem=frac(rF[mr]) if mr.sum() else None,
                   site_op=frac(sO[ms]), local_op=frac(lO[ms]),
                   rand_op=frac(rO[mr]) if mr.sum() else None)
        row['local_pp'] = row['site_fem'] - row['local_fem']
        row['local_ci'] = block_boot({'s': sF[ms].astype(float), 'l': lF[ms].astype(float)},
                                     blk[ms], lambda v: 100 * (v['s'].mean() - v['l'].mean()))
        strat[lab] = row
        print('  %-11s 对数 %4d  测点 %.1f / 局地 %.1f / 全球 %s   测点−局地 %+.1f pp(%+.1f 至 %+.1f)'
              % (lab, ms.sum(), row['site_fem'], row['local_fem'],
                 '%.1f' % row['rand_fem'] if row['rand_fem'] is not None else '—',
                 row['local_pp'], *row['local_ci']))
    out['by_relief'] = strat

    print('\n══ 地形位置(中心低于瓦片均值的比例,%)══')
    pos = dict(site=frac(rpos[S] < 0), local=frac(rpos[L] < 0), rand=frac(rpos[R] < 0))
    pos['local_pp'] = pos['site'] - pos['local']
    pos['local_ci'] = block_boot({'s': (rpos[S] < 0).astype(float), 'l': (rpos[L] < 0).astype(float)},
                                 blk, lambda v: 100 * (v['s'].mean() - v['l'].mean()))
    out['topo_position_below_mean'] = pos
    print('  测点 %.1f / 局地 %.1f / 全球 %.1f;测点 − 局地 %+.1f pp(%+.1f 至 %+.1f)'
          % (pos['site'], pos['local'], pos['rand'], pos['local_pp'], *pos['local_ci']))

    print('\n══ 算子相对有限元:符号一致率与偏差(全部 %d 块瓦片)══' % len(Bf))
    agree = {}
    for lo_, hi_, lab in ((0, .005, '<0.005'), (.005, .02, '0.005-0.02'), (.02, .10, '0.02-0.10'),
                          (.10, 9, '>=0.10')):
        m = (np.abs(Bf - 1) >= lo_) & (np.abs(Bf - 1) < hi_)
        agree[lab] = dict(n=int(m.sum()), sign_agree=frac(np.sign(Bf[m] - 1) == np.sign(B_op[m] - 1)),
                          bias=float(np.mean(B_op[m] - Bf[m])),
                          op_gt1=frac(B_op[m] > 1), fem_gt1=frac(Bf[m] > 1))
        print('  |B_fem−1| %-11s n=%5d  一致 %.1f%%  偏差 %+.5f  算子判 B>1 %.1f%% / 有限元 %.1f%%'
              % (lab, m.sum(), agree[lab]['sign_agree'], agree[lab]['bias'],
                 agree[lab]['op_gt1'], agree[lab]['fem_gt1']))
    out['operator_vs_fem'] = agree

    json.dump(out, open(os.path.join(HERE, 'valley_bias_matched.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('\n已保存 valley_bias_matched.json / .npz,总用时 %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
