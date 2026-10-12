#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全球普通克里金:一次性用全部观测格点,不分区、不截断邻居。

为什么要换掉 krige_ihfc_style.py
--------------------------------
旧脚本把世界切成 6 个矩形经纬度区块(N.America / S.America / Europe / Africa /
Asia / Oceania),每块单独拟合变差函数、单独预测,再用
``result[li_min:li_max, lj_min:lj_max] = z`` 整块写入 —— 字典序靠后的区块会
覆盖先写入的重叠部分,交界两侧用的是不同模型。实测(只统计陆地格点)相对
校正场在接缝上的跳变:

    lat +40.0(欧非交界,穿地中海与华北)   2.4 倍本底
    lon +45.0(非亚交界,穿中东与俄罗斯)   2.1 倍本底
    lat +35.0(欧洲南界)                   2.0 倍本底
    lon +110.0(大洋洲西界)                1.6 倍本底

跳变量 1.0-1.25 个百分点。海岸线掩膜遮得住海上那段,穿过陆地的遮不住 ——
全球图上出现与地理无关的直角边界,一眼能看出是算法分区痕迹。

为什么不用移动窗口克里金
------------------------
先试过每格点只取最近 N 个观测格点的移动窗口方案,它天然连续、没有接缝。
但收敛检验否掉了它(1,500 个随机格点,以 N=160 为参照):

    q_meas 场   N=48 中位差 2.90   N=96 中位差 1.19 mW m-2   (场本身 s.d. 11.11)
    dq 场       N=48 中位差 0.106  N=96 中位差 0.056 mW m-2  (场本身 s.d. 0.73)

N 从 48 加到 160,结果一路在变,没有收敛的迹象。原因在变差函数本身:
q_meas 场拟合出 psill=187.3、nugget=499.6、range=159.2° —— 块金是基台的 2.7 倍,
空间相关性极弱而相关距离极大。这种情形下克里金权重趋于均匀,截断到任何
有限邻居数都会系统性偏移。移动窗口对这份数据不适用。

于是回到最朴素也最可辩护的做法:全部 8,232 个分箱观测格点一次性参与,
一个全局变差函数,不分区也不截断。这既消除了接缝,又比旧方案用得更足
(旧方案每区块还要抽样到 3,000 点以内)。

怎么让它跑得动
--------------
普通克里金方程 M [w; mu] = [gamma0; 1],M 是 (n+1)x(n+1) 的对称阵。
逐点求解要 216,000 次,不可行。但预测值不需要逐点的权重 w:

    z = w' z_obs = [w; mu]' [z_obs; 0] = (M^-1 rhs)' [z_obs; 0]
      = rhs' M^-1 [z_obs; 0]          (M 对称,M^-1 也对称)
      = rhs' lam,   其中 lam = M^-1 [z_obs; 0] 与预测点无关,只解一次。

于是每个格点只剩一次长度 n 的点积。克里金方差 sigma^2 = rhs' M^-1 rhs 仍
要逐点,但可以整批交给 LAPACK 的 getrs,一批 2,000 个右端项。实测:
LU 分解 8,233x8,233 用 1.9 s,批量三角回代约 1 s/批。

与 PyKrige 的关系
-----------------
大圆距离用它 core.great_circle_distance 的 arctan2 形式,球状变差函数、
克里金矩阵的符号约定(M[:n,:n] = -gamma、对角置 0、边界行列为 1)全部照抄;
变差函数参数直接取 PyKrige 拟合的结果。--verify 会随机抽格点与 PyKrige
逐点比对,要求最大绝对差小于 1e-8。

用法
----
    python krige_global_ok.py --verify     # 只做一致性验证
    python krige_global_ok.py              # 全量,写 kriged_heatflow_05deg_global.npz
"""
import argparse
import io
import os
import sys
import time

import numpy as np
import scipy.linalg as sla

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
RES = 0.5
BATCH = 2000          # 每批预测点。(BATCH, n) 的距离矩阵约 132 MB
LAT_FLOOR = -60.0     # 南极不产出:全库无南极测点(方法中已声明剔除)。
                      # 旧脚本靠"没有区块覆盖 lat < -56"隐式做到,这里写明。


def great_circle_distance(lon1, lat1, lon2, lat2):
    """与 pykrige.core.great_circle_distance 完全一致的 arctan2 形式,返回度。"""
    lat1 = np.asarray(lat1) * np.pi / 180.0
    lat2 = np.asarray(lat2) * np.pi / 180.0
    dlon = (np.asarray(lon1) - np.asarray(lon2)) * np.pi / 180.0
    c1, s1 = np.cos(lat1), np.sin(lat1)
    c2, s2 = np.cos(lat2), np.sin(lat2)
    cd = np.cos(dlon)
    return 180.0 / np.pi * np.arctan2(
        np.sqrt((c2 * np.sin(dlon)) ** 2 + (c1 * s2 - s1 * c2 * cd) ** 2),
        s1 * s2 + c1 * c2 * cd)


def spherical(params, d):
    """pykrige.variogram_models.spherical_variogram_model,params = [psill, range, nugget]."""
    psill, rng, nugget = float(params[0]), float(params[1]), float(params[2])
    return np.where(d <= rng,
                    psill * (3.0 * d / (2.0 * rng) - d ** 3 / (2.0 * rng ** 3)) + nugget,
                    psill + nugget)


def bin_medians(lon_pts, lat_pts, vals, grid_lon, grid_lat):
    """0.5° 分箱取中位数 —— 与旧脚本同口径,只是把双重 Python 循环换成排序分组。"""
    jj = np.clip(((lon_pts + 180) / RES).astype(int), 0, len(grid_lon) - 1)
    ii = np.clip(((lat_pts + 90) / RES).astype(int), 0, len(grid_lat) - 1)
    key = ii.astype(np.int64) * len(grid_lon) + jj
    order = np.argsort(key, kind='stable')
    key_s, val_s = key[order], vals[order]
    uniq, start = np.unique(key_s, return_index=True)
    ends = np.append(start[1:], len(val_s))
    med = np.array([np.median(val_s[a:b]) for a, b in zip(start, ends)])
    return grid_lon[uniq % len(grid_lon)], grid_lat[uniq // len(grid_lon)], med


def fit_variogram(blon, blat, bval):
    """变差函数交给 PyKrige 拟合,与旧结果同源。"""
    from pykrige.ok import OrdinaryKriging
    ok = OrdinaryKriging(blon, blat, bval, variogram_model='spherical',
                         verbose=False, enable_plotting=False,
                         coordinates_type='geographic')
    return ok, np.asarray(ok.variogram_model_parameters, dtype=float)


def build_system(blon, blat, bval, params):
    """组装并 LU 分解克里金矩阵,返回 (lu_piv, lam, n)。"""
    n = len(bval)
    d = great_circle_distance(blon[:, None], blat[:, None], blon, blat)
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = -spherical(params, d)
    np.fill_diagonal(M, 0.0)
    M[n, :] = 1.0
    M[:, n] = 1.0
    M[n, n] = 0.0
    del d
    lu_piv = sla.lu_factor(M, overwrite_a=True)
    # lam = M^-1 [z_obs; 0] —— 与预测点无关,只解这一次
    rhs = np.zeros(n + 1)
    rhs[:n] = bval
    lam = sla.lu_solve(lu_piv, rhs)
    return lu_piv, lam, n


def krige_points(plon, plat, blon, blat, params, lu_piv, lam, n,
                 want_sigma=True, batch=BATCH, progress=False):
    """全局普通克里金预测。预测值走 lam 点积;方差按批交给 LAPACK 回代。"""
    npt = len(plon)
    z = np.empty(npt)
    sig = np.full(npt, np.nan) if want_sigma else None

    for s in range(0, npt, batch):
        e = min(s + batch, npt)
        d0 = great_circle_distance(plon[s:e, None], plat[s:e, None], blon, blat)
        RHS = np.empty((e - s, n + 1))
        RHS[:, :n] = -spherical(params, d0)
        # exact_values=True:预测点落在观测点上时精确复现该点,与 pykrige 同
        RHS[:, :n] = np.where(np.abs(d0) <= 1e-10, 0.0, RHS[:, :n])
        RHS[:, n] = 1.0
        del d0

        z[s:e] = RHS @ lam
        if want_sigma:
            X = sla.lu_solve(lu_piv, RHS.T)             # (n+1, m)
            sig[s:e] = np.sqrt(np.maximum(
                -np.einsum('ij,ji->i', RHS, X), 0.0))
            del X
        del RHS
        if progress and (s // batch) % 20 == 0:
            print('    {:>7,}/{:,}'.format(e, npt), flush=True)
    return z, sig


def verify(blon, blat, bval, n_test=150, seed=7, n_sub=900):
    """与 PyKrige 逐点比对。

    观测集降到 n_sub 再比:PyKrige 每调用一次 execute() 都要重建并求逆一遍
    n x n 的矩阵,8,232 个点时逐点比对跑不完。两种实现是否一致与观测集多大
    无关 —— 比的是"同样的点、同样的变差函数参数,解出来是不是同一个数"。
    """
    from pykrige.ok import OrdinaryKriging
    rng = np.random.RandomState(seed)
    sub = rng.choice(len(bval), min(n_sub, len(bval)), replace=False)
    sl, sa, sv = blon[sub], blat[sub], bval[sub]
    ok = OrdinaryKriging(sl, sa, sv, variogram_model='spherical',
                         verbose=False, enable_plotting=False,
                         coordinates_type='geographic')
    params = np.asarray(ok.variogram_model_parameters, dtype=float)
    print('  验证用观测子集 {:,} 点,变差函数 psill={:.3f} range={:.3f} nugget={:.3f}'
          .format(len(sv), *params))

    tlon = rng.uniform(-180, 180, n_test)
    tlat = rng.uniform(-56, 78, n_test)

    t0 = time.time()
    zk, ss = ok.execute('points', tlon, tlat, backend='loop')
    sk = np.sqrt(np.maximum(np.asarray(ss).ravel(), 0.0))
    t_pk = time.time() - t0

    t0 = time.time()
    lu_piv, lam, n = build_system(sl, sa, sv, params)
    zm, sm = krige_points(tlon, tlat, sl, sa, params, lu_piv, lam, n)
    t_mw = time.time() - t0

    dz = float(np.max(np.abs(zm - np.asarray(zk).ravel())))
    ds = float(np.max(np.abs(sm - sk)))
    print('  预测值最大绝对差 {:.3e}   标准差最大绝对差 {:.3e}'.format(dz, ds))
    print('  PyKrige {:.1f}s / 本实现 {:.2f}s'.format(t_pk, t_mw))
    good = dz < 1e-8 and ds < 1e-8
    print('  {}'.format('✓ 与 PyKrige 数值一致' if good else '✗ 不一致,不可用'))
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true', help='只做一致性验证')
    args = ap.parse_args()

    grid_lon = np.arange(-180 + RES / 2, 180, RES)
    grid_lat = np.arange(-90 + RES / 2, 90, RES)

    pts = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    lon_pts, lat_pts = pts['lon'], pts['lat']
    q_meas, q_corr = pts['q_measured'], pts['q_corrected']
    print('输入测点 {:,}'.format(len(lon_pts)))

    LONg, LATg = np.meshgrid(grid_lon, grid_lat)
    # 只在陆地格点上产出。旧产物在 6 个矩形区块内不分海陆一律出值,
    # 149,063 个有效格点里有一半以上是海洋 —— 那里的值是从陆地测点外推的,
    # 没有物理意义,却被 emit_numbers.py 算进了"大陆平均通量"。
    # 名实不符,这次一并改正;顺带把预测点数从 216,000 降到约 67,000。
    import cartopy.feature as cfeature
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    land = rasterize([(g, 1) for g in cfeature.LAND.geometries()],
                     out_shape=LONg.shape,
                     transform=from_origin(-180, 90, RES, RES),
                     fill=0, all_touched=True).astype(bool)[::-1]
    keep = (LATg >= LAT_FLOOR).ravel() & land.ravel()
    plon, plat = LONg.ravel()[keep], LATg.ravel()[keep]
    print('预测格点 {:,}(陆地,{:+.0f}° 以北)'.format(len(plon), LAT_FLOOR))

    fields = {}
    for name, vals in (('qm', q_meas), ('dq', q_corr - q_meas)):
        blon, blat, bval = bin_medians(lon_pts, lat_pts, vals, grid_lon, grid_lat)
        print('\n=== {} ===\n  分箱观测格点 {:,}'.format(name, len(bval)))

        if args.verify:
            if not verify(blon, blat, bval):
                sys.exit(1)
            continue

        _, params = fit_variogram(blon, blat, bval)
        print('  球状变差函数 psill={:.3f} range={:.3f}° nugget={:.3f}'.format(*params))
        t0 = time.time()
        lu_piv, lam, n = build_system(blon, blat, bval, params)
        print('  矩阵 {}×{} 组装并分解完毕 {:.0f}s'.format(n + 1, n + 1, time.time() - t0))
        t0 = time.time()
        zz, ss = krige_points(plon, plat, blon, blat, params, lu_piv, lam, n,
                              progress=True)
        print('  预测 {:,} 格点用时 {:.0f}s'.format(len(plon), time.time() - t0))
        del lu_piv

        z = np.full(LONg.size, np.nan)
        sd = np.full(LONg.size, np.nan)
        z[keep], sd[keep] = zz, ss
        fields[name] = (z.reshape(LONg.shape).astype(np.float32),
                        sd.reshape(LONg.shape).astype(np.float32))

    if args.verify:
        print('\n✓ 两个场都与 PyKrige 一致')
        return

    krige_qm, std_qm = fields['qm']
    krige_dq, std_dq = fields['dq']
    krige_qc = krige_qm + krige_dq        # 与旧脚本同:qc 由 qm+dq 重建,保证 b-a≡c
    std_qc = np.sqrt(std_qm ** 2 + std_dq ** 2)

    out = os.path.join(HERE, 'kriged_heatflow_05deg_global.npz')
    np.savez_compressed(out, krige_qm=krige_qm, krige_qc=krige_qc,
                        krige_dq=krige_dq, std_qm=std_qm, std_qc=std_qc,
                        std_dq=std_dq, lons=grid_lon, lats=grid_lat)
    print('\n写出 {}'.format(os.path.basename(out)))
    for k, v in (('qm', krige_qm), ('qc', krige_qc), ('dq', krige_dq)):
        f = v[np.isfinite(v)]
        print('  {:<3} [{:7.2f}, {:7.2f}] mean {:6.2f}'.format(k, f.min(), f.max(), f.mean()))
    print('  一致性 max|(qc-qm)-dq| = {:.2e}'.format(
        np.nanmax(np.abs((krige_qc - krige_qm) - krige_dq))))


if __name__ == '__main__':
    main()
