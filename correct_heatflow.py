# -*- coding: utf-8 -*-
"""给任意一批热流测点做地形校正 —— 随论文发布的主脚本。

    python correct_heatflow.py --sites my_sites.csv --dem SRTM15_V2.6.tif -o out.csv

输入 CSV 至少要有经度、纬度、实测热流三列(列名不区分大小写,常见写法都认:
lon/longitude/x、lat/latitude/y、q/heat_flow/hf)。有钻孔深度就再给一列
depth/depth_m/q_bottom,单位米 —— 没有也能跑,缺失的按 338 m(本文汇编中
有深度记录测点的中位深度)处理。

输出在原表后面追加四列:
    B_surface     地表传递因子,算子的直接输出
    B_effective   按钻孔深度做过深度平均的传递因子,实际用于校正的就是它
    q_corrected   校正后的热流 = q / B_effective
    relief_m      该测点 57 km 窗口内的地形起伏,供判断校正是否可信

高程数据不随本脚本分发(全球一张图约 7 GB)。使用者自备任一高程栅格即可:
坐标系不限,非 EPSG:4326 的会被自动重投影(WarpedVRT + 双线性);
像元不粗于约 450 m —— 更粗的高程会系统性抹平地形、把校正压向 1。
获取途径见 fetch_dem.py。

**重投影不是免费的**:它意味着高程被重采样两次,短波成分会被磨掉一些。
在阿尔卑斯一带 30 个测点上实测,与直接用经纬度栅格相比,相对差中位 2.9%、
P90 9.6%、最大 16.3%,且校正量越大差异越大。能拿到原生经纬度栅格就用它。

为什么是这个形式
----------------
地形校正的物理早已成熟,难点一直在于每做一个测点就要建一次三维模型。
本脚本把这一步换成一次前向推理:使用者不需要有限元环境,也不需要自己
决定网格与边界条件,校正的口径与本文汇编中 36,392 个测点完全一致。
"""
import argparse
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ── 深度平均 ────────────────────────────────────────────────────────────
# 这几个常量与 rebuild_correction.py 中的同名常量必须逐字一致 —— 一旦漂移,
# 使用者算出的校正就与论文报告的不是同一个量。该一致性由
# check_pipeline_consistency.py 机械核对,不靠人记得同步。
_F_DEPTH_M = np.array([0.0, 100.0, 146.0, 250.0, 390.0, 500.0, 1000.0, 1450.0, 2000.0])
_F_DEPTH_V = np.array([1.000, 0.987, 0.970, 0.929, 0.873, 0.833, 0.677, 0.574, 0.476])
MEDIAN_DEPTH_M = 338.0

LON_KEYS = ('lon', 'longitude', 'long', 'x', 'lon_deg')
LAT_KEYS = ('lat', 'latitude', 'y', 'lat_deg')
Q_KEYS = ('q', 'heat_flow', 'heatflow', 'hf', 'q_obs', 'q_measured', 'qs')
D_KEYS = ('depth', 'depth_m', 'q_bottom', 'borehole_depth', 'bottom_depth')

MAX_PIXEL_DEG = 0.0056      # 约 620 m —— 比 SRTM15+ 的 464 m 留一点余量


def f_depth(depth_m):
    """深度 D 的钻孔保留下来的地表偏离比例 f(D) = (B_eff − 1)/(B_surf − 1)。"""
    if depth_m is None or not np.isfinite(depth_m) or depth_m <= 0:
        depth_m = MEDIAN_DEPTH_M
    return float(np.interp(depth_m, _F_DEPTH_M, _F_DEPTH_V))


def center_above_min_m(dem):
    """测点所在中心列的地表高出瓦片最低点多少米(与 fem3d.build 同一口径:
    节点高程取 128→41 线性缩放后的中心节点,z_min 取原瓦片最小值)。"""
    from scipy.ndimage import zoom
    d = np.asarray(dem, dtype=np.float64)
    dc = zoom(d, (41 / d.shape[0], 41 / d.shape[1]), order=1)
    return float(dc[20, 20] - d.min())


def implicit_depth_m(relief_m, center_above_min=0.0):
    """生产有限元解在测点所在中心列隐含的平均深度 = 该列顶层单元厚度。

    """
    return (max(3000.0, 1.5 * float(relief_m)) + float(center_above_min)) / 10.0


def depth_factor(borehole_depth_m, relief_m, center_above_min=0.0):
    """f(D) / f(implicit) —— 只补增量,不重复扣减浅部。"""
    return f_depth(borehole_depth_m) / f_depth(implicit_depth_m(relief_m, center_above_min))


def _pick(cols, keys, what):
    low = {c.strip().lower(): c for c in cols}
    for k in keys:
        if k in low:
            return low[k]
    raise SystemExit(
        '× 输入表里找不到%s列。认得的写法:%s\n  实际列名:%s'
        % (what, '、'.join(keys), '、'.join(cols)))


def main():
    ap = argparse.ArgumentParser(
        description='用训练好的神经算子给热流测点做地形校正',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('为什么是这个形式')[0].split('\n', 2)[2])
    ap.add_argument('--sites', required=True, help='测点 CSV')
    ap.add_argument('--dem', required=True,
                    help='高程栅格,任意坐标系,像元不粗于约 450 m')
    ap.add_argument('-o', '--out', default='corrected.csv', help='输出 CSV')
    ap.add_argument('--model', default=os.path.join(HERE, 'fno3d_large_model.pt'))
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--device', default='auto', choices=('auto', 'cuda', 'cpu'))
    a = ap.parse_args()

    import pandas as pd
    import rasterio
    import torch
    from tile_reader import read_tile
    from step3_train_large import FNO3D

    df = pd.read_csv(a.sites)
    c_lon = _pick(df.columns, LON_KEYS, '经度')
    c_lat = _pick(df.columns, LAT_KEYS, '纬度')
    c_q = _pick(df.columns, Q_KEYS, '热流')
    c_d = next((df.columns[list(map(str.lower, map(str.strip, df.columns))).index(k)]
                for k in D_KEYS
                if k in [str(x).strip().lower() for x in df.columns]), None)
    print('列映射:经度=%s 纬度=%s 热流=%s 深度=%s'
          % (c_lon, c_lat, c_q, c_d if c_d else '(无,按 %.0f m 处理)' % MEDIAN_DEPTH_M))

    _src = rasterio.open(a.dem)
    if _src.crs is None:
        raise SystemExit('× 高程栅格没有坐标系信息,无法定位测点')
    if _src.crs.to_epsg() == 4326:
        ds = _src
    else:
        # 非经纬度栅格:包一层 WarpedVRT,对下游完全透明。
        # 高程是连续场,重采样取双线性 —— 最近邻会在瓦片里留下阶梯,
        # 而阶梯会被求解器当作真实地形读进去。
        from rasterio.vrt import WarpedVRT
        from rasterio.warp import Resampling
        print('高程栅格为 %s,自动重投影到 EPSG:4326' % _src.crs)
        print('  ! 重投影会引入额外的重采样误差:在阿尔卑斯一带 30 个测点上实测,')
        print('    相对差中位 2.9%%、P90 9.6%%、最大 16.3%%,且校正量越大差异越大。')
        print('    若能拿到原生经纬度栅格,优先用它。')
        ds = WarpedVRT(_src, crs='EPSG:4326', resampling=Resampling.bilinear)
    px = abs(ds.transform.a)
    if px > MAX_PIXEL_DEG:
        raise SystemExit(
            '× 高程像元 %.5f° (约 %.0f m) 过粗,上限约 %.0f m。\n'
            '  更粗的高程会抹平短波地形,把校正系统性压向 1 —— 结果会看起来\n'
            '  "校正很小",而那是分辨率造成的,不是地形造成的。'
            % (px, px * 111320, MAX_PIXEL_DEG * 111320))

    dev = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    ck = torch.load(a.model, map_location=dev, weights_only=False)
    FN = 40
    model = FNO3D(modes=24, width=96, H_out=FN + 1, W_out=FN + 1, n_layers=6).to(dev).eval()
    model.load_state_dict(ck['state'])
    from step3_train_large import D4Average
    model = D4Average(model).eval()      # 与生产流水线一致:8 个朝向取平均
    X_m, X_s, Y_m, Y_s = ck['X_mean'], ck['X_std'], ck['Y_mean'], ck['Y_std']
    c = FN // 2
    print('模型已载入(%s);高程 %s,像元 %.5f° (约 %.0f m)'
          % (dev, os.path.basename(a.dem), px, px * 111320))

    n = len(df)
    B_surf = np.full(n, np.nan)
    relief = np.full(n, np.nan)
    above = np.zeros(n)
    buf_dem, buf_i = [], []

    def flush():
        if not buf_dem:
            return
        xb = (np.stack(buf_dem).reshape(-1, 1, 128, 128) - X_m) / X_s
        with torch.no_grad():
            pred = model(torch.tensor(xb, dtype=torch.float32).to(dev))
        Bv = pred.cpu().numpy()[:, 0, c, c] * Y_s + Y_m
        for k, i in enumerate(buf_i):
            B_surf[i] = Bv[k]
            relief[i] = buf_dem[k].max() - buf_dem[k].min()
            above[i] = center_above_min_m(buf_dem[k])
        buf_dem.clear()
        buf_i.clear()

    for i, (lon, lat) in enumerate(zip(df[c_lon].values, df[c_lat].values)):
        try:
            dem = read_tile(ds, float(lon), float(lat))
        except Exception:
            dem = None
        if dem is not None:
            buf_dem.append(dem.astype(np.float32))
            buf_i.append(i)
        if len(buf_dem) >= a.batch:
            flush()
        if (i + 1) % 2000 == 0:
            print('  %s/%s' % (format(i + 1, ','), format(n, ',')))
    flush()
    ds.close()
    if ds is not _src:
        _src.close()

    depth = df[c_d].values if c_d else np.full(n, np.nan)
    B_eff = np.full(n, np.nan)
    ok = np.isfinite(B_surf)
    for i in np.nonzero(ok)[0]:
        B_eff[i] = 1.0 + depth_factor(depth[i], relief[i], above[i]) * (B_surf[i] - 1.0)

    out = df.copy()
    out['B_surface'] = B_surf
    out['B_effective'] = B_eff
    out['q_corrected'] = df[c_q].values / B_eff
    out['relief_m'] = relief
    out.to_csv(a.out, index=False, encoding='utf-8')

    nb = int((~ok).sum())
    print()
    print('完成 %s / %s 个测点%s'
          % (format(int(ok.sum()), ','), format(n, ','),
             '' if nb == 0 else ',%d 个落在高程数据覆盖之外(输出为空)' % nb))
    if ok.any():
        d = np.abs(B_eff[ok] - 1.0)
        print('  |B − 1| 中位 %.4f;超过 2%% 的 %.1f%%,超过 10%% 的 %.1f%%'
              % (np.median(d), 100 * (d > 0.02).mean(), 100 * (d > 0.10).mean()))
    print('已写出 %s' % a.out)


if __name__ == '__main__':
    main()
