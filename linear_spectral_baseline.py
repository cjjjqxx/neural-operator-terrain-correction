# -*- coding: utf-8 -*-
"""线性谱解基线 —— 不经网络、只用线性化物理,能把有限元参考解复现到什么程度。


线性化:以瓦片平均高程为参考面,h 为相对高程。地表温度 T = 0 施加在 z = −h 上,
一阶展开后扰动温度 θ 满足 ∇²θ = 0、θ(z=0) = G·h、底面 z = H 处 ∂θ/∂z = 0(均匀通量),
侧边零通量(对应余弦变换)。于是
    θ̂(k, z) = G·ĥ(k)·cosh(k(H − z)) / cosh(kH)
地表 B_s − 1 = F⁻¹[−k·tanh(kH)·ĥ];在 [0, D] 上平均
    B_D − 1 = F⁻¹[ĥ·(cosh(k(H − D)) − cosh(kH)) / (D·cosh(kH))]
其中 H 为参考面到计算域底面的深度,D 取中心列有限元顶层单元的实际厚度
(有限元的 B 正是顶层两排节点之间的温差商)。

几何与有限元一致:同一锥化后的瓦片、同一 41×41 节点采样(scipy zoom,一阶)、
同一域深 max(3000, 1.5·起伏)。另给一个用 128×128 原分辨率高程的版本作对照。
输出 linear_spectral_baseline.json。
"""
import json
import os
import sys

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import zoom

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tile_reader import CELL_M  # noqa: E402

NX = 40                                  # 生产有限元水平单元数


def linear_B(dem_nodes, L, H, D):
    """dem_nodes: (n, n) 节点高程(侧边零通量 → DCT-I)。返回 (B_s 场, B_D 场)。"""
    n = dem_nodes.shape[0]
    h = dem_nodes - dem_nodes.mean()
    hh = dctn(h, type=1)
    k1 = np.pi * np.arange(n) / L
    k = np.hypot(k1[:, None], k1[None, :])
    kH = np.minimum(k * H, 50.0)
    Bs = 1.0 + idctn(-k * np.tanh(kH) * hh, type=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        ker = (np.cosh(np.minimum(k * (H - D), 50.0)) - np.cosh(kH)) / (D * np.cosh(kH))
    ker[0, 0] = 0.0
    BD = 1.0 + idctn(ker * hh, type=1)
    return Bs, BD


def metrics(p, t):
    e = p - t
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)),
                r2=float(1 - (e ** 2).sum() / ((t - t.mean()) ** 2).sum()),
                sign_agree=float(100 * np.mean(np.sign(p - 1) == np.sign(t - 1))),
                amp_ratio_median=float(np.median(np.abs(p - 1)[np.abs(t - 1) > 0.02] /
                                                 np.abs(t - 1)[np.abs(t - 1) > 0.02]))
                if (np.abs(t - 1) > 0.02).any() else None)


def main():
    v = np.load(os.path.join(HERE, 'validation_independent_500.npz'))
    dems, Ff, Pf, rl = v['dems'].astype(float), v['B_fem'].astype(float), v['B_fno'].astype(float), v['reliefs']
    n = len(dems)
    L = dems.shape[1] * CELL_M
    LinD41 = np.zeros((n, NX + 1, NX + 1))
    LinD128c = np.zeros(n)
    for i, dem in enumerate(dems):
        zmin, zmax = dem.min(), dem.max()
        depth = max(3000.0, 1.5 * (zmax - zmin))
        z_bot = zmin - depth
        nodes = zoom(dem, ((NX + 1) / dem.shape[0], (NX + 1) / dem.shape[1]), order=1)
        H = nodes.mean() - z_bot
        c = NX // 2
        D = (nodes[c, c] - z_bot) / 10.0                        # 中心列顶层单元厚度
        _, LinD41[i] = linear_B(nodes, L, H, D)
        _, bd128 = linear_B(dem, L, dem.mean() - z_bot, D)
        LinD128c[i] = bd128[dem.shape[0] // 2, dem.shape[1] // 2]

    c = NX // 2
    Fc, Pc, Lc = Ff[:, c, c], Pf[:, c, c], LinD41[:, c, c]
    out = dict(n=n, note='中心像元与全场;线性解 = 一阶谱解(有限深度、侧边零通量、顶层厚度平均)')
    out['center'] = dict(operator=metrics(Pc, Fc), linear41=metrics(Lc, Fc), linear128=metrics(LinD128c, Fc))
    out['field'] = dict(operator=metrics(Pf.ravel(), Ff.ravel()), linear41=metrics(LinD41.ravel(), Ff.ravel()))
    ab = np.abs(Fc - 1)
    out['center_by_band'] = {}
    for lo, hi, lab in ((0, .02, '<0.02'), (.02, .05, '0.02-0.05'), (.05, .10, '0.05-0.10'), (.10, 9, '>=0.10')):
        m = (ab >= lo) & (ab < hi)
        if m.sum() < 5:
            continue
        out['center_by_band'][lab] = dict(n=int(m.sum()), operator=metrics(Pc[m], Fc[m]),
                                          linear41=metrics(Lc[m], Fc[m]))
    out['center_by_relief'] = {}
    for lo, hi, lab in ((0, 100, '<100 m'), (100, 500, '100-500 m'), (500, 1e9, '>500 m')):
        m = (rl >= lo) & (rl < hi)
        out['center_by_relief'][lab] = dict(n=int(m.sum()), operator=metrics(Pc[m], Fc[m]),
                                            linear41=metrics(Lc[m], Fc[m]))

    def row(tag, a):
        amp = a['amp_ratio_median']
        return '  %-26s RMSE %.4f  偏差 %+.5f  R² %.3f  符号一致 %.1f%%  幅度比中位 %s' % (
            tag, a['rmse'], a['bias'], a['r2'], a['sign_agree'], '—' if amp is None else '%.2f' % amp)
    print('中心像元(n = %d):' % n)
    for k2, a in out['center'].items():
        print(row(k2, a))
    print('全场 41×41:')
    for k2, a in out['field'].items():
        print(row(k2, a))
    for grp in ('center_by_band', 'center_by_relief'):
        print('\n' + grp + ':')
        for lab, r in out[grp].items():
            print(' %s  n=%d' % (lab, r['n']))
            print(row('算子', r['operator']))
            print(row('线性谱解', r['linear41']))
    # 计时:单线程 CPU,41×41 节点版(含 128→41 重采样),对全部 500 块取中位
    import time
    tt = []
    for dem in dems:
        t0 = time.perf_counter()
        zmin = dem.min(); z_bot = zmin - max(3000.0, 1.5 * (dem.max() - zmin))
        nodes = zoom(dem, ((NX + 1) / dem.shape[0], (NX + 1) / dem.shape[1]), order=1)
        linear_B(nodes, L, nodes.mean() - z_bot, (nodes[NX // 2, NX // 2] - z_bot) / 10.0)
        tt.append(time.perf_counter() - t0)
    out['ms_per_tile_cpu'] = float(1000 * np.median(tt))
    print('线性谱解耗时:每瓦片中位 %.2f ms(单线程 CPU,含重采样)' % out['ms_per_tile_cpu'])
    json.dump(out, open(os.path.join(HERE, 'linear_spectral_baseline.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('\n已保存 linear_spectral_baseline.json')


if __name__ == '__main__':
    main()
