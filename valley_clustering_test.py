# -*- coding: utf-8 -*-
"""谷地过度代表性:计入测点空间聚集后的显著性(B5b)。


做法:**空间块自助法**。把测点按经纬度网格分块,以块为单位有放回重采样,
每次重算 B>1 的占比,由此得到考虑了块内相关性的抽样分布。
再与随机陆地参照(2,000 个瓦片,52.05%)比较。参照本身也带抽样误差,一并计入。

块尺度是关键参数,故扫描多个尺度并全部报告,而不是只挑一个。
"""
import os
import sys
import json

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 4000
SEED = 20260825


def block_bootstrap(lon, lat, gt1, deg, rng, n_boot=N_BOOT):
    """按 deg x deg 经纬度块有放回重采样,返回 B>1 占比的自助分布。"""
    key = (np.floor(lon / deg).astype(np.int64) * 100000
           + np.floor(lat / deg).astype(np.int64))
    uniq, inv = np.unique(key, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    counts = np.bincount(inv, minlength=len(uniq))
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    gt1_sorted = gt1[order]

    out = np.empty(n_boot)
    nb = len(uniq)
    for b in range(n_boot):
        pick = rng.integers(0, nb, nb)
        num = den = 0
        for j in pick:
            s, c = starts[j], counts[j]
            num += gt1_sorted[s:s + c].sum()
            den += c
        out[b] = num / den
    return out, nb


def main():
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    lon, lat, B = d['lon'], d['lat'], d['B']
    gt1 = (B > 1).astype(np.float64)
    p1 = gt1.mean()

    base = json.load(open(os.path.join(HERE, 'random_land_baseline.json'),
                          encoding='utf-8'))
    p0, n0 = base['frac_B_gt1'] / 100.0, base['n']
    se0 = np.sqrt(p0 * (1 - p0) / n0)

    rng = np.random.default_rng(SEED)
    print(f'测点 {len(B):,},B>1 占比 {100*p1:.2f}%')
    print(f'随机陆地参照 {n0:,} 瓦片,{100*p0:.2f}% (SE {100*se0:.2f} pp)\n')

    # 独立性假设下的朴素结果,作为对照
    se_naive = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / len(B))
    z_naive = (p1 - p0) / se_naive
    print(f'{"块尺度":>8}{"块数":>8}{"占比SE":>10}{"合并SE":>10}'
          f'{"超出pp":>9}{"z":>8}{"p":>12}')
    print('-' * 66)
    print(f'{"独立假设":>8}{"—":>8}{100*np.sqrt(p1*(1-p1)/len(B)):>9.2f}'
          f'{100*se_naive:>10.2f}{100*(p1-p0):>9.2f}{z_naive:>8.2f}'
          f'{2*stats.norm.sf(abs(z_naive)):>12.2e}')

    rows = []
    for deg in (0.5, 1.0, 2.0, 5.0):
        boot, nb = block_bootstrap(lon, lat, gt1, deg, rng)
        se1 = boot.std(ddof=1)
        se = np.sqrt(se0 ** 2 + se1 ** 2)
        z = (p1 - p0) / se
        p = 2 * stats.norm.sf(abs(z))
        print(f'{deg:>7.1f}°{nb:>8,}{100*se1:>9.2f}{100*se:>10.2f}'
              f'{100*(p1-p0):>9.2f}{z:>8.2f}{p:>12.2e}')
        rows.append(dict(block_deg=deg, n_blocks=int(nb),
                         se_frac_pp=round(100 * se1, 3),
                         se_combined_pp=round(100 * se, 3),
                         excess_pp=round(100 * (p1 - p0), 2),
                         z=round(float(z), 3), p=float(f'{p:.3g}')))

    out = dict(n_sites=int(len(B)), frac_B_gt1_pct=round(100 * p1, 2),
               baseline_pct=round(100 * p0, 2), baseline_n=int(n0),
               excess_pp=round(100 * (p1 - p0), 2),
               p_independent=float(f'{2*stats.norm.sf(abs(z_naive)):.3g}'),
               blocks=rows, n_boot=N_BOOT, seed=SEED,
               note='块自助法以经纬度块为重采样单位,吸收块内相关;'
                    '参照的抽样误差按独立项并入合并 SE')
    with open(os.path.join(HERE, 'valley_clustering_test.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print('\n已保存 valley_clustering_test.json')


if __name__ == '__main__':
    main()
