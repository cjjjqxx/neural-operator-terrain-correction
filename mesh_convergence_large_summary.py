# -*- coding: utf-8 -*-
"""把 mesh_convergence_large.json(150 测点 × 5 种网格)补写成与 mesh_resolution_test.json 相同的汇总格式,
供误差模型、附表 S1、附图 S2a 与 emit_numbers 直接读取。不重解,只重排。

分档变量:生产配置(n_x = 40)有限元的中心像元 |B − 1| —— 与误差模型「按实际施加的 B 分档」一致,
不再用抽样时的算子预测值(旧试验的做法,附表 S1 表头曾为此加注)。
写入字段:nx_ladder、rows(absB1、band、relief、B)、summary(n、median_absB1、pct_change_40_to_128)。
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, 'mesh_convergence_large.json')
KEYS = [(0.0, 0.02, '0.0-0.02'), (0.02, 0.05, '0.02-0.05'), (0.05, 0.10, '0.05-0.1'),
        (0.10, 0.20, '0.1-0.2'), (0.20, 9.0, '0.2-9.0')]
NX = [40, 64, 96, 128]


def main():
    d = json.load(open(P, encoding='utf-8'))
    relief = np.load(os.path.join(HERE, 'site_terrain.npz'))['relief_m']
    rows = []
    for s in d['sites']:
        if not all(np.isfinite(s['B'].get(str(n), np.nan)) for n in NX):
            continue
        a = abs(s['B']['40'] - 1)
        band = next(k for lo, hi, k in KEYS if lo <= a < hi)
        rows.append(dict(idx=s['idx'], absB1=a, band=band, relief=float(relief[s['idx']]), B=s['B']))
    summ = {}
    for _, _, k in KEYS:
        rr = [r for r in rows if r['band'] == k]
        med = {str(n): float(np.median([abs(r['B'][str(n)] - 1) for r in rr])) for n in NX}
        summ[k] = dict(n=len(rr), median_absB1=med,
                       pct_change_40_to_128=100 * (med['128'] - med['40']) / med['40'])
    d.update(nx_ladder=NX, rows=rows, summary=summ)
    json.dump(d, open(P, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    for k, v in summ.items():
        print('%-10s n=%2d  中位 |B−1| %s  40→128 %+.1f%%' % (
            k, v['n'], ' '.join('%.4f' % v['median_absB1'][str(n)] for n in NX), v['pct_change_40_to_128']))


if __name__ == '__main__':
    main()
