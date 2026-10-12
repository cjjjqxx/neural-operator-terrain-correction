# -*- coding: utf-8 -*-
"""邻近配对检验的稳健口径置换对照(审稿 R1-M3)。

"""
import os
import sys
import json

import numpy as np
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
RADIUS_KM, N_PERM, SEED = 10.0, 400, 0
EDGES = [0, 0.02, 0.05, 0.10, 0.20, np.inf]
LABELS = ['< 0.02', '0.02-0.05', '0.05-0.10', '0.10-0.20', '> 0.20']


def mad(x):
    return float(np.median(np.abs(x - np.median(x))))


def main():
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    lon, lat, qm, B = d['lon'], d['lat'], d['q_measured'], d['B']
    r = np.radians
    xyz = np.column_stack([np.cos(r(lat)) * np.cos(r(lon)), np.cos(r(lat)) * np.sin(r(lon)),
                           np.sin(r(lat))]) * 6371.0
    i, j = cKDTree(xyz).query_pairs(RADIUS_KM, output_type='ndarray').T
    l0 = np.log(qm[i] / qm[j])
    ab = np.abs(B - 1.0)
    band = np.digitize(np.minimum(ab[i], ab[j]), EDGES[1:-1])
    Bi, Bj = B[i], B[j]

    def red(bi, bj, m):
        return 100.0 * (1 - mad(l0[m] - np.log(bi[m]) + np.log(bj[m])) / mad(l0[m]))

    rng = np.random.default_rng(SEED)
    srcs = []
    for _ in range(N_PERM):
        src = np.empty(len(i), dtype=int)
        for s in range(len(EDGES) - 1):
            k = np.where(band == s)[0]
            if len(k):
                src[k] = k[rng.permutation(len(k))]
        srcs.append(src)
    out = {}
    print('%-10s %7s %9s %9s %9s %8s' % ('档', 'n', '真实', '对照', '优势', 'p'))
    for s, lab in list(enumerate(LABELS)) + [(None, '全部')]:
        m = np.ones(len(i), bool) if s is None else band == s
        real = red(Bi, Bj, m)
        nb = np.array([red(Bi[src], Bj[src], m) for src in srcs])
        p = max(float(np.mean(nb >= real)), 1.0 / (N_PERM + 1))
        out[lab] = dict(n=int(m.sum()), real=real, null=float(nb.mean()), margin=float(real - nb.mean()),
                        null_lo=float(np.percentile(nb, 2.5)), null_hi=float(np.percentile(nb, 97.5)), p=p)
        print('%-10s %7d %+9.2f %+9.2f %+9.2f %8.4f' % (lab, m.sum(), real, nb.mean(), real - nb.mean(), p))
    json.dump(out, open(os.path.join(HERE, 'paired_null_robust_test.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
