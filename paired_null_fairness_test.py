# -*- coding: utf-8 -*-
"""检验配对验证所用零模型是否公平。


本脚本增设一个保留共模关系的零模型:把整对 (B_i, B_j) 从同档位的另一组配对整体
搬过来。这样量级与配对内相干性都与真实一致,被破坏的只是"这一对 B 是否属于这一对
测点"。真实校正若仍显著优于该对照,则原结论成立;若不再显著,则原零模型确实不公平。
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
RADIUS_KM, N_PERM, SEED = 10.0, 400, 0
EDGES = [0, 0.02, 0.05, 0.10, 0.20, np.inf]
LABELS = ['< 0.02', '0.02-0.05', '0.05-0.10', '0.10-0.20', '> 0.20']


def main():
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    lon, lat, qm, B = d['lon'], d['lat'], d['q_measured'], d['B']
    r = np.radians
    xyz = np.column_stack([np.cos(r(lat)) * np.cos(r(lon)),
                           np.cos(r(lat)) * np.sin(r(lon)),
                           np.sin(r(lat))]) * 6371.0
    i, j = cKDTree(xyz).query_pairs(RADIUS_KM, output_type='ndarray').T
    dq = qm[i] - qm[j]
    absB1 = np.abs(B - 1.0)
    band = np.digitize(np.minimum(absB1[i], absB1[j]), EDGES[1:-1])
    strata = np.digitize(absB1, EDGES[1:-1])

    def red(Bi, Bj, m):
        """给定每对两端的 B,返回该子集配对散度的下降百分比"""
        dr = dq[m]
        new = qm[i[m]] / Bi[m] - qm[j[m]] / Bj[m]
        return 100.0 * (1 - new.std() / dr.std())

    rng = np.random.default_rng(SEED)
    Bi_real, Bj_real = B[i], B[j]

    # 零模型 A:原方案,档位内逐测点置换(破坏配对内共模)
    nullA = []
    for _ in range(N_PERM):
        Bp = B.copy()
        for s in range(len(EDGES) - 1):
            idx = np.where(strata == s)[0]
            Bp[idx] = B[idx][rng.permutation(len(idx))]
        nullA.append((Bp[i], Bp[j]))

    # 零模型 B:整对搬运,保留配对内共模关系
    nullB = []
    for _ in range(N_PERM):
        src = np.empty(len(i), dtype=int)
        for s in range(len(EDGES) - 1):
            k = np.where(band == s)[0]
            if len(k):
                src[k] = k[rng.permutation(len(k))]
        nullB.append((Bi_real[src], Bj_real[src]))

    print(f'配对 {len(i):,}  |  置换 {N_PERM} 次\n')
    print(f'{"档位":<12}{"配对数":>8}{"真实":>9}{"对照A":>9}{"margin A":>10}'
          f'{"对照B":>9}{"margin B":>10}{"p(B)":>9}')
    out = {}
    for s, lab in list(enumerate(LABELS)) + [(None, '全部')]:
        m = np.ones(len(i), bool) if s is None else (band == s)
        if m.sum() < 80:
            print(f'{lab:<12}{m.sum():>8}   样本不足')
            continue
        real = red(Bi_real, Bj_real, m)
        a = np.array([red(x, y, m) for x, y in nullA])
        b = np.array([red(x, y, m) for x, y in nullB])
        pB = max((b >= real).mean(), 1.0 / (N_PERM + 1))
        out[lab] = dict(n=int(m.sum()), real=float(real), nullA=float(a.mean()),
                        marginA=float(real - a.mean()), nullB=float(b.mean()),
                        marginB=float(real - b.mean()),
                        nullB_lo=float(np.percentile(b, 2.5)),
                        nullB_hi=float(np.percentile(b, 97.5)), p_B=float(pB))
        print(f'{lab:<12}{m.sum():>8}{real:>+9.2f}{a.mean():>+9.2f}{real - a.mean():>+10.2f}'
              f'{b.mean():>+9.2f}{real - b.mean():>+10.2f}{pB:>9.4f}')

    json.dump(out, open(os.path.join(HERE, 'paired_null_fairness_test.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('\n对照A = 原方案(逐测点置换)   对照B = 整对搬运(保留配对内共模)')
    print('已保存 paired_null_fairness_test.json')


if __name__ == '__main__':
    main()
