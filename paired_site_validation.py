# -*- coding: utf-8 -*-
"""
paired_site_validation.py -- 用真实实测数据检验地形校正(独立于本文自己的有限元解)

思路
----
互距很近(<=10 km)的两个测点,其深部热流可近似认为相同,因此实测热流之差里
应当含有一个由两点地形差异造成的成分。若校正真实有效,扣除该成分后配对散度
应当下降。

为什么不能只看"校正后散度有没有下降"
------------------------------------
校正是乘性的(q_corrected = q_measured / B)。当邻近两点共享同一种地形特征时
(实测:|B-1|>0.1 的配对里 84% 同为山脊或同为谷地,两点 B 仅相差约 3%),
除以相近的 B 会把两点差异按 1/B 放大。这是物理上正确的行为 —— 两点若被地形
同等压低,其真实深部热流之差本就大于实测之差 —— 但它意味着"校正应让邻近测点
更一致"这一直觉判据在共享地形时并不成立。

因此判据取"真实校正 vs 量级匹配的随机对照"。对照用分层置换构造:仅在相同
|B-1| 档位内部打乱 B,使随机对照的校正量级与真实情形一致。
(若在全体测点间打乱,则该分布由 83% 的近平坦测点主导、中位校正仅 0.6%,
高起伏档会变成"大校正 vs 小校正"的不对等比较。)
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
RADIUS_KM = 10.0
N_PERM = 400
EDGES = [0, 0.02, 0.05, 0.10, 0.20, np.inf]
SEED = 0


def main():
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    lon, lat, qm, B = d['lon'], d['lat'], d['q_measured'], d['B']
    n_sites = len(qm)

    # 球面坐标下取近邻配对
    r = np.radians
    xyz = np.column_stack([np.cos(r(lat)) * np.cos(r(lon)),
                           np.cos(r(lat)) * np.sin(r(lon)),
                           np.sin(r(lat))]) * 6371.0
    i, j = cKDTree(xyz).query_pairs(RADIUS_KM, output_type='ndarray').T
    dq = qm[i] - qm[j]
    s0_all = dq.std()

    absB1 = np.abs(B - 1.0)
    strata = np.digitize(absB1, EDGES[1:-1])       # 每个测点所属 |B-1| 档位
    pair_band = np.digitize(np.minimum(absB1[i], absB1[j]), EDGES[1:-1])

    def reduction(Bv, mask):
        """给定 B 场,返回该配对子集的散度下降百分比"""
        c = qm / Bv - qm
        dr = dq[mask]
        return 100.0 * (1 - (dr + (c[i[mask]] - c[j[mask]])).std() / dr.std())

    rng = np.random.default_rng(SEED)
    perm_B = []
    for _ in range(N_PERM):
        Bp = B.copy()
        for s in range(len(EDGES) - 1):            # 逐档位内部置换,保持量级
            idx = np.where(strata == s)[0]
            Bp[idx] = B[idx][rng.permutation(len(idx))]
        perm_B.append(Bp)

    print(f'测点 {n_sites:,}  |  {RADIUS_KM:.0f} km 内配对 {len(i):,}  |  置换 {N_PERM} 次\n')
    print(f'{"|B-1| 档位":<14}{"配对数":>8}{"真实":>9}{"随机对照":>10}'
          f'{"对照95%区间":>20}{"真实-对照":>11}')

    out = {}
    labels = ['< 0.02', '0.02-0.05', '0.05-0.10', '0.10-0.20', '> 0.20']
    for s, lab in enumerate(labels):
        m = pair_band == s
        if m.sum() < 80:
            print(f'{lab:<14}{m.sum():>8}   样本不足')
            continue
        real = reduction(B, m)
        null = np.array([reduction(Bp, m) for Bp in perm_B])
        out[lab] = dict(n=int(m.sum()), real=float(real),
                        null_mean=float(null.mean()),
                        null_lo=float(np.percentile(null, 2.5)),
                        null_hi=float(np.percentile(null, 97.5)),
                        margin=float(real - null.mean()))
        print(f'{lab:<14}{m.sum():>8}{real:>+9.2f}{null.mean():>+10.2f}'
              f'{f"{np.percentile(null,2.5):+.2f} 到 {np.percentile(null,97.5):+.2f}":>20}'
              f'{real - null.mean():>+11.2f}')

    m = np.ones(len(i), bool)
    real = reduction(B, m)
    null = np.array([reduction(Bp, m) for Bp in perm_B])
    p = max((null >= real).mean(), 1.0 / N_PERM)
    out['all'] = dict(n=int(len(i)), real=float(real), null_mean=float(null.mean()),
                      null_lo=float(np.percentile(null, 2.5)),
                      null_hi=float(np.percentile(null, 97.5)),
                      margin=float(real - null.mean()), p=float(p))
    print(f'{"全部":<14}{len(i):>8}{real:>+9.2f}{null.mean():>+10.2f}'
          f'{f"{np.percentile(null,2.5):+.2f} 到 {np.percentile(null,97.5):+.2f}":>20}'
          f'{real - null.mean():>+11.2f}')
    print(f'\n单侧 p = {"< " + str(round(1.0 / N_PERM, 4)) if p <= 1.0/N_PERM else round(p,4)}'
          f'  (真实值超过全部 {N_PERM} 次随机对照)' if p <= 1.0 / N_PERM else f'\n单侧 p = {p:.4f}')

    # 佐证:地形信号在实测数据里本身可见 —— B 较大的点实测通量应较大
    dB = B[i] - B[j]
    print('\n佐证 corr(Δq_measured, ΔB) 逐档(为正说明地形信号在实测数据中可见):')
    for s, lab in enumerate(labels):
        m = pair_band == s
        if m.sum() < 80:
            continue
        print(f'   {lab:<12}{np.corrcoef(dq[m], dB[m])[0, 1]:+.3f}')

    json.dump(out, open(os.path.join(HERE, 'paired_site_validation.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('\n已保存 paired_site_validation.json')


if __name__ == '__main__':
    main()
