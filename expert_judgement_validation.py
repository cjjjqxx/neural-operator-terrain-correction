# -*- coding: utf-8 -*-
"""用原始研究者自己的地形效应判断,独立检验算子的预测。

IHFC GHFDB 的 C15 字段 corr_TOPO_flag 记录了每条测量的原始研究者对地形效应的
判断。这些人到过现场、了解当地地形,他们的判断与本文的算子完全独立 —— 算子只
看 DEM,从未接触过这个字段。

检验设计
--------
主对比只用两组,二者的采纳通量**都未经地形校正**,口径一致:
    [Present not significant]    研究者认为地形效应存在但不足以校正
    [Present and not corrected]  研究者识别出地形效应但未施加校正
若算子有效,后者的 |B-1| 应显著大于前者。

("Present and corrected" 的测点已在数据构建阶段整点排除,因其采纳通量已含地形
校正,再除一次会重复扣除;它们不参与本检验。)
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get('IHFC_GHFDB', os.path.join('data', 'IHFC_2024_GHFDB_v.2026.03.txt'))
NS, NC = '[Present not significant]', '[Present and not corrected]'


def main():
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    ours = {(round(float(a), 4), round(float(b), 4)): abs(float(c) - 1.0)
            for a, b, c in zip(d['lon'], d['lat'], d['B'])}

    # 每个坐标收集其全部子记录的标记;标记不一致的测点整点弃用,避免口径含混
    flags = {}
    for l in open(DB, encoding='latin-1').read().split('\n')[13:]:
        p = l.strip().split('\t')
        if len(p) < 66:
            continue
        try:
            k = (round(float(p[4]), 4), round(float(p[3]), 4))
        except (ValueError, IndexError):
            continue
        if k in ours:
            flags.setdefault(k, set()).add(p[27].strip())

    groups, mixed = {NS: [], NC: []}, 0
    for k, fs in flags.items():
        rel = fs & {NS, NC}
        if len(rel) == 1:
            groups[rel.pop()].append(ours[k])
        elif len(rel) == 2:
            mixed += 1
    ns, nc = np.array(groups[NS]), np.array(groups[NC])
    print(f'匹配测点 {len(flags):,}  |  标记自相矛盾而弃用 {mixed:,}\n')

    print(f'{"原始研究者的现场判断":<26}{"n":>7}{"中位|B-1|":>11}{"P90":>9}{">5%":>8}{">10%":>8}')
    for lab, v in [('地形效应存在但不显著', ns), ('识别出地形效应但未校正', nc)]:
        print(f'{lab:<26}{len(v):>7}{np.median(v):>11.4f}{np.percentile(v, 90):>9.4f}'
              f'{100 * (v > .05).mean():>7.1f}%{100 * (v > .10).mean():>7.1f}%')

    u, p = mannwhitneyu(nc, ns, alternative='greater')
    # Cliff's delta:非参数效应量,读作"随机取一对,后者更大的净概率"
    delta = 2 * u / (len(nc) * len(ns)) - 1
    print(f'\nMann-Whitney 单侧 p = {p:.2e}   Cliff\'s delta = {delta:+.3f}')
    print(f'中位数比值 = {np.median(nc) / np.median(ns):.2f}×')

    json.dump(dict(n_not_significant=len(ns), n_not_corrected=len(nc), n_mixed=mixed,
                   median_ns=float(np.median(ns)), median_nc=float(np.median(nc)),
                   frac_gt5_ns=float(100 * (ns > .05).mean()),
                   frac_gt5_nc=float(100 * (nc > .05).mean()),
                   p=float(p), cliffs_delta=float(delta)),
              open(os.path.join(HERE, 'expert_judgement_validation.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('\n已保存 expert_judgement_validation.json')


if __name__ == '__main__':
    main()
