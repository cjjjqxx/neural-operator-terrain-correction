# -*- coding: utf-8 -*-
"""C15 原研究者现场判断检验 —— 与建模无关的外部检验之一。

IHFC 的 C15 字段以有序类别记录原研究者对地形影响的判定,由弱到强为
    [Present not significant] → [Present and not corrected] → [Present and corrected];
[not recognized] 表示未评估(信息缺失),不在有序尺度内,单独作对照。该信息从未进入算子训练。

做法:
  - 同一位置(坐标取 4 位小数)的记录若带有不同的有序类别,无法判定原研究者的判断,整点剔除
    (location_categories);
  - 第一级(判定存在但不显著 → 识别出但未校正)以单侧 Mann–Whitney 检验比较算子 |B−1|,
    相邻类别给 Cliff's δ;另报三类的 Jonckheere–Terpstra 趋势检验(正态近似)作参考。
  - [Present and corrected] 的测点不进入交付产品,本检验读 rebuild_correction.py
    --keep-topo-corrected 写出的 rebuild_points_with_topocorr.npz。
输出 c15_ordinal_test.json。
"""
import os
import sys
import json
import collections

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy import stats

# ── IHFC 数据库路径 ──────────────────────────────────────────────────
# 不写死本地路径:优先取环境变量 IHFC_GHFDB,否则找本脚本同级的 data/。
# 数据本身从 GFZ Data Services 公开获取(doi:10.5880/fidgeo.2024.014)。
def _ihfc_path():
    p = os.environ.get('IHFC_GHFDB')
    if p and os.path.exists(p):
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, '..', 'IHFC_2024_full', 'IHFC_2024_GHFDB_v.2026.03.txt'),
                 os.path.join(here, 'data', 'IHFC_2024_GHFDB_v.2026.03.txt'),
                 os.path.join(here, 'IHFC_2024_GHFDB_v.2026.03.txt')):
        if os.path.exists(cand):
            return cand
    raise SystemExit(
        'IHFC database not found.\n'
        '  Set the environment variable IHFC_GHFDB to the file path, or place\n'
        '  IHFC_2024_GHFDB_v.2026.03.txt under ./data/ next to this script.\n'
        '  The dataset is openly available from GFZ Data Services, '
        'doi:10.5880/fidgeo.2024.014')



HERE = os.path.dirname(os.path.abspath(__file__))
IHFC = _ihfc_path()
C15_COL, LAT_COL, LON_COL, PID_COL = 27, 3, 4, 65

ORDER = ['[Present not significant]',
         '[Present and not corrected]', '[Present and corrected]']
UNASSESSED = '[not recognized]'
SHORT = {'[not recognized]': '未评估(对照)', '[Present not significant]': '存在但不显著',
         '[Present and not corrected]': '识别未校正', '[Present and corrected]': '已校正'}


def site_key(lon, lat):
    return (round(lon, 4), round(lat, 4))


def location_categories():
    """每个位置(约 11 m)的 C15 类别,返回 (位置 → 类别, 矛盾位置集合)。

    同一位置的记录若带有不同的有序类别,元数据自相矛盾,无法判定原研究者的判断,整点剔除。
    "未评估"属信息缺失,不与有序类别构成矛盾:位置另有一个有序类别时取该类别。

    """
    cats = collections.defaultdict(set)
    with open(IHFC, encoding='latin-1') as f:
        for _ in range(13):
            next(f)
        for line in f:
            t = line.rstrip('\n').split('\t')
            if len(t) <= PID_COL:
                continue
            c = t[C15_COL].strip()
            if c not in ORDER and c != UNASSESSED:
                continue
            try:
                lat, lon = float(t[LAT_COL]), float(t[LON_COL])
            except ValueError:
                continue
            cats[site_key(lon, lat)].add(c)
    assign, conflict = {}, set()
    for k, cs in cats.items():
        o = [c for c in cs if c in ORDER]
        if len(o) > 1:
            conflict.add(k)
        else:
            assign[k] = o[0] if o else UNASSESSED
    return assign, conflict


def main():
    # ── 1. 从原始库取每个位置的 C15 类别(有序类别相互矛盾的位置整点剔除) ──
    best, conflict = location_categories()
    print(f'原始库中带 C15 标记的测点位置: {len(best) + len(conflict):,};'
          f'有序类别相互矛盾、整点剔除 {len(conflict):,}')

    # ── 2. 与本文校正结果按坐标匹配 ──
    # 优先用验证集(含被排除的 [Present and corrected] 一档);没有则退回交付产品,
    # 此时最强的一档 n=0,趋势检验只能在前两档上做。
    _v = os.path.join(HERE, 'rebuild_points_with_topocorr.npz')
    _p = _v if os.path.exists(_v) else os.path.join(HERE, 'rebuild_points.npz')
    print(f'读取: {os.path.basename(_p)}')
    if _p != _v:
        print('  ⚠ 未找到验证集,[Present and corrected] 一档将为空。'
              '需先跑 rebuild_correction.py --keep-topo-corrected')
    d = np.load(_p)
    lon, lat, B = d['lon'], d['lat'], d['B']
    ab = np.abs(B - 1.0)
    groups = collections.defaultdict(list)
    n_conflict_sites = 0
    for i in range(len(lon)):
        k = site_key(float(lon[i]), float(lat[i]))
        if k in conflict:
            n_conflict_sites += 1
            continue
        c = best.get(k)
        if c is not None:
            groups[c].append(float(ab[i]))
    print(f'匹配到的测点中因类别矛盾剔除 {n_conflict_sites:,} 个')
    print()
    print(f'{"类别":<14}{"n":>7}{"|B-1| 中位":>12}{"均值":>10}{">5% 占比":>10}')
    print('-' * 54)
    rows, samples = [], []
    for c in ORDER + [UNASSESSED]:
        v = np.array(groups.get(c, []))
        if len(v) == 0:
            print(f'{SHORT[c]:<14}{0:>7}   (匹配为空)')
            samples.append(v)
            continue
        # |B-1| > 0.05 的占比(不等同于校正量超过 5%:相对校正量为 |1/B − 1|)
        frac = 100 * float((v > 0.05).mean())
        print(f'{SHORT[c]:<14}{len(v):>7}{np.median(v):>12.4f}'
              f'{v.mean():>10.4f}{frac:>9.1f}%')
        rows.append(dict(category=c, label=SHORT[c], n=int(len(v)),
                         in_ordinal_scale=(c != UNASSESSED),
                         median_absB1=round(float(np.median(v)), 5),
                         mean_absB1=round(float(v.mean()), 5),
                         frac_gt5pct=round(frac, 2)))
        if c != UNASSESSED:
            samples.append(v)

    # ── 3. Jonckheere-Terpstra 有序趋势检验(正态近似)──
    使用 = [s for s in samples if len(s) > 0]
    if len(使用) >= 2:
        JT = 0
        for a in range(len(使用) - 1):
            for b in range(a + 1, len(使用)):
                x, y = 使用[a], 使用[b]
                # U 统计量:y 大于 x 的对数
                u = stats.mannwhitneyu(y, x, alternative='greater')
                JT += u.statistic
        ns = np.array([len(s) for s in 使用], float)
        N = ns.sum()
        mu = (N ** 2 - (ns ** 2).sum()) / 4.0
        sd = np.sqrt((N ** 2 * (2 * N + 3) - (ns ** 2 * (2 * ns + 3)).sum()) / 72.0)
        z = (JT - mu) / sd
        p = stats.norm.sf(z)          # 单侧:递增趋势
        print()
        print(f'Jonckheere-Terpstra 有序趋势检验(备择:|B-1| 随类别递增)')
        print(f'  z = {z:.3f}   单侧 p = {p:.3e}')
    else:
        z = p = float('nan')
        print('\n可用组数不足,跳过趋势检验')

    # ── 4. 相邻类别的 Cliff's delta ──
    print()
    print('相邻类别 Cliff\'s delta(正值表示后者更大):')
    deltas = []
    for a in range(len(ORDER) - 1):
        x, y = np.array(groups.get(ORDER[a], [])), np.array(groups.get(ORDER[a + 1], []))
        if len(x) == 0 or len(y) == 0:
            continue
        u = stats.mannwhitneyu(y, x, alternative='two-sided').statistic
        dl = 2 * u / (len(x) * len(y)) - 1
        print(f'  {SHORT[ORDER[a]]} -> {SHORT[ORDER[a+1]]:<14} delta = {dl:+.3f}')
        deltas.append(dict(frm=SHORT[ORDER[a]], to=SHORT[ORDER[a + 1]],
                           cliffs_delta=round(float(dl), 4)))

    # 第一级(判定存在但不显著 → 识别出但未校正)的单侧 Mann–Whitney 检验:正文与摘要所报的 p
    x0, x1 = np.array(groups.get(ORDER[0], [])), np.array(groups.get(ORDER[1], []))
    p_step1 = float(stats.mannwhitneyu(x1, x0, alternative='greater').pvalue)
    print(f'第一级单侧 Mann–Whitney p = {p_step1:.2e}')

    out = dict(note='C15 原研究者现场判断检验;[Present and corrected] 组不进入交付产品,'
                    '仅用于本检验;有序类别相互矛盾的位置整点剔除', groups=rows,
               n_conflict_locations=len(conflict), n_conflict_sites=n_conflict_sites,
               step1_mw_p=float(f'{p_step1:.3g}'),
               jonckheere_z=None if np.isnan(z) else round(float(z), 4),
               jonckheere_p=None if np.isnan(p) else float(f'{p:.3g}'),
               adjacent_deltas=deltas)
    with open(os.path.join(HERE, 'c15_ordinal_test.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print('\n已保存 c15_ordinal_test.json')


if __name__ == '__main__':
    main()
