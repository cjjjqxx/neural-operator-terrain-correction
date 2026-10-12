# -*- coding: utf-8 -*-
"""D10:按 IHFC 质量码分层的敏感性检验(替代"是否被评估过"这一无效分组)。


质量码由 U(不确定度)与 M(方法学)两段构成,稿件已解析出来但只用了 U 的有无。
本脚本按实际可靠性等级分层:
  U1/U2(不确定度较小) vs U3/U4(较大)
  M1-M3(方法学较可靠) vs M4/Mx
并对每一层的校正量分布给出自助置信区间,用区间是否重叠取代
"大于抽样噪声"这一无定量支撑的定性断言。
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 2000
SEED = 42


def boot_ci(x, fn, n=N_BOOT, seed=SEED):
    """给定统计量的自助 95% 区间。"""
    rng = np.random.default_rng(seed)
    if len(x) < 8:
        return (np.nan, np.nan)
    vals = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def describe(rel, label, out):
    if len(rel) < 8:
        print(f'  {label:<26} n={len(rel):>6}  样本不足')
        return
    med, p90, p99 = np.median(rel), np.percentile(rel, 90), np.percentile(rel, 99)
    ci_med = boot_ci(rel, np.median)
    ci_p99 = boot_ci(rel, lambda v: np.percentile(v, 99))
    g10 = 100 * (rel > 10).mean()
    out[label] = dict(n=int(len(rel)), median=float(med), median_ci=ci_med,
                      p90=float(p90), p99=float(p99), p99_ci=ci_p99, frac_gt10=float(g10))
    print(f'  {label:<26} n={len(rel):>6}  中位 {med:.3f} [{ci_med[0]:.3f},{ci_med[1]:.3f}]'
          f'   P90 {p90:>5.2f}   P99 {p99:>5.2f} [{ci_p99[0]:.2f},{ci_p99[1]:.2f}]'
          f'   >10% {g10:>5.2f}%')


def main():
    d = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    qm, qc = d['q_measured'], d['q_corrected']
    rel = 100 * np.abs(qc - qm) / qm
    qual = np.array([str(x) for x in d['quality_score']])

    # 质量码形如 "U2.M3" 或 "Ux.M1x";取各段首字母数字
    U = np.array([s.split('.')[0].strip() if '.' in s else s.strip() for s in qual])
    M = np.array([s.split('.')[1].strip() if '.' in s and len(s.split('.')) > 1 else ''
                  for s in qual])

    print(f'测点 {len(rel):,}')
    print(f'U 分布: ' + '  '.join(f'{k}:{int((U==k).sum()):,}'
                                  for k in sorted(set(U))[:8]))
    print(f'M 分布: ' + '  '.join(f'{k}:{int((M==k).sum()):,}'
                                  for k in sorted(set(M))[:8]) + '\n')

    out = {}
    print('=== 按不确定度分数 U 分层 ===')
    describe(rel, '全样本', out)
    describe(rel[np.isin(U, ['U1', 'U2'])], 'U1/U2(不确定度较小)', out)
    describe(rel[np.isin(U, ['U3', 'U4'])], 'U3/U4(不确定度较大)', out)
    describe(rel[np.char.startswith(U, 'Ux')], 'Ux(未评估)', out)

    print('\n=== 按方法学分数 M 分层 ===')
    good_m = np.array([m[:2] in ('M1', 'M2', 'M3') and 'x' not in m for m in M])
    describe(rel[good_m], 'M1-M3(方法学较可靠)', out)
    describe(rel[~good_m & (M != '')], 'M4/Mx(较弱或未评估)', out)

    print('\n=== 判读 ===')
    k1, k2 = 'U1/U2(不确定度较小)', 'U3/U4(不确定度较大)'
    if k1 in out and k2 in out:
        a, b = out[k1]['p99_ci'], out[k2]['p99_ci']
        overlap = not (a[1] < b[0] or b[1] < a[0])
        print(f'  U1/U2 与 U3/U4 的 P99 自助区间{"重叠" if overlap else "不重叠"}')
        print(f'  → {"校正量分布对测量不确定度等级不敏感" if overlap else "两档之间存在实质差异,需在正文说明"}')
    print('  注:该分层回答的是"结果是否依赖测量可靠性";'
          '而"是否被评估过"只反映评估投入的分配,不能回答这个问题。')

    json.dump(out, open(os.path.join(HERE, 'quality_stratified_test.json'), 'w',
                        encoding='utf-8'), indent=2, ensure_ascii=False)
    print('\n已保存 quality_stratified_test.json')


if __name__ == '__main__':
    main()
