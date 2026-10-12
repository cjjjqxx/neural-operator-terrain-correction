# -*- coding: utf-8 -*-
"""生成两张收敛性附表(Markdown),供 Extended Data Table 2/3 使用。

为什么不用旧的 footprint_depth_expand_results.npz
------------------------------------------------
旧的 10 点压力测试固定 res_px=128、令 cell_m = 窗口宽度/128,因此窗口与网格分辨率
是绑在一起变的:窗口变大纳入更多长波使 |B-1| 增大,同时网格变粗抹掉短波使 |B-1|
减小,两个方向相反的效应无法分离。它报出的 43% 变化里,绝大部分是这一设计缺陷。

本脚本改用两组把二者分开的干净测试:
  Table 2  窗口收敛  —— 单元尺寸锁死 1,425 m,只放大窗口(footprint_fixed_resolution_test
           + footprint_narrow_test,两次运行拼成 20-160 km)
  Table 3  网格收敛  —— 窗口锁死 57 km,只加密 nx(mesh_convergence_large)
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BAND_LABEL = {'0.05-0.1': '0.05-0.10', '0.1-0.2': '0.10-0.20', '0.2-9.0': '> 0.20',
              '0.0-0.02': '< 0.02', '0.02-0.05': '0.02-0.05'}


def fmt_band(b):
    return BAND_LABEL.get(b, b)


def table_footprint():
    """窗口收敛表 —— 合并窄档(20/28/40 km)与宽档(80/114/160 km)两次运行。

    """
    dw = json.load(open(os.path.join(HERE, 'footprint_fixed_resolution_test.json'),
                        encoding='utf-8'))
    pn = os.path.join(HERE, 'footprint_narrow_test.json')
    dn = json.load(open(pn, encoding='utf-8')) if os.path.exists(pn) else None

    if dn is None:
        print('  ⚠ 缺 footprint_narrow_test.json,窗口表仅覆盖宽档')
        d, wins, rows = dw, list(dw['windows_km']), list(dw['rows'])
        summ = {b: (s['median_absB1'], s['pct_change'], s['n'])
                for b, s in dw['summary'].items() if isinstance(s, dict)}
    else:
        wins = sorted(set(dn['windows_km']) | set(dw['windows_km']))
        idx = {round(r['absB1'], 12): r for r in dn['rows']}
        assert len(idx) == len(dn['rows']), '窄档存在重复 absB1,无法唯一配对'
        rows = []
        for r in dw['rows']:
            k = round(r['absB1'], 12)
            assert k in idx, '宽档测点 %.6f 在窄档中找不到对应项' % r['absB1']
            a, b = idx[k]['B']['57'], r['B']['57']
            assert abs(a - b) < 1e-9, ('57 km 锚点不一致', r['absB1'], a, b)
            merged = dict(r)
            merged['B'] = dict(idx[k]['B'])
            merged['B'].update(r['B'])
            for w in wins:
                assert str(w) in merged['B'], (r['absB1'], w)
            rows.append(merged)
        summ = {}
        for b, s in dw['summary'].items():
            if not isinstance(s, dict) or b not in dn['summary']:
                continue
            m = dict(dn['summary'][b]['median_absB1'])
            m.update(s['median_absB1'])
            lo, hi = m[str(wins[0])], m[str(wins[-1])]
            summ[b] = (m, 100 * (hi - lo) / lo, s['n'])
        d = dict(dw)
        d['windows_km'] = wins

    rows = sorted(rows, key=lambda r: -r['absB1'])
    head = ('| Site (\\|B - 1\\| at production) | Band | ' +
            ' | '.join('%s km' % w for w in wins) +
            ' | Change %s to %s km |' % (wins[0], wins[-1]))
    sep = '|---' * (len(wins) + 3) + '|'
    out = [head, sep]
    for r in rows:
        b0 = abs(r['B'][str(wins[0])] - 1)
        b1 = abs(r['B'][str(wins[-1])] - 1)
        out.append('| %.3f | %s | ' % (r['absB1'], fmt_band(r['band'])) +
                   ' | '.join('%.4f' % abs(r['B'][str(w)] - 1) for w in wins) +
                   ' | %+.1f%% |' % (100 * (b1 - b0) / max(b0, 1e-9)))
    out.append(sep.replace('---', ':---:'))
    for band, (m, chg, n) in summ.items():
        out.append('| **median, n = %d** | **%s** | ' % (n, fmt_band(band)) +
                   ' | '.join('**%.4f**' % m[str(w)] for w in wins) +
                   ' | **%+.1f%%** |' % chg)
    return '\n'.join(out), d


def table_mesh():
    """2026-09-30 M11:150 测点扩大试验,逐点行过多,改为分档汇总(逐点数据在 mesh_convergence_large.json)。"""
    d = json.load(open(os.path.join(HERE, 'mesh_convergence_large.json'), encoding='utf-8'))
    nxs = d['nx_ladder']
    lab = {'0.0-0.02': '<0.02', '0.02-0.05': '0.02-0.05', '0.05-0.1': '0.05-0.10', '0.1-0.2': '0.10-0.20',
           '0.2-9.0': '>0.20'}
    head = ('| \\|B - 1\\| at production | n | ' + ' | '.join(f'nx = {n}' for n in nxs) +
            ' | Bias 40 to 128 (95% CI) | Site scatter | Step 96 to 128 | Anti-aliased 40 to 128 | Sign flips |')
    sep = '|---' * (len(nxs) + 7) + '|'
    out = [head, sep]
    for k, s in d['summary'].items():
        b = d['bands'][lab[k]]
        out.append(f'| {fmt_band(k)} | {s["n"]} | ' +
                   ' | '.join(f'{s["median_absB1"][str(n)]:.4f}' for n in nxs) +
                   f' | {b["bias_40_to_128_pct"]:+.1f}% ({b["bias_ci"][0]:+.1f}, {b["bias_ci"][1]:+.1f})'
                   f' | {b["scatter_40_to_128_pct"]:.1f}% | {b["step_96_to_128_pct"]:+.1f}%'
                   f' | {b["bias_40aa_to_128_pct"]:+.1f}% | {b["sign_flip_40"]} |')
    return '\n'.join(out), d


def main():
    t2, d2 = table_footprint()
    t3, d3 = table_mesh()
    w0, w1 = d2['windows_km'][0], d2['windows_km'][-1]
    txt = (
        '**Extended Data Table 2 | Footprint convergence at fixed grid resolution.** '
        'Centre-pixel |B - 1| for 30 real sites with |B - 1| >= 0.05 at the production setting, '
        f'as the tile footprint varies from {w0} to {w1} km (a factor of {w1/w0:.0f} in linear '
        f'scale) with the element size held at {d2["elem_m"]:.0f} m throughout, so that changing '
        'the window adds or removes long-wavelength terrain without also changing how finely the '
        'mesh resolves it. Sites are ordered by their production |B - 1|.\n\n' + t2 + '\n\n' +
        '**Extended Data Table 3 | Grid convergence at fixed footprint.** '
        f'Median centre-pixel |B - 1| for {len(d3["rows"])} sites (about 30 per band, bands by the production |B - 1|) as the tile is discretised more finely, '
        f'from nx = {d3["nx_ladder"][0]} (element {57000/d3["nx_ladder"][0]:.0f} m, the production '
        f'setting) to nx = {d3["nx_ladder"][-1]} (element {57000/d3["nx_ladder"][-1]:.0f} m, '
        'matching the 450-m digital elevation model), with the 57-km footprint held fixed. '
        'A positive change means the production setting underestimates the terrain effect.\n\n'
        + t3 + '\n')
    p = os.path.join(HERE, 'convergence_tables.md')
    open(p, 'w', encoding='utf-8').write(txt)
    print(f'已写出 {os.path.basename(p)}')
    print(f'  Table 2: {len(d2["rows"])} 个测点 x {len(d2["windows_km"])} 档窗口 '
          f'({w0}-{w1} km)')
    print(f'  Table 3: {len(d3["rows"])} 个测点 '
          f'x {len(d3["nx_ladder"])} 档网格')


if __name__ == '__main__':
    main()
