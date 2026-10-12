# -*- coding: utf-8 -*-
"""方法评价计分表 —— 把分散在全文的验证结果,按「指标 + 判定标准 + 分级结论」统一成一张表。

为什么要有这张表
----------------
各项验证原本是先报数字、再用文字解释,读者看不出「多少算合格」,不利的结果也要靠
一整段话去消化。这里给每项检验写明判定标准,并按统一的三档给出结论:
    通过 / 部分通过(注明哪一部分不通过)/ 未通过


阈值来源:
  (1) 不劣化  方法误差折算成通量的相对误差 < 5%,即 IHFC 最优不确定度等级 U1 的
             变异系数上限(Fuchs et al., 2023)—— 校正不应给最好的测量添上可比的误差
  (2) 可分辨  信号 ≥ 2 倍方法误差(通行的 2σ 约定)
  (3) 泛化    独立集与空间留出集的 R² 之差小于四折空间交叉验证的折间标准差
  (4) 收敛    继续加密/放大带来的末级变化小于有限元自身的离散误差(error_model.fe_bias,约 9%)
  (5) 观测    物理预测方向上的单侧检验 p < 0.05

四个层级对应真值的来源:
  I   数值真值    有解析解的正弦地形(检验有限元求解器)
  II  有限元真值  独立 500 瓦片集(检验算子)
  III 无真值      观测记录中的独立信息(检验地形效应是否真实存在)
  IV  适用范围    窗口、网格、垂向离散的收敛性

全部数字读自已有产物,不做新计算。输出 evaluation_scorecard.json。
"""
import json
import math
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(HERE, f), encoding='utf-8'))

U1_PCT = 5.0            # (1) IHFC U1 的变异系数上限
SNR_MIN = 2.0           # (2)
sys.path.insert(0, HERE)
from error_model import fe_bias  # noqa: E402
FE_REL_PCT = round(100 * fe_bias(), 1)   # (4) 有限元离散误差(与误差模型同一取值)
ALPHA = 0.05            # (5)

PASS, PART, FAIL = '通过', '部分通过', '未通过'
rows = []


def add(tier, rid, name, metric, criterion, value, verdict, note=''):
    rows.append(dict(tier=tier, id=rid, name=name, metric=metric, criterion=criterion,
                     value=value, verdict=verdict, note=note))


TIER_ZH = {'I': 'I 解析真值', 'II': 'II 有限元真值', 'III': 'III 观测一致性', 'IV': 'IV 收敛性'}
TABLE2_HEAD = ('| 层级 | 检验 | 指标 | 参照 | 实测 | 结果说明 |\n'
               '|---|---|---|---|---|---|')


_SUPS = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


def sci(p):
    """p 值按正文体例写:≥ 0.001 写小数;更小写「a × 10⁻ⁿ」;极小写「< 10⁻ⁿ」。"""
    if p >= 0.1:
        return 'p = %.2f' % p
    if p >= 1e-3:
        return 'p = %.3f' % p
    e = math.floor(math.log10(p))
    if p < 1e-20:
        return 'p < 10' + str(e + 1).translate(_SUPS)
    m = round(p / 10 ** e)
    if m == 10:
        m, e = 1, e + 1
    return 'p = %d × 10%s' % (m, str(e).translate(_SUPS))


def _zh_punct(s):
    """中文稿体例:全角括号、分号、逗号(千分位逗号不动),负号用 U+2212。"""
    import re
    s = s.replace('(', '（').replace(')', '）').replace(';', '；')
    s = re.sub(r',(?!\d{3})', '，', s)
    s = re.sub(r'(?<=[\u4e00-\u9fff\w)）%]):(?=\S)', '：', s)   # 中文表格用全角冒号
    return re.sub(r'(?<![\w)）])-(?=\d)', '−', s)


def render_table2_zh(sc=None):
    """稿件表 2 的正文行,由 evaluation_scorecard.json 直接渲染。

    check_pipeline_consistency.py 用它核对稿件里的表 2 —— 表中 15 行数字若靠手抄,
    重跑任何一项检验后都会悄悄过时。
    """
    sc = sc or J('evaluation_scorecard.json')
    lines = [TABLE2_HEAD]
    for r in sc['rows']:
        verdict = r['note'] if r['note'] else ('达到参照' if r['verdict'] == PASS else '见正文')
        cells = [TIER_ZH[r['tier']], r['name'], r['metric'], r['criterion'], r['value'], verdict]
        lines.append('| ' + ' | '.join(_zh_punct(c).replace('|', '\\|') for c in cells) + ' |')
    return '\n'.join(lines)


def compute():
    """重算全部判定,返回将写入 evaluation_scorecard.json 的内容(不写盘)。"""
    rows.clear()
    N = J('numbers.json')

    # ── I 数值真值:有限元对解析深度平均解 ─────────────────────────────
    s = J('depth_decay_spectral.json')
    fe_med, fe_max = s['fe_D0_median_pct'], s['fe_D0_max_pct']
    add('I', 'I-1', '有限元离散误差', 'ε_FE = |B_FE − B_ana| / |B_ana − 1|(参照深度取计算域深度的十分之一;40 条真实剖面插值)',
        'ε_FE < 1/2,即校正量 ≥ 2 倍离散误差',
        '典型波长 %.1f%%;剖面插值中位 %.1f%%,最大 %.1f%%' % (FE_REL_PCT, fe_med, fe_max),
        PASS if fe_max < 100 / SNR_MIN else PART,
        '方向单一(低估 |B−1|),作为系统偏差单列于附表 S3')

    # ── II 有限元真值:独立 500 瓦片集 ────────────────────────────────
    r2i, r2h, sd = N['r2_independent'], N['r2_spatial_split'], N['cv_r2_std']
    ex = J('validation_independent_excl.json') if os.path.exists(os.path.join(HERE, 'validation_independent_excl.json')) else None
    _hd = os.path.join(HERE, 'holdout_d4_r2.json')
    if os.path.exists(_hd):
        r2h = J('holdout_d4_r2.json')['r2_d4']
    if ex:
        r2e = ex['r2_field']
        add('II', 'II-1', '泛化(与训练地形无重叠)', '|R²(排除距离集) − R²(留出集)|,排除距离集与任何训练瓦片中心相距 ≥ %.1f km' % ex['excl_km'],
            '< 四折折间标准差 %.3f' % sd, '%.4f(R² %.3f 对 %.3f)' % (abs(r2e - r2h), r2e, r2h),
            PASS if abs(r2e - r2h) < sd else FAIL)
    else:
        add('II', 'II-1', '泛化(模型选择无乐观偏差)', '|R²_独立 − R²_留出|',
            '< 四折折间标准差 %.3f' % sd, '%.4f' % abs(r2i - r2h),
            PASS if abs(r2i - r2h) < sd else FAIL)

    sr0 = J('site_reliability.json')['by_band']
    rel = {k: v['median_rel_sigma_pct'] for k, v in sr0.items()}
    add('II', 'II-2', '不劣化(必要条件,非充分)', 'σ_q / q_cor(逐测点随机误差,档内中位)',
        '< %.0f%%(IHFC U1 上限)' % U1_PCT,
        ' / '.join('%.2f%%' % v for v in rel.values()) + '(|B−1| 五档由低到高)',
        PASS if max(rel.values()) < U1_PCT else PART,
        '' if max(rel.values()) < U1_PCT else '最高档为通量的 %.1f%%,与 U1 上限相当;其余各档低于上限' % max(rel.values()))

    sr = J('site_reliability.json')
    lab = {'<0.02': '< 0.02', '0.02-0.05': '0.02–0.05', '0.05-0.10': '0.05–0.10',
           '0.10-0.20': '0.10–0.20', '>0.20': '> 0.20'}
    for k, v in sr['by_band'].items():
        snr = v['median_snr']
        ver = PASS if snr >= SNR_MIN else (PART if snr >= 1 else FAIL)
        add('II', 'II-3', '显著(|B−1| %s 档,n = %s)' % (lab[k], format(v['n'], ',')),
            '信噪比 |Δq| / σ_q(逐测点传播,档内中位)', '≥ %.0f' % SNR_MIN,
            '%.2f' % snr, ver,
            {FAIL: '近平坦档校正量本身很小,与方法误差同量级',
             PART: '校正量为方法误差的 1–2 倍'}.get(ver, ''))

    g = J('geometry_consistency_test.json')
    n = g['n']
    k = round(g['sign_agree_pct'] / 100 * n)
    p_sign = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    se_diff = math.sqrt(2 * 0.7 * 0.3 / n) * 100
    dd = abs(g['sign_from_relh_fno_pct'] - g['sign_from_relh_fem_pct'])
    add('II', 'II-4', '几何物理(脊谷符号)', '预测与参考偏离方向一致率;仅凭高低判符号的正确率之差',
        '一致率显著高于 50%%(p < %.2f);两者之差 < 2 倍标准误(%.1f 个百分点)' % (ALPHA, 2 * se_diff),
        '%.1f%%(%s);%.1f 个百分点' % (g['sign_agree_pct'], sci(p_sign), dd),
        PASS if (p_sign < ALPHA and dd < 2 * se_diff) else PART)

    # 两套独立样本都报:原独立集(按预测档)与排除距离集(按有限元档),二者结论不同,不能只取一套
    lb = J('linear_spectral_baseline.json')['center_by_band']
    lx = J('validation_independent_excl.json')['by_band']
    pairs = [('独立生成集 ' + b.replace('>=', '≥ ').replace('-', '–'), v['operator']['rmse'], v['linear41']['rmse'])
             for b, v in lb.items()]
    pairs += [('排除距离集 ' + b.replace('>=', '≥ ').replace('-', '–'), v['rmse_center'], v['linear_rmse_center'])
              for b, v in lx.items()]
    win = [p for p in pairs if p[1] < p[2]]
    add('II', 'II-5', '相对线性谱解的增益', '中心像元 RMSE:算子对线性谱解(Lachenbruch 型一阶解,同一几何)',
        '各档算子 RMSE 低于线性谱解',
        '; '.join('%s:%.4f 对 %.4f' % p for p in pairs),
        PASS if len(win) == len(pairs) else (PART if win else FAIL),
        '' if len(win) == len(pairs) else '%d / %d 个档位算子更准(%s);其余档位线性谱解更准,|B−1| < 0.10 时二者互有优劣(3.1 节)'
        % (len(win), len(pairs), '、'.join(w[0] for w in win)))

    # ── III 无真值:观测记录中的独立信息 ────────────────────────────
    pn = J('paired_null_fairness_test.json')
    bands = [b for b in pn if b not in ('全部',) and isinstance(pn[b], dict) and 'p_B' in pn[b]]
    ok = [b for b in bands if pn[b]['p_B'] < ALPHA and pn[b]['marginB'] > 0]
    bad = [b for b in bands if b not in ok]
    add('III', 'III-1', '邻近测点配对(逐档)', '配对热流差标准差的降幅,真实校正减同档置换对照(降幅差)',
        '各档降幅差 > 0 且单侧 p < %.2f' % ALPHA, '%d / %d 档满足' % (len(ok), len(bands)),
        PASS if not bad else PART,
        '; '.join('%s 档 n = %s、降幅差 %+.2f 个百分点、%s'
                  % (b, format(pn[b]['n'], ','), pn[b]['marginB'], sci(pn[b]['p_B'])) for b in bad))

    ms = J('magnitude_scaling_test.json')
    pb = [b for b in ms['paired'] if b != '全部']
    net = {b: ms['paired'][b]['mad_reduction_at_1'] for b in pb}
    add('III', 'III-1b', '邻近测点配对:净效果', '真实校正本身使配对对数热流比的离散度(MAD)减小的百分比',
        '各档 > 0', ' / '.join('%+.2f%%' % v for v in net.values()) + '(|B−1| 五档由低到高)',
        PASS if all(v > 0 for v in net.values()) else (PART if any(v > 0 for v in net.values()) else FAIL),
        '在 %d / %d 档减小离散;用于检验方向,不用于标定幅度(稳健口径下观测最支持的缩放为几何校正的 %.2f–%.2f 倍)'
        % (sum(v > 0 for v in net.values()), len(net),
           min(ms['paired'][b]['alpha_best_mad'] for b in pb[2:]), max(ms['paired'][b]['alpha_best_mad'] for b in pb[2:])))

    c = J('c15_ordinal_test.json')
    d1, d2 = c['adjacent_deltas'][0]['cliffs_delta'], c['adjacent_deltas'][1]['cliffs_delta']
    add('III', 'III-2', '原研究者现场判断', '“识别出但未校正”与“判定存在但不显著”两类的 |B−1|(单侧 Mann–Whitney);Cliff\'s δ',
        '单侧 p < %.2f 且 δ > 0' % ALPHA,
        '%s;δ = %+.2f' % (sci(c['step1_mw_p']), d1),
        PASS if (c['step1_mw_p'] < ALPHA and d1 > 0) else FAIL,
        '“研究者自行校正”一类的 |B−1| 最低(δ = %+.2f),三类不单调;同一位置类别矛盾者整点剔除' % d2)
    ct = J('c15_terrain_baseline.json')
    add('III', 'III-2b', '现场判断:控制起伏后', '逻辑回归(前两类):|B−1| 与瓦片起伏的秩一并进入时 |B−1| 的系数',
        '系数 > 0 且 p < %.2f' % ALPHA,
        '系数 %+.2f(%s);起伏本身 %+.2f(%s)' % (ct['logit_both']['absB1']['coef'], sci(ct['logit_both']['absB1']['p']),
                                              ct['logit_both']['relief']['coef'], sci(ct['logit_both']['relief']['p'])),
        PASS if (ct['logit_both']['absB1']['coef'] > 0 and ct['logit_both']['absB1']['p'] < ALPHA) else FAIL,
        '')

    dv, tb = J('davies_validation.json'), J('davies_terrain_baseline.json')
    _md = J('magnitude_scaling_test.json')['davies']
    cis = {b: (_md[b]['ci_cluster'][0], _md[b]['ci_cluster'][1], _md[b]['slope']) for b in ('全部', '>= 0.05', '>= 0.10')}
    dir_ok = all(v[0] > 0 for v in cis.values()) and tb['partial_B_p'] < ALPHA
    add('III', 'III-3a', '独立编制的热流图:方向', '残差对相对传递因子的斜率(95% CI);控制裸地形代理后的偏系数',
        '斜率 CI(按 Davies 格点聚类自助)不含 0;偏系数 p < %.2f' % ALPHA,
        '斜率 %.2f–%.2f(三档);偏系数 %s'
        % (min(v[2] for v in cis.values()), max(v[2] for v in cis.values()), sci(tb['partial_B_p'])),
        PASS if dir_ok else PART)
    mag_ok = any(v[0] <= 1 <= v[1] for v in cis.values())
    add('III', 'III-3b', '独立编制的热流图:幅度', '同一斜率与 1 的比较(斜率 1 即观测与预测等幅)',
        '至少一档的 95% CI 包含 1',
        '; '.join('%s:%.2f–%.2f' % (b.replace('>= ', '≥ '), v[0], v[1]) for b, v in cis.items()),
        PASS if mag_ok else FAIL,
        '' if mag_ok else '观测支持方向,幅度不作标定(3.3.3 节);并入地表温度随高程递减(区域地温梯度)后斜率为 %.2f–%.2f'
        % (min(_md[b]['slope_lapse_bg_5.5'] for b in _md), max(_md[b]['slope_lapse_bg_5.5'] for b in _md)))

    # ── IV 适用范围:收敛性 ──────────────────────────────────────────
    steps = [N['mesh_002_005_step_96_128_pct'], N['mesh_005_01_step_96_128_pct'],
             N['mesh_01_02_step_96_128_pct'], N['mesh_02_90_step_96_128_pct']]
    add('IV', 'IV-1', '水平网格收敛', '|B−1| 中位数在 n_x 96→128 的末级变化(|B−1| ≥ 0.02 各档)',
        '< %.1f%%(有限元离散误差)' % FE_REL_PCT, '最大 %.1f%%' % max(abs(v) for v in steps),
        PASS if max(abs(v) for v in steps) < FE_REL_PCT else PART,
        '近平坦档 %.1f%%(该档校正量本身很小)' % N['mesh_00_002_step_96_128_pct'])
    vs = N['vert_step_40_80_pct']
    add('IV', 'IV-2', '垂向离散收敛', '|B−1| 中位数在 n_z 40→80 的末级变化',
        '< %.1f%%' % FE_REL_PCT, '%.1f%%' % vs, PASS if abs(vs) < FE_REL_PCT else PART)
    # 用生产窗口 57 km 的截断(57→160 km),与误差预算的窗口项同一口径
    fw = {'0.05–0.10': N['fp_005_01_pct'], '0.10–0.20': N['fp_01_02_pct'],
          '> 0.20': N['fp_02_90_pct']}
    badw = {k_: v for k_, v in fw.items() if abs(v) >= FE_REL_PCT}
    add('IV', 'IV-3', '窗口收敛', '|B−1| 中位数由生产窗口 57 km 放大到 160 km 的总变化(非末级变化;|B−1| ≥ 0.05 各档)',
        '< %.1f%%' % FE_REL_PCT, ' / '.join('%+.1f%%' % v for v in fw.values()),
        PASS if not badw else PART,
        '; '.join('%s 档中位数 %+.1f%%(逐点变化量的中位只有 %+.1f%%,由个别测点驱动);误差预算按逐点变化计入(附表 S3);|B−1| < 0.05 未做窗口试验'
                  % (k_, v, 100 * J('error_budget.json')['terms']['win']['0.05-0.10']['bias_rel']) for k_, v in badw.items()))

    out = dict(anchors=dict(u1_pct=U1_PCT, snr_min=SNR_MIN, fe_rel_pct=FE_REL_PCT, alpha=ALPHA),
               fe_err_median_pct=round(fe_med, 1), fe_err_max_pct=round(fe_max, 1),
               rows=rows,
               summary={v: sum(r['verdict'] == v for r in rows)
                        for v in (PASS, PART, FAIL)})
    return out


def main():
    out = compute()
    json.dump(out, open(os.path.join(HERE, 'evaluation_scorecard.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    for r in rows:
        print('%-4s %-6s %-26s %-40s %s' % (r['tier'], r['id'], r['name'], r['value'], r['verdict']))
        if r['note']:
            print('%12s└ %s' % ('', r['note']))
    print('\n合计:' + ',  '.join('%s %d' % kv for kv in out['summary'].items()))
    print('已保存 evaluation_scorecard.json')


if __name__ == '__main__':
    main()
