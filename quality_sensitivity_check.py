#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quality_sensitivity_check.py — IHFC质量码敏感性检验 (论文附录用)


按 IHFC_2024论文精读与写作指南.md 第7/14节的警告: 不能只设一个笼统质量阈值,
U-score和M-score要分开处理, Ux/Mx不等于数据差而是信息不足。

用法: python quality_sensitivity_check.py   (需先跑完 rebuild_correction.py)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

d = np.load('rebuild_points.npz')
lon, lat = d['lon'], d['lat']
q_meas, q_corr = d['q_measured'], d['q_corrected']
qual = d['quality_score']  # 形如 "U2.M3x" 的字符串数组

u_score = np.array([s.split('.')[0] if '.' in s else '' for s in qual])
m_score = np.array([s.split('.')[1] if '.' in s else '' for s in qual])

dq_rel = 100 * (q_corr - q_meas) / q_meas

def summarize(name, mask):
    n = int(mask.sum())
    pct = 100 * n / len(mask)
    if n == 0:
        print(f'{name:28s}  n=0, 跳过')
        return
    print(f'{name:28s}  n={n:,} ({pct:.1f}%)  '
          f'q_meas均值={q_meas[mask].mean():.1f}  q_corr均值={q_corr[mask].mean():.1f}  '
          f'|校正幅度|均值={np.abs(dq_rel[mask]).mean():.1f}%  '
          f'校正幅度中位数={np.median(dq_rel[mask]):.1f}%')

print('=== 全量 vs. 质量码子集 对比 (论文正文用全量, 以下仅作敏感性检验) ===\n')
summarize('全量 (与Neumann et al.同口径)', np.ones(len(qual), dtype=bool))
summarize('U-score已评估 (U1~U4, 排除Ux)', u_score != 'Ux')
for u in ['U1', 'U2', 'U3', 'U4']:
    summarize(f'  仅{u}', u_score == u)
summarize('M-score较优 (M1/M2/M3, 排除M4/Mx及其x后缀)', np.isin(m_score, ['M1', 'M2', 'M3']))
summarize('U与M均已评估且较优', (u_score != 'Ux') & np.isin(m_score, ['M1', 'M2', 'M3']))

print('\n注意: U-score已评估子集通常只占全量的一小部分(参见精读指南统计, Ux约占83%)。')
print('若子集统计量与全量接近, 说明结果对质量码筛选不敏感, 可以在附录如实报告两组数字;')
print('若明显不同, 不能简单认为"用了质量码筛选的结果更可信"——小子集本身可能有地域/方法学偏倚,')
print('需要在文中说明子集的代表性局限, 而不是用它替换主结果。')
