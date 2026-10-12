# -*- coding: utf-8 -*-
"""3.1 脊位/谷位几何一致性检验 —— 补上它缺失的产物。



检验的内容
----------
算子的预测本身是否带有地形校正理论所要求的几何:地势高处传导热向两侧发散,
B 应当小于 1;地势低处相反。**这必须拿预测去问,不能拿参考场去问** ——
拿参考场问只能验证有限元,验证不了算子。

在独立 500 瓦片集上(该集不参与任何模型决策):
  · 按中心像元高于还是低于瓦片平均高程分为脊位与谷位
  · 比较预测 B 与参考 B 在两类上的中位数
  · 统计预测与参考的偏离方向一致率
  · 预测偏离量与中心像元相对高程的相关
  · 仅凭相对高程的符号能预测多少比例的 B 符号 —— 在有限元场与算子场上各算一次。
    两者接近,说明算子既没有在物理不遵循简单规则处强加该规则,
    也没有在物理遵循处把它丢掉。
"""
import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))

d = np.load(os.path.join(HERE, 'validation_independent_500.npz'))
dems = d['dems']                       # (500, 128, 128) 输入高程
b_fem = d['B_fem_center'].astype(float)
b_fno = d['B_fno_center'].astype(float)

# 中心像元相对该瓦片平均高程的偏差。128×128 的中心取 [64, 64],
# 与 tile_reader 的重采样口径一致。
c = dems.shape[1] // 2
h_c = dems[:, c, c].astype(float)
h_mean = dems.reshape(len(dems), -1).mean(axis=1)
rel_h = h_c - h_mean

ridge = rel_h > 0                      # 中心高于瓦片均值 = 脊位
valley = ~ridge

dev_fno = b_fno - 1.0
dev_fem = b_fem - 1.0

res = dict(
    n=int(len(b_fno)),
    n_ridge=int(ridge.sum()), n_valley=int(valley.sum()),
    # 预测与参考在两类上的中位数
    pred_med_ridge=round(float(np.median(b_fno[ridge])), 3),
    pred_med_valley=round(float(np.median(b_fno[valley])), 3),
    fem_med_ridge=round(float(np.median(b_fem[ridge])), 3),
    fem_med_valley=round(float(np.median(b_fem[valley])), 3),
    # 预测与参考的偏离方向一致率
    sign_agree_pct=round(100 * float((np.sign(dev_fno) == np.sign(dev_fem)).mean()), 1),
    # 预测偏离量与中心像元相对高程的相关
    corr_dev_relh=round(float(np.corrcoef(dev_fno, rel_h)[0, 1]), 3),
    # 仅凭相对高程的符号能预测多少 B 的符号
    # (相对高程为正 -> B 应当小于 1,故取相反号)
    sign_from_relh_fem_pct=round(
        100 * float((np.sign(dev_fem) == -np.sign(rel_h)).mean()), 1),
    sign_from_relh_fno_pct=round(
        100 * float((np.sign(dev_fno) == -np.sign(rel_h)).mean()), 1),
)
json.dump(res, open(os.path.join(HERE, 'geometry_consistency_test.json'), 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1)

print('独立 500 瓦片集:脊位 %d,谷位 %d' % (res['n_ridge'], res['n_valley']))
print()
print('%-10s %10s %10s' % ('', '脊位', '谷位'))
print('%-10s %10.3f %10.3f' % ('预测 B 中位', res['pred_med_ridge'], res['pred_med_valley']))
print('%-10s %10.3f %10.3f' % ('参考 B 中位', res['fem_med_ridge'], res['fem_med_valley']))
print()
print('预测与参考的偏离方向一致率      %.1f%%' % res['sign_agree_pct'])
print('预测偏离量与中心像元相对高程相关  r = %+.3f' % res['corr_dev_relh'])
print()
print('仅凭相对高程预测 B 符号的准确率')
print('  在有限元场上  %.1f%%' % res['sign_from_relh_fem_pct'])
print('  在算子场上    %.1f%%' % res['sign_from_relh_fno_pct'])
print()
print('已保存 geometry_consistency_test.json')
