# -*- coding: utf-8 -*-
"""按波长扫描的解析解基准 —— 用于判定生产配置在**应用集代表性波长**上的真实离散误差。


本脚本把两者分开:
  1. 解析地表解      B_surf - 1 = -a*k              (波峰)
  2. 解析深度平均解  B_avg  - 1 = (B_surf-1) * (1-e^{-kD})/(kD)
     (Lachenbruch 扰动按 e^{-kz} 衰减,故 [0,D] 平均有闭式解)
  3. 生产配置 FEM 解
误差 = (3) 相对 (2) 的偏差,才是**扣除设计选择之后的真实离散误差**。

参照深度的口径:(2) 必须用校正流水线实际假定的平均深度。rebuild_correction.py 把算子
输出当作 [0, D_0] 的平均,D_0 = 域深/10;而以 vertical_mesh_test 反推的等效深度
(444 m)作参照,会把一部分离散误差吸收进"口径"(那个深度正是让 n_z = 10 与细网格对上
的深度)。故主结果 err_vs_D0_pct 取 D = 域深/10;444 m 口径与实际顶层厚度口径并列保留。

同时扫描多个波长,检验稿件"真实高起伏地形由更短波长主导,故 28.5 km 不代表应用集"
这一辩护是否成立。应用集实测:质心波长 7.8-35.6 km,e 折深度 675-2,281 m。
"""
import os
import sys
import json
import math

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from fem3d import Fem3D

HERE = os.path.dirname(os.path.abspath(__file__))
DOMAIN_KM = 57.0
RES = 128
CELL_M = DOMAIN_KM * 1000 / RES
EQUIV_DEPTH_M = 444.35          # vertical_mesh_test.json 实测的等效深度平均区间


def sine_dem(a_m, lam_m):
    x = np.arange(RES) * CELL_M
    return np.tile(a_m * np.cos(2 * np.pi * x / lam_m), (RES, 1))


def solve_centre_B(dem, nx, nz, depth_m):
    f = Fem3D(nx=nx, ny=nx, nz=nz)
    f.build(dem, CELL_M, depth=depth_m)
    f.solve_heat()
    f.compute_B()
    G = f.B_grid
    return float(G[G.shape[0] // 2, G.shape[0] // 2])


def main():
    rows = []
    # N_PERIODS 必须整除以保证左右边界落在对称面上
    print(f'{"lambda":>8}{"a":>6}{"e折km":>8}{"域深m":>8}{"域/e折":>8}'
          f'{"解析地表":>10}{"解析平均":>10}{"生产FEM":>10}'
          f'{"vs地表":>9}{"vs平均":>9}')
    print('-' * 96)
    for n_per in (2, 4, 6, 8, 10):
        lam = DOMAIN_KM * 1000 / n_per
        k = 2 * math.pi / lam
        e_fold = 1.0 / k
        shrink = (1 - math.exp(-k * EQUIV_DEPTH_M)) / (k * EQUIV_DEPTH_M)
        for a_m in (100.0, 300.0):
            dep_surf = -a_m * k                       # B_surf - 1 (波峰)
            dep_avg = dep_surf * shrink               # B_avg  - 1
            relief = 2 * a_m
            depth_dom = max(3000.0, 1.5 * relief)
            B = solve_centre_B(sine_dem(a_m, lam), 40, 10, depth_dom)
            dep_fem = B - 1.0
            e_surf = 100 * (dep_fem - dep_surf) / abs(dep_surf)
            e_avg = 100 * (dep_fem - dep_avg) / abs(dep_avg)
            d0 = depth_dom / 10.0                          # 流水线的 D_0
            dact = (2 * a_m + depth_dom) / 10.0            # 波峰处顶层单元的实际厚度
            avg_at = lambda D: dep_surf * (1 - math.exp(-k * D)) / (k * D)
            e_d0 = 100 * (dep_fem - avg_at(d0)) / abs(avg_at(d0))
            e_act = 100 * (dep_fem - avg_at(dact)) / abs(avg_at(dact))
            print(f'{lam/1000:>7.1f}k{a_m:>6.0f}{e_fold/1000:>8.2f}'
                  f'{depth_dom:>8.0f}{depth_dom/e_fold:>8.2f}'
                  f'{dep_surf:>10.5f}{dep_avg:>10.5f}{dep_fem:>10.5f}'
                  f'{e_surf:>8.1f}%{e_avg:>8.1f}%')
            rows.append(dict(lambda_km=lam / 1000, a_m=a_m,
                             e_fold_m=e_fold, domain_m=depth_dom,
                             domain_over_efold=depth_dom / e_fold,
                             dep_surface=dep_surf, dep_depthavg=dep_avg,
                             dep_fem=dep_fem,
                             err_vs_surface_pct=e_surf,
                             err_vs_depthavg_pct=e_avg,
                             err_vs_D0_pct=e_d0, err_vs_actual_layer_pct=e_act))
    out = dict(equiv_depth_m=EQUIV_DEPTH_M, domain_km=DOMAIN_KM,
               note='err_vs_depthavg_pct 是扣除深度平均设计选择之后的真实离散误差',
               rows=rows)
    with open(os.path.join(HERE, 'analytic_benchmark_wavelength.json'),
              'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print('\n已保存 analytic_benchmark_wavelength.json')


if __name__ == '__main__':
    main()
