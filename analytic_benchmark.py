# -*- coding: utf-8 -*-
"""有限元求解器对解析解的验证 —— Lachenbruch (1968) 型周期地形。


解析解
------
地表 z = a cos(kx),k = 2 pi / lambda,等温地表 T = 0,深部通量 q0。
线性化(a k << 1)得地表通量 q(x) = q0 [1 - a k cos(kx)],即
    波峰下 B = 1 - 2 pi a / lambda
    波谷下 B = 1 + 2 pi a / lambda
误差为 O((ak)^2),故用一列递减的 a/lambda 检验收敛阶。

边界相容性
----------
域宽取整数个周期,则左右边界落在波峰(对称面),与零通量侧边界条件严格相容;
地形只沿 x 变化,y 方向亦然。因此不施加渐变(taper),以免破坏解析形式。

扰动随深度按 e^{kz} 衰减,e 折深度 = lambda / 2 pi。域深须远大于该值,否则底部
截断会与离散误差混在一起;本脚本默认取 5 个 e 折深度,并另报生产深度公式的结果。
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from fem3d import Fem3D

HERE = os.path.dirname(os.path.abspath(__file__))
DOMAIN_KM = 57.0
RES = 128
N_PERIODS = 2                       # 域内周期数 -> lambda = 57/2 = 28.5 km
# 地表通量精度主要由垂向分辨率决定,故固定 nx=40 加密 nz,另加一档水平检查。
# nx=128/nz=40 为 68 万节点,直接法分解内存不足,已剔除。
LADDER = [(40, 10), (40, 40), (40, 80), (80, 40)]
LAMBDA_M = DOMAIN_KM * 1000 / N_PERIODS
CELL_M = DOMAIN_KM * 1000 / RES


def sine_dem(a_m):
    """域宽内 N_PERIODS 个完整周期,左右边界落在波峰。"""
    x = np.arange(RES) * CELL_M
    prof = a_m * np.cos(2 * np.pi * x / LAMBDA_M)
    return np.tile(prof, (RES, 1))


def solve_B(dem, nx, nz, depth_m):
    f = Fem3D(nx=nx, ny=nx, nz=nz)
    f.build(dem, CELL_M, depth=depth_m)
    f.solve_heat(); f.compute_B()
    G = f.B_grid
    c = G.shape[0] // 2
    # 域内 N_PERIODS 个整周期时,域中心落在 x = N_PERIODS/2 个波长处。
    # N_PERIODS=2 -> 中心在 x = lambda,cos(2pi)=+1,即**波峰**。
    return float(G[c, c]), G


def main():
    k = 2 * np.pi / LAMBDA_M
    e_fold = 1.0 / k
    deep = 5 * e_fold
    print(f'域宽 {DOMAIN_KM:.0f} km,周期数 {N_PERIODS},lambda = {LAMBDA_M/1000:.1f} km')
    print(f'扰动 e 折深度 = lambda/2pi = {e_fold/1000:.2f} km,基准域深取 5 倍 = {deep/1000:.1f} km\n')

    out = {'lambda_m': LAMBDA_M, 'cases': []}
    print(f'{"a (m)":>7}{"a·k":>9}{"解析 B(峰)":>12}'
          + ''.join(f'{f"nx{nx}/nz{nz}":>13}' for nx, nz in LADDER)
          + f'{"生产配置":>13}')
    for a_m in [50.0, 100.0, 200.0, 400.0]:
        dem = sine_dem(a_m)
        ak = a_m * k
        B_true = 1.0 - ak            # 域中心为波峰(N_PERIODS=2,见 solve_B 注释)
        row = {'a_m': a_m, 'ak': ak, 'B_analytic': B_true, 'fem': {}}
        cells = []
        for nx, nz in LADDER:
            B, _ = solve_B(dem, nx, nz, deep)
            err = 100 * (B - B_true) / (B_true - 1)
            row['fem'][f'nx{nx}_nz{nz}'] = dict(B=B, rel_err_pct=err)
            cells.append(f'{B:.4f}({err:+.1f}%)')
        # 生产配置:nx=40, nz=10, depth = max(3000, 1.5*relief)
        prod_depth = max(3000.0, 1.5 * 2 * a_m)
        Bp, _ = solve_B(dem, 40, 10, prod_depth)
        errp = 100 * (Bp - B_true) / (B_true - 1)
        row['production'] = dict(B=Bp, depth_m=prod_depth, rel_err_pct=errp)
        out['cases'].append(row)
        print(f'{a_m:>7.0f}{ak:>9.4f}{B_true:>12.4f}'
              + ''.join(f'{c:>13}' for c in cells) + f'{f"{Bp:.4f}({errp:+.0f}%)":>13}')

    print(f'\n说明:括号内为相对解析偏离量 (B-1) 的误差百分比。')
    print(f'"生产配置"沿用 max(3,000 m, 1.5x起伏) 的域深公式,对本例的长波地形而言')
    print(f'远浅于 e 折深度 {e_fold/1000:.2f} km,故该列同时含底部截断误差。')
    json.dump(out, open(os.path.join(HERE, 'analytic_benchmark.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('\n已保存 analytic_benchmark.json')


if __name__ == '__main__':
    main()
