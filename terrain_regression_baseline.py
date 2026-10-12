# -*- coding: utf-8 -*-
"""
terrain_regression_baseline.py -- 简单地形描述子基线

问题:B 的正负主要由"测点位于山脊还是山谷"决定,那么用少数几个标量地形
描述子做回归,能不能达到与神经算子相当的精度?本脚本量化"简单描述子捕捉
不到多少",而不是论证算子的必要性。

描述子(全部对 128x128 高程瓦片计算,只预测中心像素 B):
  1  relief          瓦片起伏 (max-min)
  2  std_elev        高程标准差
  3  rel_elev        中心高程相对瓦片均值(区分山脊/山谷,物理上最直接的量)
  4  rel_elev_norm   上一项除以起伏(无量纲化)
  5  slope_c         中心处坡度
  6  lap_c           中心处拉普拉斯(曲率,凸/凹)
  7  mean_slope      全瓦片平均坡度
  8  pct_rank_c      中心高程在瓦片内的百分位排名

模型:线性回归 与 梯度提升树(后者给"最好情况下的简单模型"上界)。
评估:与算子完全相同的空间不相交划分(k-means 12 类, seed=42, 测试集={0,4,8})。
"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

CELL_M = 57000.0 / 128


def descriptors(tile):
    """由 128x128 高程瓦片算出 8 个标量描述子。"""
    z = tile.astype(float)
    c = z.shape[0] // 2
    zc = z[c, c]
    zmean = z.mean()
    relief = z.max() - z.min()
    gy, gx = np.gradient(z, CELL_M)
    slope = np.hypot(gx, gy)
    lap = (np.gradient(gx, CELL_M, axis=1) + np.gradient(gy, CELL_M, axis=0))
    return [
        relief,
        z.std(),
        zc - zmean,
        (zc - zmean) / relief if relief > 1e-6 else 0.0,
        slope[c, c],
        lap[c, c],
        slope.mean(),
        float((z < zc).mean()),
    ]


NAMES = ['relief', 'std_elev', 'rel_elev', 'rel_elev_norm',
         'slope_c', 'lap_c', 'mean_slope', 'pct_rank_c']


def main():
    d = np.load('global_fem_dataset.npz')
    X, Y, lon, lat = d['X'], d['Y'], d['lon'], d['lat']
    c = Y.shape[-1] // 2
    target = Y[:, 0, c, c]          # 中心像素 B,与生产用法一致

    print(f'瓦片 {len(X):,},计算描述子...', flush=True)
    t0 = time.time()
    F = np.array([descriptors(X[i].squeeze()) for i in range(len(X))])
    print(f'  完成 ({time.time()-t0:.0f}s)', flush=True)

    # 与训练完全相同的空间划分
    km = KMeans(n_clusters=12, random_state=42, n_init='auto')
    labels = km.fit_predict(np.column_stack([lon, lat]))
    test_clusters = {0, 4, 8}
    te = np.isin(labels, list(test_clusters))
    tr = ~te
    print(f'空间划分: 训练 {tr.sum():,} / 测试 {te.sum():,}', flush=True)

    results = {}
    for name, model in [('Linear regression', LinearRegression()),
                        ('Gradient-boosted trees',
                         HistGradientBoostingRegressor(max_iter=400, random_state=42))]:
        model.fit(F[tr], target[tr])
        pred = model.predict(F[te])
        r2 = r2_score(target[te], pred)
        mae = mean_absolute_error(target[te], pred)
        results[name] = dict(r2=float(r2), mae=float(mae))
        print(f'{name:26s} R² = {r2:.4f}   MAE = {mae:.4f}', flush=True)

    # 按起伏分档,看差距集中在哪里
    absB1 = np.abs(target - 1)
    gb = HistGradientBoostingRegressor(max_iter=400, random_state=42).fit(F[tr], target[tr])
    pred = gb.predict(F[te])
    print('\n按 |B-1| 分档 (梯度提升树, 测试集):', flush=True)
    print(f'{"档位":<12}{"n":>7}{"R²":>10}{"MAE":>10}', flush=True)
    bands = [(0, 0.02, '近平坦'), (0.02, 0.1, '中等'), (0.1, 0.2, '高'), (0.2, 10, '极端')]
    band_stats = {}
    a_te = absB1[te]
    for lo, hi, lab in bands:
        m = (a_te >= lo) & (a_te < hi)
        if m.sum() < 10:
            continue
        r2b = r2_score(target[te][m], pred[m])
        maeb = mean_absolute_error(target[te][m], pred[m])
        band_stats[lab] = dict(n=int(m.sum()), r2=float(r2b), mae=float(maeb))
        print(f'{lab:<12}{int(m.sum()):>7}{r2b:>10.4f}{maeb:>10.4f}', flush=True)

    # 线性回归系数(看哪个描述子最重要)
    lr = LinearRegression().fit(F[tr], target[tr])
    print('\n线性回归标准化系数:', flush=True)
    sd = F[tr].std(axis=0)
    for n_, co in sorted(zip(NAMES, lr.coef_ * sd), key=lambda x: -abs(x[1])):
        print(f'  {n_:16s} {co:+.4f}', flush=True)

    json.dump(dict(overall=results, by_band=band_stats,
                   n_train=int(tr.sum()), n_test=int(te.sum())),
              open('terrain_regression_baseline.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('\n已保存 terrain_regression_baseline.json', flush=True)


if __name__ == '__main__':
    main()
