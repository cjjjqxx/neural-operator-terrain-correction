# -*- coding: utf-8 -*-
"""D11:蒙特卡洛 dropout 不确定度的校准检验。


本脚本在独立 500 瓦片集上做真正的校准检验(该集有有限元真值):
  1. 对每个瓦片跑 N 次 dropout 前向,得到中心像素的 std;
  2. 按预测 std 分箱,比较各箱的实际 RMSE 与平均预测 std —— 完美校准应落在 1:1 线;
  3. 报告 +-1 sigma 与 +-2 sigma 的经验覆盖率(理想值 68% 与 95%);
  4. 给出采样次数从 20 增到 100 时 std 估计本身的变化。
"""
import sys, os, json, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def enable_dropout(model):
    """推理时保持 dropout 激活,其余层保持 eval。"""
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-mc', type=int, default=100)
    ap.add_argument('--batch', type=int, default=64)
    a = ap.parse_args()

    import torch
    from step3_train_large import FNO3D

    v = np.load(os.path.join(HERE, 'validation_independent_500.npz'))
    dems, B_fem = v['dems'], v['B_fem']
    reliefs = v['reliefs']
    n = len(dems)
    c = B_fem.shape[1] // 2
    truth = B_fem[:, c, c]

    ck = torch.load(os.path.join(HERE, 'fno3d_large_model.pt'),
                    map_location='cuda', weights_only=False)
    model = FNO3D(modes=24, width=96, H_out=41, W_out=41, n_layers=6).cuda().eval()
    model.load_state_dict(ck['state'])
    enable_dropout(model)
    Xm, Xs, Ym, Ys = ck['X_mean'], ck['X_std'], ck['Y_mean'], ck['Y_std']

    print(f'独立集 {n} 瓦片,MC 采样 {a.n_mc} 次')
    draws = np.zeros((a.n_mc, n))
    x_all = (dems.reshape(-1, 1, 128, 128).astype(np.float32) - Xm) / Xs
    for k in range(a.n_mc):
        preds = []
        for i in range(0, n, a.batch):
            xb = torch.tensor(x_all[i:i + a.batch]).cuda()
            with torch.no_grad():
                pr = model(xb).cpu().numpy()[:, 0] * Ys + Ym
            preds.append(pr[:, c, c])
        draws[k] = np.concatenate(preds)
        if (k + 1) % 20 == 0:
            print(f'  {k+1}/{a.n_mc}', flush=True)

    mean_pred = draws.mean(0)
    std_pred = draws.std(0, ddof=1)
    err = mean_pred - truth
    out = {'n': int(n), 'n_mc': int(a.n_mc)}

    print(f'\n预测 std 中位 {np.median(std_pred):.5f}   实际 |误差| 中位 {np.median(np.abs(err)):.5f}')
    ratio = np.median(np.abs(err)) / max(np.median(std_pred), 1e-12)
    print(f'实际误差 / 预测 std = {ratio:.2f}   (>1 表示 dropout 低估了误差)')
    out['median_std_pred'] = float(np.median(std_pred))
    out['median_abs_err'] = float(np.median(np.abs(err)))
    out['err_over_std'] = float(ratio)

    print('\n=== 可靠性曲线(按预测 std 分箱) ===')
    print(f'{"箱":<6}{"n":>6}{"平均预测 std":>14}{"实际 RMSE":>12}{"比值":>9}')
    qs = np.quantile(std_pred, np.linspace(0, 1, 6))
    rel = []
    for i in range(5):
        m = (std_pred >= qs[i]) & (std_pred <= qs[i + 1] if i == 4 else std_pred < qs[i + 1])
        if m.sum() < 5:
            continue
        s_ = float(std_pred[m].mean()); r_ = float(np.sqrt((err[m] ** 2).mean()))
        rel.append(dict(bin=i + 1, n=int(m.sum()), mean_std=s_, rmse=r_, ratio=r_ / max(s_, 1e-12)))
        print(f'{i+1:<6}{m.sum():>6}{s_:>14.5f}{r_:>12.5f}{r_/max(s_,1e-12):>9.2f}')
    out['reliability'] = rel

    print('\n=== 经验覆盖率 ===')
    for k, ideal in [(1, 68.3), (2, 95.4)]:
        cov = 100 * (np.abs(err) <= k * std_pred).mean()
        out[f'coverage_{k}sigma'] = float(cov)
        print(f'  ±{k}σ 覆盖 {cov:>5.1f}%   (理想 {ideal}%)'
              f'   → {"区间过窄,低估不确定度" if cov < ideal - 5 else "大致合理" if cov < ideal + 5 else "区间过宽"}')

    print('\n=== 采样次数的影响 ===')
    for k in (20, 50, a.n_mc):
        if k <= a.n_mc:
            s_ = draws[:k].std(0, ddof=1)
            print(f'  N={k:>3}: std 中位 {np.median(s_):.5f}')
            out[f'median_std_N{k}'] = float(np.median(s_))

    print('\n=== 按起伏分档 ===')
    for lo, hi, nm in [(0, 100, '<100 m'), (100, 500, '100-500 m'), (500, 1e9, '>500 m')]:
        m = (reliefs >= lo) & (reliefs < hi)
        if m.sum() < 5:
            continue
        print(f'  {nm:<12} n={m.sum():>4}  预测 std {np.median(std_pred[m]):.5f}'
              f'   实际 RMSE {np.sqrt((err[m]**2).mean()):.5f}')

    json.dump(out, open(os.path.join(HERE, 'dropout_calibration_test.json'), 'w',
                        encoding='utf-8'), indent=2, ensure_ascii=False)
    print('\n已保存 dropout_calibration_test.json')


if __name__ == '__main__':
    main()
