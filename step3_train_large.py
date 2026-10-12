#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_train_large.py — 大模型FNO-3D训练脚本 (服务器版)

用法:
  # 默认配置 (~1.5M 参数)
  python step3_train_large.py --data global_fem_dataset.npz

  # 自定义模型规模
  python step3_train_large.py \
    --data global_fem_dataset.npz \
    --modes 32 --width 128 \
    --epochs 1200 --batch 16

  # 空间分区训练/测试 (避免spatial leakage)
  python step3_train_large.py \
    --data global_fem_dataset.npz \
    --spatial-split

输出:
  fno3d_large_model.pt     — 模型权重+归一化参数
  fno3d_large_metrics.json — 训练指标
  figs/                      — 训练曲线+预测散点图
"""

import sys, io, os, time, json, argparse
# reconfigure 防止nohup重定向时 stdout 块缓冲 (同时需 PYTHONIOENCODING=utf-8)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# FNO-3D 模型 (可学习复数权重 + Pearson loss)
# ══════════════════════════════════════════════════════════════
class FNO3D(nn.Module):
    """
    Fourier Neural Operator — 3D地形 → B场映射
    6层谱卷积, 每层: rfft2 → 截断高频 → 可学习复数权重 → irfft2 + 残差
    """
    def __init__(self, modes=24, width=96, H_out=21, W_out=21, n_layers=6):
        super().__init__()
        self.modes = modes
        self.H_out, self.W_out = H_out, W_out
        self.n_layers = n_layers

        # 通道提升: 1 → width
        self.fc0 = nn.Linear(1, width)

        # 可学习复数权重 (每层) — 极小初始化
        scale = 1.0 / (width * width * modes)
        self.weights = nn.ParameterList([
            nn.Parameter(scale * torch.randn(width, width, modes, modes,
                                             dtype=torch.cfloat))
            for _ in range(n_layers)
        ])

        # 谱域后的1×1卷积 + 残差投影
        self.conv = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])
        self.w = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])

        # GroupNorm 稳定每层FFT→spatial后的幅度, 防止复数权重累积爆炸
        self.norm = nn.ModuleList([
            nn.GroupNorm(min(32, width), width) for _ in range(n_layers)
        ])

        # 通道压缩: width → 64 → 1
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.drop = nn.Dropout(0.05)

        self.gelu = nn.GELU()

    def forward(self, x):
        # x: [B, 1, H, W] → [B, width, H, W]
        x = x.permute(0, 2, 3, 1)   # [B, H, W, 1]
        x = self.fc0(x)              # [B, H, W, width]
        x = x.permute(0, 3, 1, 2)   # [B, width, H, W]

        for i in range(self.n_layers):
            x_in = x
            # 谱域卷积
            xf = torch.fft.rfft2(x_in, norm='ortho')
            mh = min(self.modes, xf.shape[-2])
            mw = min(self.modes, xf.shape[-1])
            xf[:, :, :mh, :mw] = torch.einsum(
                'bixy,ioxy->boxy',
                xf[:, :, :mh, :mw],
                self.weights[i][:, :, :mh, :mw],
            )
            xf_out = torch.fft.irfft2(xf, s=(x_in.shape[-2], x_in.shape[-1]),
                                       norm='ortho')
            # 1×1 conv + GroupNorm + 残差 (GroupNorm防止Fourier域幅度累积)
            x = xf_out + self.conv[i](x_in)
            x = self.gelu(self.norm[i](x))
            x = self.drop(x)
            x = self.w[i](x) + x_in   # 残差连接

        # 自适应池化到输出尺寸
        x = F.adaptive_avg_pool2d(x, (self.H_out, self.W_out))

        # 通道压缩
        x = x.permute(0, 2, 3, 1)   # [B, H_out, W_out, width]
        x = self.gelu(self.fc1(x))   # [B, H_out, W_out, 128]
        x = self.fc2(x)              # [B, H_out, W_out, 1]
        x = x.permute(0, 3, 1, 2)   # [B, 1, H_out, W_out]
        return x


class D4Average(nn.Module):
    """推理时对二面体群 D4 的 8 个朝向取平均(不重训、不改权重)。

    物理问题对瓦片的旋转与翻转对称,但上面的谱卷积只对实傅里叶谱的一个象限加权,
    网络本身不等变:同一块地形换个朝向,中心像元预测的变化与算子误差同量级
    (operator_flip_symmetry_test.json)。把输入依次变换到 8 个朝向、预测后逆变换回来
    再平均,输出即严格满足这一对称性。128→41 的自适应平均池化在翻转下对称,故输出网格
    的逆变换是精确的(fem_orientation_test.py 已按同一方式验证)。
    所有生产推理(测点校正、全球场、发布脚本、独立集评价)一律经过它。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    @staticmethod
    def _fwd(x, k):
        if k % 2:
            x = torch.flip(x, dims=[-2])
        return torch.rot90(x, k // 2, dims=[-2, -1])

    @staticmethod
    def _inv(y, k):
        y = torch.rot90(y, -(k // 2), dims=[-2, -1])
        if k % 2:
            y = torch.flip(y, dims=[-2])
        return y

    def forward(self, x):
        return sum(self._inv(self.model(self._fwd(x, k)), k) for k in range(8)) / 8.0


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════
def pearson_corr(pred, target):
    """批次平均Pearson相关系数"""
    pm = pred - pred.mean(dim=(1, 2, 3), keepdim=True)
    tm = target - target.mean(dim=(1, 2, 3), keepdim=True)
    num = (pm * tm).sum(dim=(1, 2, 3))
    den = (pm.pow(2).sum(dim=(1, 2, 3)).sqrt()
           * tm.pow(2).sum(dim=(1, 2, 3)).sqrt() + 1e-8)
    return (num / den).mean()


def compute_r2(pred_np, target_np):
    """计算 R² (numpy)"""
    yt = target_np.ravel()
    yp = pred_np.ravel()
    ssr = np.sum((yt - yp) ** 2)
    sst = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ssr / sst) if sst > 1e-12 else 0.0


def make_plots(hist, pred_te, Y_te_np, out_dir='figs'):
    """生成训练曲线和预测散点图"""
    os.makedirs(out_dir, exist_ok=True)

    # ── 训练曲线 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    epochs = [h['epoch'] for h in hist]
    ax1.plot(epochs, [h['train_loss'] for h in hist],
             color='#E74C3C', lw=1.2, alpha=0.7, label='Train')
    ax1.plot(epochs, [h['val_loss'] for h in hist],
             color='#2471A3', lw=2, label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve'); ax1.legend(frameon=False)
    ax1.grid(True, ls=':', lw=0.5, color='#ddd')

    ax2.plot(epochs, [h['r2'] for h in hist],
             color='#27AE60', lw=2)
    ax2.axhline(y=0, color='#999', ls='--', lw=0.8)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Test R²')
    ax2.set_title(f"Test R² (best = {max(h['r2'] for h in hist):.4f})")
    ax2.grid(True, ls=':', lw=0.5, color='#ddd')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'training_curve.png'), dpi=200)
    plt.close(fig)

    # ── 预测散点图 ──
    fig, ax = plt.subplots(figsize=(5.5, 5))
    yt = Y_te_np.ravel()
    yp = pred_te.ravel()
    r2 = compute_r2(pred_te, Y_te_np)

    # 下采样显示 (太多点看不清)
    n_show = min(5000, len(yt))
    idx = np.random.choice(len(yt), n_show, replace=False)

    ax.scatter(yt[idx], yp[idx], s=2, alpha=0.35, c='#2471A3',
               edgecolors='none', rasterized=True)
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], '--', color='#E74C3C', lw=1.2, label='y=x')
    ax.set_xlabel('True B'); ax.set_ylabel('Predicted B')
    ax.set_title(f'FNO-3D Prediction (R²={r2:.4f})')
    ax.legend(frameon=False)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'scatter.png'), dpi=200)
    plt.close(fig)

    return r2


# ══════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='FNO-3D 大模型训练')
    parser.add_argument('--data', default='global_fem_dataset.npz',
                        help='数据集 .npz 路径')
    parser.add_argument('--modes', type=int, default=24,
                        help='Fourier modes (default: 24)')
    parser.add_argument('--width', type=int, default=96,
                        help='隐藏层宽度 (default: 96)')
    parser.add_argument('--layers', type=int, default=6,
                        help='谱卷积层数 (default: 6)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='训练轮数 (default: 1000)')
    parser.add_argument('--batch', type=int, default=4,
                        help='批次大小 (default: 4, 8GB卡安全)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='学习率 (default: 5e-5, 新数据集+GroupNorm内置)')
    parser.add_argument('--corr-weight', type=float, default=0.1,
                        help='Pearson loss 权重 (default: 0.1)')
    parser.add_argument('--spatial-split', action='store_true',
                        help='按经纬度空间分区 (避免spatial leakage)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='测试集比例 (default: 0.15)')
    parser.add_argument('--model-out', default='fno3d_large_model.pt',
                        help='模型输出路径')
    parser.add_argument('--metrics-out', default='fno3d_large_metrics.json',
                        help='指标输出路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--ckpt', default='training_checkpoint.pt',
                        help='训练断点文件路径 (default: training_checkpoint.pt)')
    parser.add_argument('--ckpt-every', type=int, default=5,
                        help='每多少个 epoch 保存一次断点 (default: 5,断电最多丢约 23 min)')
    parser.add_argument('--fresh', action='store_true',
                        help='忽略已有断点, 从头开始训练')
    parser.add_argument('--force-resume', action='store_true',
                        help='断点与当前数据/参数不一致时仍强行续训(危险,会污染结果)')
    parser.add_argument('--patience', type=int, default=80,
                        help='EarlyStopping + ReduceLROnPlateau 耐心值 (default: 80)')
    parser.add_argument('--lr-factor', type=float, default=0.5,
                        help='Plateau时LR衰减因子 (default: 0.5)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='最低学习率 (default: 1e-6)')
    parser.add_argument('--test-clusters', type=str, default='0,4,8',
                        help='逗号分隔的k-means聚类编号(0-11), 作为空间留出测试集 (default: 0,4,8)')
    args = parser.parse_args()

    # ── 设备 ──
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    else:
        device = torch.device('cpu')
        print('WARNING: 无GPU, 使用CPU训练 (会很慢!)')
    print(f'Device: {device}')

    # ── 加载数据 ──
    print(f'\n加载数据: {args.data}')
    d = np.load(args.data)
    X_raw = d['X'].astype(np.float32)
    Y_raw = d['Y'].astype(np.float32)
    n_total = len(X_raw)
    print(f'  数据量: {n_total} tiles')
    print(f'  X shape: {X_raw.shape}, Y shape: {Y_raw.shape}')
    print(f'  B range: [{Y_raw.min():.4f}, {Y_raw.max():.4f}]')
    print(f'  B mean: {Y_raw.mean():.4f}, std: {Y_raw.std():.4f}')

    X_mean = X_raw.mean()
    X_std = X_raw.std() + 1e-8
    Y_mean = Y_raw.mean()
    Y_std = Y_raw.std() + 1e-8
    Y_clip = np.clip(Y_raw, 0.05, 5.0)
    X = (X_raw - X_mean) / X_std
    Y = (Y_clip - Y_mean) / Y_std

    # ── 训练/测试划分 ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.spatial_split:
        from sklearn.cluster import KMeans
        coords = np.stack([d['lon'], d['lat']], axis=1)
        # 10个地理cluster, 选3个作为独立测试集(~30%)
        km = KMeans(n_clusters=12, random_state=args.seed, n_init='auto')
        labels = km.fit_predict(coords)
        test_clusters = set(int(x) for x in args.test_clusters.split(','))
        i_te = np.where(np.isin(labels, list(test_clusters)))[0]
        i_tr = np.where(~np.isin(labels, list(test_clusters)))[0]
        print(f'空间分区 (k-means 12 clusters): test clusters={test_clusters}')
        print(f'Train: {len(i_tr)}, Test: {len(i_te)}')
    else:
        n_test = int(n_total * args.test_ratio)
        perm = np.random.permutation(n_total)
        i_tr, i_te = perm[n_test:], perm[:n_test]

    print(f'Train: {len(i_tr)}, Test: {len(i_te)}')

    X_tr = torch.tensor(X[i_tr], device=device)
    Y_tr = torch.tensor(Y[i_tr], device=device)
    X_te = torch.tensor(X[i_te], device=device)
    Y_te = torch.tensor(Y[i_te], device=device)

    # ── 构建模型 ──
    H_out = Y_raw.shape[2]
    W_out = Y_raw.shape[3]
    model = FNO3D(
        modes=args.modes, width=args.width,
        n_layers=args.layers,
        H_out=H_out, W_out=W_out,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n模型:')
    print(f'  modes={args.modes}, width={args.width}, layers={args.layers}')
    print(f'  总参数: {n_params:,}')
    print(f'  可训练参数: {n_trainable:,}')

    # ── 优化器 ──
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=3e-5,
        betas=(0.9, 0.999),
    )
    # ReduceLROnPlateau: R²不涨才降LR, 避免余弦退火的单调衰减过拟合
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=args.lr_factor,
        patience=args.patience // 2, min_lr=args.min_lr,
        threshold=1e-4, threshold_mode='rel',
    )

    # ── 训练 ──
    BATCH = args.batch
    hist = []
    best_r2 = -999
    best_state = None
    t0 = time.time()
    grad_clip = 0.1  # 收紧裁剪: width=96+空间分区需更保守
    start_epoch = 1
    epochs_no_improve = 0  # EarlyStopping计数器

    FP = {
        'data': os.path.basename(args.data),
        'n_total': int(n_total),
        'y_shape': tuple(int(v) for v in Y_raw.shape),
        'y_min': round(float(Y_raw.min()), 6),
        'y_max': round(float(Y_raw.max()), 6),
        'y_mean': round(float(Y_raw.mean()), 6),
        'y_std': round(float(Y_raw.std()), 6),
        'n_train': int(len(i_tr)),
        'n_test': int(len(i_te)),
    }

    def save_checkpoint(ep):
        ck = {
            'epoch': ep,
            'model_state': model.state_dict(),
            'opt_state': opt.state_dict(),
            'sch_state': sch.state_dict(),
            'hist': hist,
            'best_r2': best_r2,
            'best_state': best_state,
            'elapsed': time.time() - t0,
            'seed': args.seed,
            'epochs': args.epochs,
            'epochs_no_improve': epochs_no_improve,
            'fingerprint': FP,
        }
        tmp = args.ckpt + '.tmp'
        torch.save(ck, tmp)
        os.replace(tmp, args.ckpt)  # 原子替换, 断电也不会损坏旧断点

    # ── 断点续跑 ──
    if (not args.fresh) and os.path.exists(args.ckpt):
        print(f'\n检测到断点文件: {args.ckpt}, 恢复训练...')
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)

        problems = []
        old_fp = ck.get('fingerprint')
        if old_fp is None:
            problems.append('断点没有数据集指纹(旧格式),无法确认它是哪一代数据训出来的')
        elif old_fp != FP:
            for k in sorted(set(old_fp) | set(FP)):
                if old_fp.get(k) != FP.get(k):
                    problems.append(f'{k}: 断点={old_fp.get(k)}  当前={FP.get(k)}')
        if ck.get('seed') != args.seed:
            problems.append(f'seed: 断点={ck.get("seed")}  当前={args.seed}')
        if ck.get('epochs') != args.epochs:
            problems.append(f'epochs: 断点={ck.get("epochs")}  当前={args.epochs}')

        if problems and not args.force_resume:
            print('\n' + '=' * 66)
            print('✗ 断点与当前训练设置不是同一代,拒绝续训:')
            for p_ in problems:
                print(f'    · {p_}')
            print('\n  跨数据集续训会污染结果:模型带着上一代数据学到的权重,')
            print('  在换数据的当轮记下一个虚高的 R² 并被存为最佳权重,')
            print('  此后再也超不过它 —— 这不是收敛值,是换数据的跳变。')
            print('  (2026-08-07 与 08-18 已因此各毁掉一次训练)')
            print('\n  正确做法:加 --fresh 从头训练。')
            print('  确知无害要强行续训,再加 --force-resume。')
            print('=' * 66)
            sys.exit(2)
        if problems:
            print('\n⚠ --force-resume:已知不一致但仍继续,后果自负:')
            for p_ in problems:
                print(f'    · {p_}')
        model.load_state_dict(ck['model_state'])
        opt.load_state_dict(ck['opt_state'])
        # 兼容旧checkpoint的CosineAnnealingLR: 加载失败则用新的ReduceLROnPlateau
        try:
            sch.load_state_dict(ck['sch_state'])
        except (KeyError, TypeError, ValueError):
            print('  (scheduler类型已从CosineAnnealingLR切换到ReduceLROnPlateau)')
        hist = ck['hist']
        best_r2 = ck['best_r2']
        best_state = ck['best_state']
        start_epoch = ck['epoch'] + 1
        t0 = time.time() - ck['elapsed']  # 累计上次已用时间
        epochs_no_improve = ck.get('epochs_no_improve', 0)
        print(f'  已完成 {ck["epoch"]} epochs, 从 epoch {start_epoch} 继续, '
              f'当前最佳 R² = {best_r2:.4f}')
    elif args.fresh and os.path.exists(args.ckpt):
        print(f'\n--fresh: 忽略已有断点 {args.ckpt}, 从头训练')

    print(f'\n开始训练 ({args.epochs} epochs, batch={BATCH})...')
    print(f'{"Epoch":>6s}  {"TrainLoss":>10s}  {"ValLoss":>10s}  '
          f'{"R²":>8s}  {"BestR²":>8s}  {"LR":>10s}')

    for ep in range(start_epoch, args.epochs + 1):
            # ── 训练阶段 ──
            model.train()
            train_losses = []
            prm = torch.randperm(len(X_tr), device=device)

            for k in range(0, len(X_tr), BATCH):
                bi = prm[k:k + BATCH]
                pred = model(X_tr[bi])

                # MSE
                mse = ((pred - Y_tr[bi]) ** 2).mean()

                # Pearson correlation 辅助损失
                if args.corr_weight > 0:
                    corr = pearson_corr(pred, Y_tr[bi])
                    loss = mse + args.corr_weight * (1 - corr)
                else:
                    loss = mse

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                train_losses.append(loss.item())

                # NaN 检测: 保存断点并退出, 用户降低LR后重跑
                if not np.isfinite(loss.item()):
                    print(f'\n[ERROR] NaN detected at epoch {ep}, step {k//BATCH}!', flush=True)
                    save_checkpoint(ep - 1)  # 保存上一个健康的epoch
                    recommended_lr = opt.param_groups[0]['lr'] * 0.5
                    print(f'训练因NaN终止。断点已保存至epoch {ep-1}.')
                    print(f'建议: --lr {recommended_lr:.1e} --fresh 重跑', flush=True)
                    return

            train_loss = np.mean(train_losses)

            # ── 验证阶段 (分批避免OOM) ──
            model.eval()
            val_bs = max(2, BATCH)
            preds_te, losses_te = [], []
            with torch.no_grad():
                for bj in range(0, len(X_te), val_bs):
                    Xb = X_te[bj:bj + val_bs]
                    Yb = Y_te[bj:bj + val_bs]
                    pb = model(Xb)
                    preds_te.append(pb)
                    losses_te.append(((pb - Yb) ** 2).mean().item())
                # 合并回完整预测用于R²
                pred_te = torch.cat(preds_te, dim=0)
                val_mse = float(np.mean(losses_te))
                yt = Y_te.cpu().numpy().ravel()
                yp = pred_te.cpu().numpy().ravel()
                r2 = compute_r2(pred_te.cpu().numpy(), Y_te.cpu().numpy())

            hist.append({
                'epoch': ep,
                'train_loss': train_loss,
                'val_loss': val_mse,
                'r2': r2,
                'lr': opt.param_groups[0]['lr'],
            })

            # Plateau调度: R²不涨才降LR (在验证后调用)
            sch.step(r2)

            improved = False
            if r2 > best_r2:
                best_r2 = r2
                improved = True
                best_state = {
                    'state': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    'X_mean': X_mean, 'X_std': X_std,
                    'Y_mean': Y_mean, 'Y_std': Y_std,
                    'r2': best_r2, 'n_params': n_params,
                    'args': vars(args),
                    'history': hist,
                }

            # 打印进度
            if ep % max(1, args.epochs // 30) == 0 or ep == 1:
                lr_now = opt.param_groups[0]['lr']
                marker = ' *' if improved else ''
                print(f'{ep:>6d}  {train_loss:>10.4f}  {val_mse:>10.4f}  '
                      f'{r2:>8.4f}  {best_r2:>8.4f}  {lr_now:>10.2e}{marker}')

            # 周期保存断点 (Ctrl+C / 关机后可续跑)
            if ep % args.ckpt_every == 0 or ep == args.epochs:
                save_checkpoint(ep)

            # EarlyStopping: best_r2 连续 patience 轮不涨则自动停止
            if improved:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                print(f'\nEarlyStopping: R² 连续 {args.patience} epoch 未超过 {best_r2:.4f}, 停止训练')
                save_checkpoint(ep)
                break

    elapsed = time.time() - t0
    print(f'\n训练完成: {elapsed:.0f}s ({elapsed/60:.1f}min)')
    print(f'最佳 R² = {best_r2:.4f}')

    # ── 保存模型 ──
    torch.save(best_state, args.model_out)
    size_mb = os.path.getsize(args.model_out) / 1e6
    print(f'模型已保存: {args.model_out} ({size_mb:.1f} MB)')

    # ── 保存指标 ──
    metrics = {
        'r2_test': best_r2,
        'n_train': int(len(i_tr)),
        'n_test': int(len(i_te)),
        'n_total': n_total,
        'n_params': n_params,
        'modes': args.modes,
        'width': args.width,
        'layers': args.layers,
        'epochs': args.epochs,
        'train_time_s': elapsed,
    }
    json.dump(metrics, open(args.metrics_out, 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print(f'指标已保存: {args.metrics_out}')

    model.load_state_dict(best_state['state'])
    model.eval()
    val_bs = max(2, BATCH)
    preds = []
    with torch.no_grad():
        for bj in range(0, len(X_te), val_bs):
            preds.append(model(X_te[bj:bj + val_bs]).cpu().numpy())
    pred_te_final = np.concatenate(preds, axis=0)
    Y_te_final = Y_te.cpu().numpy()
    final_r2 = make_plots(hist, pred_te_final, Y_te_final)
    print(f'图片已保存: figs/training_curve.png, figs/scatter.png')
    print(f'\nDone! Final Test R² = {final_r2:.4f}')


if __name__ == '__main__':
    main()
