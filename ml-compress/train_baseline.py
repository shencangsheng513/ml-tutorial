# -*- coding: utf-8 -*-
"""压缩第0课·备料：把 ml-vision 第5课那个 MNIST CNN 复现出来并存档成 fp32 基线。
为什么要有这一课：第5课是实验脚本，只画图不落盘（"跑过一次"和"随时能重跑"是两回事）。
压缩课需要一个明确的"压缩前"参照物，所以先补一个会存盘的训练。
配方一字不改：同种子、同 10000/10000 切分、同 20 epoch、同优化器——
跑完和复盘档案对账（第5课 CNN 0.9798）。**对不上就别往下走**：基线不可信，
后面所有压缩比、精度损失都是白算，这正是本仓库一贯的"先验证仪器再读数"。
用法：python train_baseline.py     # 纯 CPU，MNIST 已在 ../data（第5课下载过），零新依赖
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torchvision.datasets import MNIST

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
MODELS = HERE / "models"
ARCHIVE_ACC = 0.9798  # 复盘 §9 的档案值：同配方 CNN 测试准确率
N_TR, B, EP = 10000, 128, 20


def build_cnn() -> nn.Module:
    """第5课"选手2"原样：2 层卷积把 28×28 收到 7×7，再接两层全连接。"""
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))


def load_split() -> tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
    train_ds = MNIST(str(DATA), train=True, download=True)
    test_ds = MNIST(str(DATA), train=False, download=True)
    rng = np.random.default_rng(42)           # 第5课同款：固定种子抽 10000 张
    idx = rng.choice(len(train_ds), N_TR, replace=False)
    xtr = train_ds.data[idx].numpy().astype(np.float32) / 255.0
    ytr = np.array(train_ds.targets)[idx]
    xte = test_ds.data.numpy().astype(np.float32) / 255.0
    yte = np.array(test_ds.targets)
    return (torch.tensor(xtr).unsqueeze(1), ytr,
            torch.tensor(xte).unsqueeze(1), yte)


def main() -> None:
    torch.manual_seed(42)
    xt, yt, xe, yte = load_split()
    model = build_cnn()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    n = sum(p.numel() for p in model.parameters())
    print(f"基线 CNN 参数量 {n/1e3:.0f}k | 训练 {N_TR} 张 | 测试 {len(yte)} 张 | {EP} epochs")
    t0 = time.time()
    for ep in range(EP):
        model.train()
        perm = torch.randperm(N_TR)
        for i in range(0, N_TR, B):
            k = perm[i:i + B]
            opt.zero_grad()
            loss = lossf(model(xt[k]), torch.tensor(yt[k]))
            loss.backward()
            opt.step()
        if (ep + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                a = accuracy_score(yte[:2000], model(xe[:2000]).argmax(1).numpy())
            print(f"  epoch {ep+1}: 子集验证 {a:.4f}  已用 {time.time()-t0:.0f}s")

    model.eval()
    with torch.no_grad():
        acc = accuracy_score(yte, model(xe).argmax(1).numpy())
    drift = abs(acc - ARCHIVE_ACC)
    print(f"\n测试准确率 {acc:.4f}（档案 {ARCHIVE_ACC:.4f}，差 {drift*100:+.2f}pp，{time.time()-t0:.0f}s）")
    if drift > 0.01:
        sys.exit(f"与档案差 {drift:.4f} > 1pp：基线不可信，先查配方再谈压缩")

    MODELS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODELS / "mnist_cnn_fp32.pt")
    (MODELS / "baseline_report.json").write_text(json.dumps({
        "arch": "2conv(16,32)+fc(64)+fc(10)，28x28 灰度 /255",
        "recipe": "seed42 / 10000 train / 10000 test / Adam 1e-3 / bs128 / 20ep（= ml-vision 第5课）",
        "n_params": n, "test_acc": round(acc, 4), "archive_acc": ARCHIVE_ACC,
        "acc_drift_pp": round(drift * 100, 2),
        "size_kb_fp32": round((MODELS / "mnist_cnn_fp32.pt").stat().st_size / 1024, 1),
        "torch": torch.__version__, "train_seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已存 models/mnist_cnn_fp32.pt + models/baseline_report.json")


if __name__ == "__main__":
    main()
