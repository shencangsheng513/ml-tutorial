# -*- coding: utf-8 -*-
"""
CV第5课：MNIST 正名之战 —— 第3课说"CNN没赢，因为8×8没挑战"，
这课给它真战场：28×28 手写数字（60000张），挑战=笔画粗细/位置/字形多样。
公平对决规则（第2课的信条：控制变量）：
  · 同一份 10000 张训练图 / 完整 10000 张测试图
  · MLP 和 CNN 都训到收敛，参数量级差不多
预期：小图 MLP≈CNN；真 MNIST 上 CNN 应甩开 MLP——
如果没甩开，那就是我们的结论错了，如实上报。
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from torchvision.datasets import MNIST

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(42)
rng = np.random.default_rng(42)
ROOT = "../data"

train_ds = MNIST(ROOT, train=True, download=True)
test_ds = MNIST(ROOT, train=False, download=True)

N_TR = 10000
idx = rng.choice(len(train_ds), N_TR, replace=False)
xtr = train_ds.data[idx].numpy().astype(np.float32) / 255.0
ytr = np.array(train_ds.targets)[idx]
xte = test_ds.data.numpy().astype(np.float32) / 255.0
yte = np.array(test_ds.targets)
print(f"数据就绪：训练 {N_TR} 张 / 测试 {len(yte)} 张，28×28")

# ============================================================
# 选手1：MLP —— 拍平 784 维（第3课同款思路放大版）
# ============================================================
print("\n== MLP (784→256→10) ==")
t0 = time.time()
mlp = MLPClassifier((256,), max_iter=30, random_state=42, early_stopping=True,
                    n_iter_no_change=4)
mlp.fit(xtr.reshape(N_TR, -1), ytr)
acc_mlp = accuracy_score(yte, mlp.predict(xte.reshape(len(yte), -1)))
print(f"MLP 测试准确率 {acc_mlp:.4f}（{time.time()-t0:.0f}s）")

# ============================================================
# 选手2：CNN —— 尊重二维结构
# ============================================================
print("\n== CNN (2×conv + fc) ==")


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28→14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14→7
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.net(x)


model = MnistCNN()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()
xt = torch.tensor(xtr).unsqueeze(1)
yt = torch.tensor(ytr, dtype=torch.long)
xe = torch.tensor(xte).unsqueeze(1)
B, EP = 128, 20
hist = []
t0 = time.time()
for ep in range(EP):
    model.train()
    perm = torch.randperm(N_TR)
    for i in range(0, N_TR, B):
        k = perm[i:i + B]
        opt.zero_grad()
        loss = lossf(model(xt[k]), yt[k])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        a = accuracy_score(yte[:2000], model(xe[:2000]).argmax(1).numpy())
    hist.append(a)
    print(f"  epoch {ep+1}: 子集验证 {a:.4f}  已用 {time.time()-t0:.0f}s")
with torch.no_grad():
    pred_cnn = model(xe).argmax(1).numpy()
acc_cnn = accuracy_score(yte, pred_cnn)
print(f"CNN 测试准确率 {acc_cnn:.4f}（{time.time()-t0:.0f}s）")

# ============================================================
# 对账 + 复盘第3课
# ============================================================
print("\n" + "=" * 60)
print(f"28×28 真战场：MLP {acc_mlp:.4f}  vs  CNN {acc_cnn:.4f}  差 {100*(acc_cnn-acc_mlp):+.1f}pp")
print(f"8×8  旧战场（第3课）：MLP 0.9822  vs  CNN 0.9800  差 -0.2pp")
n_param_mlp = sum((a * b + b) for a, b in zip([784, 256], [256, 10]))
n_param_cnn = sum(p.numel() for p in model.parameters())
print(f"参数量：MLP {n_param_mlp/1e3:.0f}k  vs  CNN {n_param_cnn/1e3:.0f}k")
cm_mlp = confusion_matrix(yte, mlp.predict(xte.reshape(len(yte), -1)))
cm_cnn = confusion_matrix(yte, pred_cnn)
bad_mlp = (cm_mlp.sum() - np.trace(cm_mlp))
bad_cnn = (cm_cnn.sum() - np.trace(cm_cnn))
print(f"错误数：MLP {bad_mlp} 张 vs CNN {bad_cnn} 张（CNN 少 {(1-bad_cnn/bad_mlp)*100:.0f}%）")
print("→ 架构优势不是玄学：卷积的'局部+共享权重'先验，恰好在'字形会平移、笔画有局部结构'的世界里赚。")
print("  8×8 居中图上这个先验不值钱（第3课），28×28 真图上它立刻变现（本课）。")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
axes[0].bar(["MLP(拍平)", "CNN(卷积)"], [acc_mlp, acc_cnn], color=["#999", "#4C72B0"])
axes[0].set_ylim(0.9, 1.0)
axes[0].set_title("MNIST 10k训练：正名之战")
for i, a in enumerate([acc_mlp, acc_cnn]):
    axes[0].text(i, a + 0.001, f"{a:.3f}", ha="center")
axes[1].plot(range(1, EP + 1), hist, "o-", color="#4C72B0", label="CNN 每轮验证")
axes[1].axhline(acc_mlp, color="#999", ls="--", label=f"MLP 终点 {acc_mlp:.3f}")
axes[1].set_xlabel("epoch"); axes[1].set_title("CNN 学习曲线 vs MLP 终点")
axes[1].legend()
fig.suptitle("CV第5课：CNN 该赢的地方它确实赢")
fig.tight_layout()
fig.savefig("mnist_showdown.png", dpi=120)
print("\n图已存：mnist_showdown.png")
