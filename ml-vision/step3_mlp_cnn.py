# -*- coding: utf-8 -*-
"""
CV第3课：神经网络正主登场 —— 全连接MLP vs 卷积CNN，外加数据增强
第2课删角实验留了个问题：真实大图没法手动挑像素，谁来发现"哪里重要"？
答案：卷积网络。它的'卷积'就是可学习的局部小放大镜。
本课三件事：
  ① MLP（把像素拍平喂网络=第2课思路的神经网络版）
  ② CNN（尊重 8×8 的二维结构）—— 看它第一层自己学出了什么"滤镜"
  ③ 数据增强：只给 200 张训练图，看"人工制造样本"能不能救回来
     （呼应项目1的"数据不够特征凑"，这里是"数据不够变换凑"）
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(42)
rng = np.random.default_rng(42)

d = load_digits()
X, y = d.data / 16.0, d.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
Xtr_i = Xtr.reshape(-1, 8, 8)
Xte_i = Xte.reshape(-1, 8, 8)

# ============================================================
# 第1步：MLP —— 把图拍平（和第2课 LR 同一输入，只是模型更聪明）
# ============================================================
print("=" * 60)
print("第1步：MLP 全连接 (64→64→10)")
print("=" * 60)
mlp = MLPClassifier((64,), max_iter=2000, random_state=42)
mlp.fit(Xtr, ytr)
acc_mlp = accuracy_score(yte, mlp.predict(Xte))
print(f"测试准确率 {acc_mlp:.4f}")

# ============================================================
# 第2步：CNN —— 尊重二维结构
# ============================================================
print("\n" + "=" * 60)
print("第2步：小 CNN（两层 3×3 卷积 + 两个全连接层）")
print("=" * 60)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 8x8→4x4
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 4x4→2x2
            nn.Flatten(), nn.Linear(16 * 2 * 2, 10))

    def forward(self, x):
        return self.net(x)


def train_cnn(xtr, ytr_, epochs=80, aug_shift=None, lr=3e-3):
    """返回 (训练好的模型, 训练曲线, 验证曲线)"""
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    xt = torch.tensor(xtr, dtype=torch.float32).unsqueeze(1)   # (N,1,8,8)
    yt = torch.tensor(ytr_, dtype=torch.long)
    xe = torch.tensor(Xte_i, dtype=torch.float32).unsqueeze(1)
    ye = torch.tensor(yte, dtype=torch.long)
    hist_tr, hist_va = [], []
    B = 64
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), B):
            idx = perm[i:i + B]
            xb, yb = xt[idx], yt[idx]
            if aug_shift:                     # 随机 ±aug_shift 像素平移（补零）
                xb = torch.stack([torch.tensor(shift_aug(xb[k].numpy(), aug_shift),
                                                dtype=torch.float32).unsqueeze(0)
                                  for k in range(len(xb))])
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            hist_tr.append(float(lossf(model(xt), yt)))
            hist_va.append(float(accuracy_score(ye, model(xe).argmax(1))))
    return model, hist_tr, hist_va


def shift_aug(img, max_shift=1):
    """把图垫边再随机裁回 8x8 = 随机平移至多 ±max_shift 像素"""
    p = max_shift
    pad = np.zeros((8 + 2 * p, 8 + 2 * p))
    pad[p:p + 8, p:p + 8] = img[0] if img.ndim == 3 else img
    r, c = rng.integers(0, 2 * p + 1, 2)
    return pad[r:r + 8, c:c + 8]


model_cnn, tr1, va1 = train_cnn(Xtr_i, ytr)
print(f"CNN 测试准确率 {va1[-1]:.4f}（最后一轮验证集）")

# ============================================================
# 第3步：CNN 第一层学到了什么滤镜？（可解释性彩蛋）
# ============================================================
w = model_cnn.net[0].weight.detach().numpy()   # (8,1,3,3)
fig, axes = plt.subplots(2, 4, figsize=(8, 4.2))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(w[i, 0], cmap="coolwarm", vmin=-abs(w).max(), vmax=abs(w).max())
    ax.set_title(f"滤镜{i}", fontsize=9)
    ax.axis("off")
fig.suptitle("CNN 第一层自学出的 8 个 3×3 滤镜（红=看到就兴奋）")
fig.tight_layout()
fig.savefig("cnn_filters.png", dpi=120)
print("\n第一层滤镜已存图：cnn_filters.png —— 没人告诉它'看边缘'，它自己学会了")

# ============================================================
# 第4步：数据饥渴实验 —— 只给 200 张，增强真能"救回来"吗？
# ============================================================
print("\n" + "=" * 60)
print("第4步：200张训练图，三臂对比（同预算240轮）")
print("=" * 60)
small = rng.choice(len(Xtr_i), 200, replace=False)
_, trA, vaA = train_cnn(Xtr_i[small], ytr[small], epochs=240)
_, trB, vaB = train_cnn(Xtr_i[small], ytr[small], epochs=240, aug_shift=1)
_, trC, vaC = train_cnn(Xtr_i[small], ytr[small], epochs=240, aug_shift=2)
print(f"  200张 无增强      测试 {vaA[-1]:.4f}")
print(f"  200张 ±1px平移    测试 {vaB[-1]:.4f}")
print(f"  200张 ±2px平移    测试 {vaC[-1]:.4f}")
print(f"  1347张 无增强     测试 {va1[-1]:.4f}（参考：全量数据）")
print("→ 预期被打破：增强不但没救回来，反而更差！为什么？")
print("  这套数据的测试图全部'居中'，而平移增强把训练分布推歪了——")
print("  我们亲手制造了项目1说过的 train/serve skew（训练=歪图，上线=正图）。")
print("  增强的铁律：只增强'真实世界会有的变化'。真实扫描/MNIST照片确实")
print("  不居中，那时平移增强就是纯收益；这里不会，它就是纯毒药。")
print("  '增强=万能药'是教程圈最大的神话之一，本课用数据戳破了它。")

# ============================================================
# 画图：训练曲线 + 成绩单
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].plot(vaA, label="200张 无增强", color="#d9534f")
axes[0].plot(vaB, label="200张 ±1px增强", color="#5cb85c")
axes[0].plot(vaC, label="200张 ±2px增强", color="#8172B3")
axes[0].plot(va1, label="1347张 无增强", color="#4C72B0", ls="--")
axes[0].set_xlabel("轮数")
axes[0].set_ylabel("测试集准确率")
axes[0].set_title("三臂学习曲线：增强在这份数据上是毒药")
axes[0].legend(fontsize=8)

names = ["MLP(拍平,全量)", "CNN(全量)", "CNN(200张)", "CNN(200+±1px)", "CNN(200+±2px)"]
accs = [acc_mlp, va1[-1], vaA[-1], vaB[-1], vaC[-1]]
axes[1].bar(names, accs, color=["#999", "#4C72B0", "#d9534f", "#5cb85c", "#8172B3"])
axes[1].set_ylim(0.7, 1.0)
axes[1].set_ylabel("测试准确率")
axes[1].set_title("总决赛")
axes[1].tick_params(axis="x", labelsize=8)
for i, a in enumerate(accs):
    axes[1].text(i, a + 0.005, f"{a:.3f}", ha="center", fontsize=9)
fig.suptitle("第3课：MLP vs CNN vs 数据增强（预期被数据纠正）", fontsize=14)
fig.tight_layout()
fig.savefig("cnn_vs_mlp.png", dpi=120)
print("\n图表已保存：cnn_filters.png / cnn_vs_mlp.png")
