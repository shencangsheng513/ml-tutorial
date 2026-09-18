# -*- coding: utf-8 -*-
"""
项目4《计算机视觉》第1课：图像就是数字 —— 手写数字 EDA
三个项目走完：表格(每行=一条记录) → 时序(每行=一刻) → 文本(每行=一段话)。
图像是第四个世界：每张图片 = 一个像素矩阵，"行"变成了二维。
本课问题：模型眼里的"7"长什么样？哪些数字天生容易混？
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

digits = load_digits()
X, y = digits.data, digits.target          # X: (1797, 64) 已拍平
img = X.reshape(-1, 8, 8)                  # 还原成 1797 张 8×8 图
print(f"数据形状：{X.shape}（1797 张 8×8 灰度图，拍平成 64 列）")
print(f"像素范围：{X.min():.0f} ~ {X.max():.0f}（0=黑，16=白，只有17个等级）")
print(f"类别分布：{dict(pd.Series(y).value_counts().sort_index())}（每类约180张，均衡）")

# ============================================================
# 第1步：亲眼看——模型收到的"64个数"
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(9, 4.6))
rng = np.random.default_rng(42)
for col in range(10):
    idx = rng.choice(np.where(y == col)[0])
    ax = axes.ravel()[col]
    ax.imshow(img[idx], cmap="gray_r", vmin=0, vmax=16)
    ax.set_title(f"{col}", fontsize=14)
    ax.axis("off")
fig.suptitle("每个数字其实是 8×8=64 个 0~16 的灰度值", fontsize=13)
fig.tight_layout()
fig.savefig("digits_samples.png", dpi=120)
print("\n第一张图：digits_samples.png —— 左到右 0~9 各一张真实样本")

# 放大一张看像素本体
i = int(np.where(y == 7)[0][0])
print(f"\n一张'7'的像素矩阵（模型看到的全部）：")
for row in img[i].astype(int):
    print("   ", " ".join(f"{v:>3}" for v in row))

# ============================================================
# 第2步：像素统计 —— 有没有"作弊特征"？
# ============================================================
print("\n" + "=" * 60)
print("第2步：每张图共 64 个特征，哪些像素位置最'忙'？")
print("=" * 60)
brightness = img.mean(axis=0)          # 8×8：所有图平均亮度
print("平均亮度图（数字常走的笔画区域会亮）：")
for row in brightness.astype(int):
    print("   ", " ".join("·" if v < 3 else str(min(v // 2, 9)) for v in row))
print("→ 中心区亮、四角黑：四角像素几乎恒为0，是'死特征'——")
print("  第2课模型对比时可以试试删掉它们，速度换精度会不会亏。")

# ============================================================
# 第3步：t-SNE —— 谁和谁抱团？（预演混淆矩阵）
# ============================================================
print("\n" + "=" * 60)
print("第3步：t-SNE 把 64 维压到 2 维，看数字们的'邻里关系'")
print("=" * 60)
Z = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(
    StandardScaler().fit_transform(X))
fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=8, alpha=0.8)
for k in range(10):
    m = y == k
    ax.text(Z[m, 0].mean(), Z[m, 1].mean(), str(k), fontsize=16, weight="bold",
            ha="center", va="center",
            bbox=dict(fc="white", alpha=0.7, ec="none", boxstyle="round,pad=0.2"))
fig.colorbar(sc, label="真实数字")
ax.set_title("t-SNE：挨着的类=长得像=将来容易混")
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig("digits_tsne.png", dpi=120)
print("→ 看图找'贴在一起'的数字对（常见：4↔9、3↔5、7↔9）。")
print("  第2课训练完，用混淆矩阵核对 t-SNE 的预言准不准——")
print("  这就是'EDA 提前预测模型错误'的实战版。")
print("\n图表已保存：digits_samples.png / digits_tsne.png")
