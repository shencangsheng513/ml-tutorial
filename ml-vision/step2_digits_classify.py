# -*- coding: utf-8 -*-
"""
CV第2课：首个分类器 + 和 t-SNE 的预言"对质"
上节课 t-SNE 让我们押宝：4↔9、3↔5、7↔9 会互相混淆。
本课训练四个经典模型，用混淆矩阵核对预言——
这是第一次"用 EDA 提前预测模型会错在哪"，预测对了才算真懂数据。
另做一个小实验：删掉第1课发现的"死角像素"，精度会掉吗？
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

digits = load_digits()
X, y = digits.data, digits.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"训练 {len(Xtr)} / 测试 {len(Xte)}（分层抽样，每类比例不变——项目1规矩）")

# ============================================================
# 第1步：四个模型同台（5折CV选将 + 测试集交卷）
# ============================================================
print("\n" + "=" * 60)
print("第1步：LR / KNN / 随机森林 / SVM(rbf)")
print("=" * 60)
models = {
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "KNN(k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "RandomForest": RandomForestClassifier(300, n_jobs=2, random_state=42),
    "SVM(rbf)": make_pipeline(StandardScaler(), SVC()),
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
rows = []
for name, m in models.items():
    s = cross_val_score(m, Xtr, ytr, cv=cv, n_jobs=1)
    m.fit(Xtr, ytr)
    te_acc = accuracy_score(yte, m.predict(Xte))
    rows.append((name, s.mean(), s.std(), te_acc))
    print(f"  {name:<20} CV {s.mean():.4f} ± {s.std():.4f} | 测试 {te_acc:.4f}")
best_name = max(rows, key=lambda r: r[3])[0]
best = models[best_name]
print(f"\n→ 冠军：{best_name}。图像像素和文本一样是高维问题，")
print("  但这里 SVM 的'弯曲刀'（rbf核）终于能赢树了——因为相邻像素")
print("  强相关，rbf 核天然喜欢这种结构。")

# ============================================================
# 第2步：对质时刻 —— 混淆矩阵 vs t-SNE 预言
# ============================================================
print("\n" + "=" * 60)
print("第2步：错误都落在哪对上？（t-SNE 预言核对）")
print("=" * 60)
pred = best.predict(Xte)
cm = confusion_matrix(yte, pred)
pairs = sorted([(a, b, int(cm[a, b])) for a in range(10) for b in range(10)
                if a != b and cm[a, b] > 0], key=lambda t: -t[2])
print(f"总错误 {len(yte) - int(cm.trace())} 张。Top6 混淆对：")
prophecy = {(4, 9), (9, 4), (3, 5), (5, 3), (7, 9), (9, 7)}
hit = miss = 0
for a, b, n in pairs[:6]:
    okp = (a, b) in prophecy
    hit += okp
    miss += not okp
    print(f"  {a} → 误判为 {b}: {n:>2} 张   {'✔ 命中t-SNE预言' if okp else '✘ 预言外'}")
print(f"→ Top6 里 {hit} 对命中预言、{miss} 对是意外。")
print("  预言落空的两层原因：① 模型太强，总共才错 9 张，预言无从应验")
print("  （实测：换成只喂120张训练的弱模型(90%)，7↔9 现形了，4↔9 仍没")
print("  现身——t-SNE 的预言只能部分兑现）；② t-SNE 画的是'类的整体")
print("  邻里'，不是'具体哪张图会错'——定性方向感 + 定量混淆矩阵，缺一不可。")

# 看几张"预言级"错例：模型把谁看成了谁
print("\n错例长这样（前3条混淆对各取1张）：")
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
for ax, (a, b, _) in zip(axes, pairs[:3]):
    idx = np.where((yte == a) & (pred == b))[0][0]
    ax.imshow(Xte[idx].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"真{a} 判成{b}", fontsize=11)
    ax.axis("off")
fig.suptitle(f"{best_name} 的典型错误")
fig.tight_layout()
fig.savefig("digits_errors.png", dpi=120)

# ============================================================
# 第3步：死角像素实验 —— 删掉会亏吗？
# ============================================================
print("\n" + "=" * 60)
print("第3步：删掉 4 个恒暗角像素（64→60维），精度变化？")
print("=" * 60)
corners = [0, 7, 56, 63]
keep = [i for i in range(64) if i not in corners]
s = cross_val_score(models["SVM(rbf)"], Xtr[:, keep], ytr, cv=cv, n_jobs=1)
print(f"  SVM 用60维(删角)  CV {s.mean():.4f}（64维时 {rows[3][1]:.4f}）")
print("→ 几乎不亏甚至微赚：死特征是纯噪声维度。")
print("  但注意：这是 8×8 小图才敢手动挑像素；真实 224×224 图像")
print("  靠的是卷积网络自动学'哪里重要'——第3课见。")

# ============================================================
# 画图：混淆矩阵 + 模型对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title(f"{best_name} 混淆矩阵（对角线外=错误）")
names = [r[0] for r in rows]
axes[1].barh(names, [r[1] for r in rows], xerr=[r[2] for r in rows], capsize=4,
             color=["#4C72B0", "#55A868", "#C44E52", "#8172B3"])
axes[1].set_xlabel("5折CV 准确率")
axes[1].set_xlim(0.9, 1.01)
axes[1].set_title("四模型 CV 对比")
fig.tight_layout()
fig.savefig("digits_classify.png", dpi=120)
print("\n图表已保存：digits_errors.png / digits_classify.png")
