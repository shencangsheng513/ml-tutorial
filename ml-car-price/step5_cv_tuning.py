# 进阶1：交叉验证调参 —— 理解偏差-方差权衡
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

train_df = pd.read_csv("data/train.csv")
features = [c for c in train_df.columns if c != "MedHouseVal"]
X, y = train_df[features], train_df["MedHouseVal"]

# ---------------------------------------------------------------
# 问题：上一课 max_depth=16 是我"拍脑袋"定的。怎么科学地选？
#
# 不能用测试集选参数（那是作弊）。正确做法：在训练集内部做 K 折交叉验证
# 把训练集切成 5 份，轮流用 4 份训练、1 份验证，跑 5 次取平均
# ---------------------------------------------------------------
print("用 5 折交叉验证扫描 max_depth（每个值都跑 5 次取平均）:\n")
depths = [3, 5, 8, 12, 16, 24, None]
cv_mean, cv_std = [], []
for d in depths:
    rf = RandomForestRegressor(n_estimators=100, max_depth=d,
                               random_state=42, n_jobs=-1)
    # negative RMSE（sklearn 约定：越大越好，所以取负）
    scores = cross_val_score(rf, X, y, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    rmse = -scores.mean()
    cv_mean.append(rmse)
    cv_std.append(scores.std())
    label = str(d) if d else "None(无限)"
    print(f"  max_depth={label:<10} CV-RMSE={rmse:.4f} ±{scores.std():.4f}")

best = depths[int(np.argmin(cv_mean))]
print(f"\n>>> 最优 max_depth = {best}（CV-RMSE 最低）")

# ---------------------------------------------------------------
# 偏差-方差权衡可视化
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 左图：模型复杂度 vs 误差（核心图）
axes[0].errorbar(range(len(depths)), cv_mean, yerr=cv_std, marker="o", capsize=4)
axes[0].set_xticks(range(len(depths)))
axes[0].set_xticklabels([str(d) if d else "∞" for d in depths])
axes[0].set_xlabel("max_depth (model complexity)")
axes[0].set_ylabel("CV RMSE (lower=better)")
axes[0].set_title("Bias-Variance Tradeoff")
axes[0].axvline(int(np.argmin(cv_mean)), color="r", ls="--", lw=1)
axes[0].grid(alpha=0.3)

# 右图：学习曲线 —— 判断该多要数据还是该换模型
rf = RandomForestRegressor(n_estimators=100, max_depth=best or None, random_state=42, n_jobs=-1)
sizes, tr, va = learning_curve(rf, X, y, cv=5, scoring="neg_root_mean_squared_error",
                               train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1)
tr, va = -tr.mean(1), -va.mean(1)
axes[1].plot(sizes, tr, "o-", label="train")
axes[1].plot(sizes, va, "o-", label="validation")
axes[1].set_xlabel("training set size")
axes[1].set_ylabel("RMSE")
axes[1].set_title(f"Learning curve (depth={best})")
axes[1].legend(); axes[1].grid(alpha=0.3)

fig.tight_layout(); fig.savefig("cv_tuning.png", dpi=110)
print("图已保存 cv_tuning.png")
print(f"\n学习曲线诊断: 训练末={tr[-1]:.3f} 验证末={va[-1]:.3f}")
if va[-1] - tr[-1] > 0.15:
    print("  → 验证明显高于训练 = 高方差(过拟合)：加数据 / 降复杂度 会有用")
elif tr[-1] > 0.5:
    print("  → 两者都高且接近 = 高偏差(欠拟合)：需要更强的模型或更多特征")
else:
    print("  → 两者都低且接近 = 拟合良好")
