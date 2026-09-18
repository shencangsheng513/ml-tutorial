# 进阶6：误差分析 —— 把全局一个数字切开放大看
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # Windows 自带中文字体
plt.rcParams["axes.unicode_minus"] = False

cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
pipe = joblib.load("models/model.pipeline.joblib")

df = pd.read_csv(cfg["data_path"])
X = df.drop(columns=[cfg["target"]])
y = df[cfg["target"]]
_, X_test, _, y_test = train_test_split(
    X, y, test_size=cfg["split"]["test_size"],
    random_state=cfg["split"]["random_state"], shuffle=True)

pred = pipe.predict(X_test)
err = pred - y_test.values          # 带符号误差：>0 高估, <0 低估
sq = err ** 2

print(f"全局 RMSE = {np.sqrt(sq.mean()):.4f}\n")

# ---------------------------------------------------------------
# 1) 切片：按真实价格分桶，每桶单独算误差
# ---------------------------------------------------------------
t = pd.DataFrame({"pred": pred, "actual": y_test.values, "sq": sq,
                  "income": X_test["MedInc"].values})
t["price_bin"] = pd.cut(t["actual"], [0, 1, 2, 3, 4, 4.99, 5.01],
                        labels=["<10万", "10-20万", "20-30万", "30-40万", "40-50万", "=50万顶"])
print("按真实价格分桶:")
t["bias"] = t["pred"] - t["actual"]
g = t.groupby("price_bin", observed=True).agg(
    n=("sq", "size"), RMSE=("sq", lambda s: np.sqrt(s.mean())),
    平均偏差=("bias", "mean"))
print(g.round(3))

# ---------------------------------------------------------------
# 2) 切片：按收入分桶（模型靠收入吃饭，低收入的它还行吗？）
# ---------------------------------------------------------------
t["income_bin"] = pd.cut(t["income"], [0, 2, 4, 8, 20],
                         labels=["极低收入", "中低", "中高", "富裕"])
print("\n按收入分桶:")
print(t.groupby("income_bin", observed=True).agg(
    n=("sq", "size"), RMSE=("sq", lambda s: np.sqrt(s.mean())),
    平均偏差=("bias", "mean")).round(3))

# ---------------------------------------------------------------
# 3) 最惨的 5 个预测 —— 看具体样本，永远值得做
# ---------------------------------------------------------------
worst = t.nlargest(5, "sq")
print("\n误差最大的 5 个社区:")
print(worst.assign(误差=worst["pred"] - worst["actual"])
      [["income", "actual", "pred", "误差"]].round(2).to_string())

# ---------------------------------------------------------------
# 4) 残差图：系统性偏差肉眼现形
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(pred, t["bias"], s=4, alpha=0.25)
axes[0].axhline(0, color="r", lw=1)
axes[0].set_xlabel("predicted"); axes[0].set_ylabel("residual (pred - actual)")
axes[0].set_title("Residuals vs prediction")
axes[0].grid(alpha=0.3)
b = t.groupby("price_bin", observed=True)["bias"].mean()
axes[1].bar(range(len(b)), b.values, color=np.where(b.values > 0, "tab:red", "tab:blue"))
axes[1].set_xticks(range(len(b))); axes[1].set_xticklabels(b.index, fontsize=8)
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_title("Mean bias per price bin (red=overestimate)")
axes[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig("error_analysis.png", dpi=110)
print("\n图已保存 error_analysis.png")
