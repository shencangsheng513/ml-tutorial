# 进阶7：Permutation Importance —— 用"破坏实验"衡量特征依赖
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
pipe = joblib.load("models/model.pipeline.joblib")

df = pd.read_csv(cfg["data_path"])
X = df.drop(columns=[cfg["target"]])
y = df[cfg["target"]]
_, X_test, _, y_test = train_test_split(
    X, y, test_size=cfg["split"]["test_size"],
    random_state=cfg["split"]["random_state"])

# ---------------------------------------------------------------
# 1) 置换重要性：对 Pipeline 整体做破坏实验（输入=原始8列）
#    打乱 MedInc 会连带让内部派生的 LogIncome 一起失效 —— 衡量的是
#    "这个原始特征（含它的全部衍生）对模型值多少"
# ---------------------------------------------------------------
res = permutation_importance(pipe, X_test, y_test,
                             scoring="neg_root_mean_squared_error",
                             n_repeats=10, random_state=42, n_jobs=1)
perm = pd.DataFrame({"RMSE增加": res.importances_mean,
                     "std": res.importances_std},
                    index=X_test.columns).sort_values("RMSE增加")

# ---------------------------------------------------------------
# 2) 内置重要性（gain）：模型内部11个特征（含派生）各自分走多少功劳
# ---------------------------------------------------------------
built = pd.Series(pipe.named_steps["model"].feature_importances_,
                  index=pipe["features"].transform(X_test[:2]).columns)

print("左=置换重要性(原始8列, RMSE增加)  右=内置gain重要性(11列):")
cmp = pd.concat([perm["RMSE增加"].rename("permutation"),
                 built.rename("gain")], axis=1)
print(cmp.round(4).sort_values("permutation", ascending=False))

# ---------------------------------------------------------------
# 3) 可视化对比
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].barh(perm.index, perm["RMSE增加"], xerr=perm["std"])
axes[0].set_title("Permutation importance (raw 8 cols)")
axes[1].barh(built.sort_values().index, built.sort_values().values)
axes[1].set_title("Built-in gain importance (11 cols)")
for a in axes: a.grid(alpha=0.3, axis="x")
fig.tight_layout(); fig.savefig("permutation_importance.png", dpi=110)
print("\n图已保存 permutation_importance.png")

# ---------------------------------------------------------------
# 4) 关键实验：MedInc 被 LogIncome "备份"导致置换值被低估
# ---------------------------------------------------------------
from sklearn.metrics import mean_squared_error
rmse0 = np.sqrt(mean_squared_error(y_test, pipe.predict(X_test)))
Xm = X_test.copy()
rng = np.random.default_rng(0)
Xm["MedInc"] = rng.permutation(Xm["MedInc"])
# 但派生列在pipeline内部会重算 —— 所以这里测的是"原始列打乱后全链路失效"
rmse1 = np.sqrt(mean_squared_error(y_test, pipe.predict(Xm)))
print(f"\n基线RMSE={rmse0:.4f} → 只打乱MedInc后={rmse1:.4f}")
print("注意: pipeline 内部 LogIncome 由打乱后的 MedInc 重算,")
print("所以这个实验测的是'MedInc信息整体消失'的代价, 而非单列冗余。")
print("若想测共线性遮蔽, 需对已派生好的11列矩阵单独打乱 MedInc —— 自己试试!")
