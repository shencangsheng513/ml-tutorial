# 第4课：预测实战 + 特征重要性 + 复盘
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # 无界面环境下保存图片
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

bundle = joblib.load("model_rf.joblib")
model, features = bundle["model"], bundle["features"]

test_df = pd.read_csv("data/test.csv")
X_test = test_df[features]

# ---------------------------------------------------------------
# 1) 实战报价：随机抽 5 个社区，模型估价 vs 真实成交价
# ---------------------------------------------------------------
sample = test_df.sample(5, random_state=7)
preds = model.predict(sample[features])
report = pd.DataFrame({
    "收入(万美元)": sample["MedInc"],
    "房龄": sample["HouseAge"],
    "真实房价(万美元)": sample["MedHouseVal"] * 10,
    "模型估价(万美元)": preds * 10,
})
report["误差(万美元)"] = report["模型估价(万美元)"] - report["真实房价(万美元)"]
print("1) 随机 5 个社区的报价对比:")
print(report.round(1).to_string(index=False))

# ---------------------------------------------------------------
# 2) 特征重要性：模型认为什么最影响房价？
# ---------------------------------------------------------------
imp = pd.Series(model.feature_importances_, index=features)
imp = (imp / imp.sum() * 100).sort_values(ascending=False)
print("\n2) 特征重要性（占比%）:")
print(imp.round(1).to_string())

# ---------------------------------------------------------------
# 3) 预测 vs 真实 散点图 —— 项目最终交付物
# ---------------------------------------------------------------
all_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(test_df["MedHouseVal"], all_pred))
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(test_df["MedHouseVal"], all_pred, s=4, alpha=0.3)
ax.plot([0, 5.2], [0, 5.2], "r--", lw=1)
ax.set_xlabel("Actual price (x$100k)")
ax.set_ylabel("Predicted price (x$100k)")
ax.set_title(f"Random Forest: pred vs actual (test RMSE={rmse:.3f})")
fig.tight_layout()
fig.savefig("pred_vs_actual.png", dpi=110)
print(f"\n3) 图已保存 pred_vs_actual.png（红虚线=完美预测线，点越贴近越好）")
print("   注意右边缘的水平'平顶'：就是第1课说的 5.00 截断造成的")
