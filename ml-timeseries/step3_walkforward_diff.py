# 时序第3课：滚动原点验证(walk-forward) + 差分变换让树模型翻盘
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("sales.csv", parse_dates=["date"], index_col="date")
s = df["sales"]

feat = pd.DataFrame(index=df.index)
for lag in (1, 2, 3, 6, 12):
    feat[f"lag{lag}"] = s.shift(lag)
feat["roll3"] = s.shift(1).rolling(3).mean()
feat["roll12"] = s.shift(1).rolling(12).mean()
feat["month"] = df.index.month
feat["t"] = np.arange(len(df))
data = feat.copy()
data["target_abs"] = s
data["target_diff"] = s.diff()          # 环比变化量
data = data.dropna()

# ---------------------------------------------------------------
# 1) 为什么 K 折交叉验证在时序上作废？
#    随机折会把"2021年的样本"分进训练、"2019年"分进验证 ——
#    用未来考过去。正确范式：模拟真实上线节奏 ——
#    每次只用"截至今天"的数据训练，预测接下来 H 个月，然后时间前移
# ---------------------------------------------------------------
years = ["2021", "2022", "2023", "2024"]     # 4 个滚动年度折
print("Walk-forward：每年重训一次，预测当年 12 个月（1步前瞻，用真实上月值）\n")

FEATS = list(feat.columns)
results = {k: [] for k in ["Naive", "Ridge(绝对值)", "RF(绝对值)", "RF(差分)"]}
pred_store, actual_store, fold_labels = [], [], []

for y in years:
    te_mask = data.index.year == int(y)
    tr = data[~te_mask & (data.index < data.index[te_mask][0])]
    te = data[te_mask]
    Xtr, Xte = tr[FEATS], te[FEATS]

    # Naive: 抄上月
    results["Naive"].append(np.sqrt(mean_squared_error(te["target_abs"], te["lag1"])))
    # Ridge 直接预测绝对值
    m = Ridge(alpha=1.0).fit(Xtr, tr["target_abs"])
    results["Ridge(绝对值)"].append(np.sqrt(mean_squared_error(te["target_abs"], m.predict(Xte))))
    # RF 预测绝对值（第2课的失败者）
    m2 = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=2)
    m2.fit(Xtr, tr["target_abs"])
    p_abs = m2.predict(Xte)
    results["RF(绝对值)"].append(np.sqrt(mean_squared_error(te["target_abs"], p_abs)))
    # RF 预测差分, 再重构: 本月 = 上月真值 + 预测变化量
    m3 = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=2)
    m3.fit(Xtr, tr["target_diff"])
    p_diff = te["lag1"].values + m3.predict(Xte)
    results["RF(差分)"].append(np.sqrt(mean_squared_error(te["target_abs"], p_diff)))

    pred_store += [te["lag1"].values, m.predict(Xte), p_abs, p_diff]
    actual_store.append(te["target_abs"].values)
    fold_labels += [y] * 12

# ---------------------------------------------------------------
# 2) 成绩单：按年 RMSE 表 + 48 个月拼接总 RMSE
# ---------------------------------------------------------------
groups = {"Naive": 0, "Ridge(绝对值)": 1, "RF(绝对值)": 2, "RF(差分)": 3}
all_actual = np.concatenate(actual_store)
flat = {}
for name, gi in groups.items():
    preds = np.concatenate([pred_store[i * 4 + gi] for i in range(len(years))])
    flat[name] = np.sqrt(mean_squared_error(all_actual, preds))

tab = pd.DataFrame(results, index=years).round(2)
tab.loc["总计(48月)"] = [round(flat[k], 2) for k in groups]
print("按年 RMSE（万元）:")
print(tab.to_string())

# ---------------------------------------------------------------
# 3) 图：2021-2024 预测对比
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(data.index[data.index.year >= 2021], all_actual, "k-", lw=2, label="真实")
colors = {"Naive": "tab:gray", "Ridge(绝对值)": "tab:blue",
          "RF(绝对值)": "tab:orange", "RF(差分)": "tab:red"}
for name, gi in groups.items():
    preds = np.concatenate([pred_store[i * 4 + gi] for i in range(len(years))])
    ax.plot(data.index[data.index.year >= 2021], preds, lw=1.2, alpha=0.8,
            color=colors[name], label=f"{name} RMSE={flat[name]:.1f}")
ax.set_title("Walk-forward 预测对比（每年重训, 2021-2024）")
ax.legend()
fig.tight_layout(); fig.savefig("ts_walkforward.png", dpi=110)
print("\n图已保存 ts_walkforward.png")
