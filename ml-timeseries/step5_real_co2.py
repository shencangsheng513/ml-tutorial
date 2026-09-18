# 时序第5课：真实数据 Mauna Loa CO2 —— 重采样 + 季节分解 + 方法复验
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------
# 1) 真实数据的第一课：它从来不是现成的月频表
#    原始是周频(1958-03~2001-12, 有缺口) -> 重采样为月均值
# ---------------------------------------------------------------
co2 = sm.datasets.co2.load_pandas().data["co2"]
monthly = co2.resample("MS").mean()
print(f"周频 {len(co2)} 条 -> 月频 {len(monthly)} 条, 缺失 {monthly.isna().sum()} 个月")
monthly = monthly.dropna()          # 教学选择：直接删缺失月；插值也是选项，各有代价
monthly.to_csv("co2_monthly.csv")

# ---------------------------------------------------------------
# 2) 季节性分解：把序列拆成 趋势 × 季节 × 残差
#    (合成数据是我们"造"出来的结构; 真实数据要靠这个"验")
# ---------------------------------------------------------------
dec = seasonal_decompose(monthly, period=12, model="additive")
fig = dec.plot()
fig.set_size_inches(10, 8)
fig.suptitle("Mauna Loa CO2: 经典分解 (趋势+季节+残差)", y=1.01)
fig.savefig("co2_decompose.png", dpi=110, bbox_inches="tight")

# 从分解结果读事实：
amp = dec.seasonal.max() - dec.seasonal.min()
print(f"季节振幅 ≈ {amp:.1f} ppm（北半球植物'呼吸'：5月峰值,10月谷值）")
print(f"趋势: {dec.trend.dropna().iloc[0]:.0f} -> {dec.trend.dropna().iloc[-1]:.0f} ppm，近乎线性上升")
print(f"残差 std ≈ {dec.resid.dropna().std():.2f} ppm —— 比销售额数据干净得多！")

# ---------------------------------------------------------------
# 3) 复验第2-3课方法：滞后特征 + walk-forward + 差分变换
#    残差这么小 => 这次 Naive 会强得离谱（惯性序列的特点）
# ---------------------------------------------------------------
s = monthly
idx = s.index
rows = []
for i in range(12, len(s) - 1):
    w = s.values[i-11:i+1]
    rows.append({"lag1": w[-1], "lag2": w[-2], "lag3": w[-3], "lag6": w[-6],
                 "lag12": w[0], "roll3": w[-3:].mean(), "roll12": w.mean(),
                 "month": idx[i].month % 12 + 1, "t": i + 1,
                 "y_abs": s.values[i+1], "y_diff": s.values[i+1] - w[-1]})
D = pd.DataFrame(rows).set_index(idx[13:])
FE = [c for c in D.columns if not c.startswith("y_")]

# 最后 60 个月做测试（1997-01 ~ 2001-12）
tr, te = D.iloc[:-60], D.iloc[-60:]
Xtr, Xte, ytr, yte = tr[FE], te[FE], tr["y_abs"], te["y_abs"]

naive = Xte["lag1"].values
ridge = Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte)
rf_abs = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42,
                               n_jobs=2).fit(Xtr, ytr).predict(Xte)
rf_dif = Xte["lag1"].values + RandomForestRegressor(
    n_estimators=200, max_depth=8, random_state=42, n_jobs=2
).fit(Xtr, tr["y_diff"]).predict(Xte)

print("\n最后60个月 walk-forward 式测试 (RMSE, ppm):")
for name, p in [("Naive(抄上月)", naive), ("Ridge(绝对值)", ridge),
                ("RF(绝对值)", rf_abs), ("RF(差分)", rf_dif)]:
    print(f"  {name:<12} {np.sqrt(mean_squared_error(yte, p)):6.3f}")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(idx[-61:], s.values[-61:], "k-", lw=1.5, label="真实")
ax.plot(te.index, ridge, label="Ridge")
ax.plot(te.index, rf_dif, label="RF(差分)")
ax.set_title("CO2 最后60个月预测"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("co2_forecast.png", dpi=110)
print("\n图: co2_decompose.png, co2_forecast.png")
