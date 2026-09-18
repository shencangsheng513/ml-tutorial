# 时序第1课：造一份"有故事"的数据 + 看懂趋势与季节性
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# 2015-01 ~ 2024-12 共 120 个月的"某连锁店月度销售额"（万元）
idx = pd.date_range("2015-01-01", "2024-12-01", freq="MS")
t = np.arange(len(idx))
month = idx.month

trend     = 80 + 0.9 * t                                # 缓慢增长
season    = 12 * np.sin(2 * np.pi * (month - 3) / 12)   # 夏季旺季
season   += np.where(month == 12, 18, 0)                # 12月年终冲刺
noise     = rng.normal(0, 3.5, len(idx))                # 随机波动
shock     = np.zeros(len(idx)); shock[36:42] -= 25      # 2018年装修潮(异常事件)
sales     = trend + season + noise + shock

df = pd.DataFrame({"date": idx, "sales": sales.round(2)}).set_index("date")
df.to_csv("sales.csv")

# ---------------------------------------------------------------
# EDA 三板斧
# ---------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 9))

axes[0].plot(df.index, df["sales"], lw=0.9, marker=".", ms=3)
axes[0].plot(df.index, df["sales"].rolling(12, center=True).mean(), lw=2, label="12月滚动平均(趋势)")
axes[0].axvspan(pd.Timestamp("2018-01-01"), pd.Timestamp("2018-06-30"), color="red", alpha=0.15)
axes[0].set_title("月度销售额：趋势 + 季节 + 异常区间(红)")
axes[0].legend()

# 季节性：按月份聚合看箱线
seasonal = df["sales"].groupby(df.index.month)
axes[1].boxplot([g.values for _, g in seasonal], tick_labels=range(1, 13))
axes[1].set_title("各月份分布（季节性现形）")

# 自相关：本月销售和 N 月前销售的相关性 —— 时序的"特征相关性"
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df["sales"], ax=axes[2])
axes[2].set_title("自相关图 ACF（记住 12 的倍数处会隆起）")

fig.tight_layout(); fig.savefig("ts_eda.png", dpi=110)

print(df.head(3))
print(f"\n行数 {len(df)}, 缺失 {df['sales'].isna().sum()}")
print("\n同月同比增速(2024 vs 2023):")
y23 = df.loc["2023", "sales"]; y24 = df.loc["2024", "sales"]
print(((y24.values - y23.values) / y23.values * 100).round(1))
print("\n图已保存 ts_eda.png")
