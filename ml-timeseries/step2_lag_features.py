# 时序第2课：滞后特征 + 按时间切分 + 作弊对照实验
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("sales.csv", parse_dates=["date"], index_col="date")
s = df["sales"]

# ---------------------------------------------------------------
# 1) 把时序"翻译"成监督学习表
#    铁律：每个特征必须来自 t 之前 —— 一律先 shift 再 rolling
# ---------------------------------------------------------------
feat = pd.DataFrame(index=df.index)
for lag in (1, 2, 3, 6, 12):
    feat[f"lag{lag}"] = s.shift(lag)
feat["roll3"]  = s.shift(1).rolling(3).mean()    # 先shift后rolling：只用过去
feat["roll12"] = s.shift(1).rolling(12).mean()
feat["month"]  = df.index.month
feat["t"]      = np.arange(len(df))
feat["target"] = s

data = feat.dropna()
print(f"可用样本: {len(data)} / {len(feat)}（前12行因 lag12 被丢弃 —— 这就是时序的代价）")

# ---------------------------------------------------------------
# 2) 按时间切分：最后24个月是"未来考卷"，绝不乱序
# ---------------------------------------------------------------
train, test = data.iloc[:-24], data.iloc[-24:]
Xtr, ytr = train.drop(columns="target"), train["target"]
Xte, yte = test.drop(columns="target"), test["target"]
print(f"训练: {train.index[0]:%Y-%m} ~ {train.index[-1]:%Y-%m} ({len(train)}月)")
print(f"测试: {test.index[0]:%Y-%m} ~ {test.index[-1]:%Y-%m} ({len(test)}月)\n")

def score(name, actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    print(f"{name:<22} RMSE={rmse:6.2f}  MAE={mae:6.2f} 万元")
    return rmse

# ---------------------------------------------------------------
# 3) 三个选手：Naive / Ridge / 随机森林
# ---------------------------------------------------------------
score("Naive(抄上月)", yte, Xte["lag1"].values)

ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
p_ridge = ridge.predict(Xte)
score("Ridge回归", yte, p_ridge)

rf = RandomForestRegressor(n_estimators=300, max_depth=8,
                           random_state=42, n_jobs=2).fit(Xtr, ytr)
p_rf = rf.predict(Xte)
score("随机森林", yte, p_rf)

# ---------------------------------------------------------------
# 3b) 关键诊断：为什么"更强的模型"反而输了？
# ---------------------------------------------------------------
print(f"\n【诊断】训练期真实值范围 [{ytr.min():.0f}, {ytr.max():.0f}]，测试期真实值范围 [{yte.min():.0f}, {yte.max():.0f}]")
print(f"  Ridge 预测范围   [{p_ridge.min():.0f}, {p_ridge.max():.0f}]  ← 能超出训练范围")
print(f"  随机森林预测范围 [{p_rf.min():.0f}, {p_rf.max():.0f}]  ← 被'困'在训练范围内")
print("  → 树模型靠'查表平均'预测，天生无法外推趋势；线性模型 extrapolate 是强项")

# ---------------------------------------------------------------
# 4) 对照实验：如果"不小心"用了未来信息会怎样？
#    roll12_center = 中心滚动均值（一半窗口来自未来！）+ 乱序切分
# ---------------------------------------------------------------
feat_cheat = feat.copy()
feat_cheat["roll12_center"] = s.rolling(12, center=True).mean()  # ← 泄漏源
d2 = feat_cheat.dropna()
rng = np.random.default_rng(0)
mask = rng.random(len(d2)) < 0.2
Xc_tr, yc_tr = d2[~mask].drop(columns="target"), d2[~mask]["target"]
Xc_te, yc_te = d2[mask].drop(columns="target"), d2[mask]["target"]
rf_cheat = RandomForestRegressor(n_estimators=300, max_depth=8,
                                 random_state=42, n_jobs=2).fit(Xc_tr, yc_tr)
print("\n【作弊对照】加了'未来滚动均值'+乱序切分:")
score("  随机森林(作弊版)", yc_te, rf_cheat.predict(Xc_te))
print("  ↑ 数字好到假 —— 上线第一天就会崩，因为生产环境永远没有'未来'")

# ---------------------------------------------------------------
# 5) 测试期预测图
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(df.index, s, color="gray", lw=0.9, label="历史")
ax.plot(test.index, yte, "k-", lw=2, label="真实(测试期)")
ax.plot(test.index, p_rf, "o--", ms=4, color="tab:orange", label="随机森林预测")
ax.plot(test.index, Xte["lag1"], "s--", ms=4, color="tab:blue", alpha=0.6, label="Naive预测")
ax.axvline(test.index[0], color="r", ls=":", lw=1)
ax.set_title("测试期（最后24个月）：模型 vs Naive")
ax.legend()
fig.tight_layout(); fig.savefig("ts_forecast.png", dpi=110)
print("\n图已保存 ts_forecast.png")
