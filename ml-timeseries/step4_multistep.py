# 时序第4课：多步预测 —— 递归式 vs 直接式
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

H = 6                                   # 提前 6 个月备货
df = pd.read_csv("sales.csv", parse_dates=["date"], index_col="date")
s = df["sales"]
v = s.values
dates = s.index


def feat_row(i):
    """锚点 = 最后一个已知月的位置 i，返回'预测下一个月'所需的特征（只用 v[:i+1]）"""
    return {
        "lag1": v[i], "lag2": v[i-1], "lag3": v[i-2],
        "lag6": v[i-5], "lag12": v[i-11],
        "roll3": v[i-2:i+1].mean(), "roll12": v[i-11:i+1].mean(),
        "month": dates[i].month % 12 + 1, "t": i + 1,
    }


# 训练表：锚点 i 从 11 开始（lag12 需要），预测目标月 i+1
FEATS = ["lag1", "lag2", "lag3", "lag6", "lag12", "roll3", "roll12", "month", "t"]
anchors = list(range(11, len(v) - 1))
Xall = pd.DataFrame([feat_row(i) for i in anchors], index=anchors)
y_next_diff = pd.Series({i: v[i+1] - v[i] for i in anchors})

# 预测原点：2021-01 起每 6 个月，且原点+H 不超数据末尾
origins = pd.date_range("2021-01-01", dates[-1] - pd.DateOffset(months=H - 1), freq="6MS")
print(f"预测原点 {len(origins)} 个: {origins[0]:%Y-%m} ~ {origins[-1]:%Y-%m}，每原点预测 {H} 步\n")

errs = {k: {h: [] for h in range(1, H + 1)} for k in ("naive", "rec", "dir")}

for o in origins:
    oi = dates.get_loc(o)              # 原点月位置；已知历史 = v[:oi]
    tr = Xall[Xall.index <= oi - 1]    # 锚点不超过已知范围

    # --- 递归式：1 个差分模型，把自己的预测喂回去 ---
    rf = RandomForestRegressor(n_estimators=200, max_depth=8,
                               random_state=42, n_jobs=2).fit(tr, y_next_diff[tr.index])
    sim = list(v[:oi])                 # 模拟"当前已知序列"
    for h in range(1, H + 1):
        i = len(sim) - 1               # 最后已知位置（h=1 时 = oi-1）
        row = pd.DataFrame([{
            "lag1": sim[i], "lag2": sim[i-1], "lag3": sim[i-2],
            "lag6": sim[i-5], "lag12": sim[i-11],
            "roll3": np.mean(sim[i-2:i+1]), "roll12": np.mean(sim[i-11:i+1]),
            "month": dates[i].month % 12 + 1, "t": i + 1,
        }])
        nxt = sim[i] + rf.predict(row)[0]
        errs["rec"][h].append(nxt - v[oi + h - 1])
        sim.append(nxt)                # ← 误差滚雪球的入口

    # --- 直接式：每步数 h 一个模型，目标 = s(i+h)-s(i)（累计变化，基准是已知值） ---
    i0 = oi - 1
    row0 = pd.DataFrame([feat_row(i0)])
    for h in range(1, H + 1):
        tgt = {i: v[i+h] - v[i] for i in tr.index if i + h < len(v)}
        ok = sorted(tgt)
        m = RandomForestRegressor(n_estimators=200, max_depth=8,
                                  random_state=42, n_jobs=2).fit(tr.loc[ok], pd.Series(tgt))
        errs["dir"][h].append(v[i0] + m.predict(row0)[0] - v[oi + h - 1])

    # --- Naive：抄原点最后一个月 ---
    for h in range(1, H + 1):
        errs["naive"][h].append(v[i0] - v[oi + h - 1])

tab = pd.DataFrame({
    "Naive": [np.sqrt(np.mean(np.square(errs["naive"][h]))) for h in range(1, H+1)],
    "递归式": [np.sqrt(np.mean(np.square(errs["rec"][h]))) for h in range(1, H+1)],
    "直接式": [np.sqrt(np.mean(np.square(errs["dir"][h]))) for h in range(1, H+1)],
}, index=[f"h={h}" for h in range(1, H+1)]).round(2)
print("各步数 RMSE（万元，8 个原点平均）:")
print(tab.to_string())

fig, ax = plt.subplots(figsize=(8, 4.5))
for col, mk in zip(tab.columns, ["s", "o", "^"]):
    ax.plot(range(1, H+1), tab[col], marker=mk, lw=2, label=col)
ax.set_xticks(range(1, H+1))
ax.set_xlabel("预测步数 h（月）"); ax.set_ylabel("RMSE（万元）")
ax.set_title("误差随步数放大：递归 vs 直接 vs Naive")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("ts_multistep.png", dpi=110)
print("\n图已保存 ts_multistep.png")
