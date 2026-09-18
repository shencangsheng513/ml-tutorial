# 时序上线应用：输入最近历史 -> 未来6个月备货预测
# 启动: python -m streamlit run app.py
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="销量预测器", page_icon="📦", layout="wide")
fc = joblib.load("models/forecaster.joblib")
report = __import__("json").loads(open("models/report.json", encoding="utf-8").read())
s = pd.read_csv("sales.csv", parse_dates=["date"], index_col="date")["sales"]

st.title("📦 未来 6 个月销量预测（备货助手）")
st.caption(f"直接式多步 RF | walk-forward 总RMSE {report['overall_rmse']} 万元 "
           f"(Naive {report['naive_overall_rmse']})")

with st.sidebar:
    st.header("情景模拟")
    adj = st.slider("假设本月销量修正（万元）", -30.0, 30.0, 0.0, 1.0)
    n_hist = st.slider("展示历史长度（月）", 12, 84, 36)

hist = s.copy()
hist.iloc[-1] = hist.iloc[-1] + adj          # 情景：本月实际与数据不同

preds = fc.predict(hist)
future_idx = pd.date_range(hist.index[-1] + pd.offsets.MonthBegin(1),
                           periods=6, freq="MS")
out = pd.DataFrame({"预测销量(万元)": [preds.get(h) for h in range(1, 7)]},
                   index=future_idx).round(1)
out["环比变化"] = out["预测销量(万元)"].diff().round(1)

c1, c2 = st.columns([2, 1])
with c1:
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(hist.index[-n_hist:], hist.values[-n_hist:], "k-", lw=1.5, label="历史")
    ax.plot(out.index, out["预测销量(万元)"], "o--", color="tab:red", label="预测")
    ax.axvline(hist.index[-1], color="r", ls=":", lw=1)
    ax.legend(); ax.grid(alpha=0.3); ax.set_title(f"历史 {n_hist} 个月 + 未来 6 个月")
    st.pyplot(fig)
with c2:
    st.dataframe(out, use_container_width=True)

st.info("提示：拖动左栏滑块做情景模拟 —— 本月比数据里多/少卖 X 万，"
        "看未来 6 个月预测如何整体平移。这就是预测系统的'what-if'用法。")
