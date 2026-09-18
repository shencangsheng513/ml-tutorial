# 进阶5后的应用：原始8列直接进 Pipeline，应用侧零特征工程
# 启动: python -m streamlit run app.py
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="加州房价估价器", page_icon="🏠", layout="wide")
pipe = joblib.load("models/model.pipeline.joblib")
train_raw = pd.read_csv("data/california_housing.csv")

st.title("🏠 加州社区房价估价器")
st.caption("Pipeline 模型：原始特征进 → 估价出，特征工程在模型内部完成")

with st.sidebar:
    st.header("社区信息")
    row = {
        "MedInc":     st.slider("收入中位数（万美元）", 0.5, 15.0, 3.9, 0.1),
        "HouseAge":   st.slider("房龄（年）", 1, 52, 29),
        "AveRooms":   st.slider("平均房间数", 0.9, 15.0, 5.4, 0.1),
        "AveBedrms":  st.slider("平均卧室数", 0.5, 5.0, 1.1, 0.1),
        "Population": st.slider("人口", 0, 40000, 1400),
        "AveOccup":   st.slider("平均每户人数", 0.5, 10.0, 2.8, 0.1),
        "Latitude":   st.slider("纬度", 32.5, 42.0, 35.6, 0.1),
        "Longitude":  st.slider("经度", -124.4, -114.3, -119.6, 0.1),
    }

pred = float(pipe.predict(pd.DataFrame([row]))[0])

c1, c2 = st.columns([1, 2])
with c1:
    st.metric("模型估价", f"${pred*10:.1f} 万")
with c2:
    st.subheader("输入 vs 全量数据中位数")
    comp = pd.DataFrame({
        "你的输入": row,
        "数据中位数": train_raw[list(row)].median(),
    })
    st.dataframe(comp.round(2), use_container_width=True)

if pred >= 4.9:
    st.warning("注意：数据集中超过 $50 万的房价被截断为 5.0，此价位预测值被'天花板'压低")

st.write(f"分类视角：估价 {'高于' if pred > 3 else '低于'} $30 万 → "
         f"{'豪宅' if pred > 3 else '普通'} 类")
