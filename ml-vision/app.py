# -*- coding: utf-8 -*-
"""
CV第4课·下半场：手绘数字识别 App（Streamlit）
两种玩法：
  ① 像素格子：直接在 8×8=64 个格子上涂色——这就是模型的全部视野，零预处理误差
  ② 拍照上传：纸上写数字拍照，走 canvas_to_input 管线（裁包围盒→居中→8×8）
预处理与训练共用 src/preprocess.py 一份代码，杜绝 train/serve skew。
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.preprocess import canvas_to_input


class SmallCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Flatten(), torch.nn.Linear(16 * 2 * 2, 10))

    def forward(self, x):
        return self.net(x)


@st.cache_resource
def load_model():
    m = SmallCNN()
    m.load_state_dict(torch.load(ROOT / "models" / "cnn_digits.pt",
                                 map_location="cpu", weights_only=True))
    m.eval()
    return m


def predict_and_show(x):
    """x: (8,8) float in [0,1]，白字黑底"""
    model = load_model()
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(x).unsqueeze(0).unsqueeze(0)), 1)[0]
    big = np.repeat(np.repeat(x, 30, 0), 30, 1)
    col_a, col_b = st.columns([1, 1.4])
    with col_a:
        st.image(big, width=240, caption="模型真正看到的 64 格")
    with col_b:
        top2 = torch.topk(probs, 2)
        st.metric("识别结果", f"{top2.indices[0].item()}",
                  f"置信度 {top2.values[0].item():.1%}")
        if top2.values[0].item() < 0.5:
            st.warning("置信度不到 50%——笔画更饱满些，或按第二候选人工复核")
        st.bar_chart({str(i): float(probs[i]) for i in range(10)}, height=170)


st.set_page_config(page_title="手绘数字识别", page_icon="✏️", layout="wide")
st.title("✏️ 手绘数字识别 —— 你只有 64 个格子")
st.caption("数据集里每位数字就是 8×8 灰度图。给模型的一切，都装不进这 64 格以外的信息。")

mode = st.radio("玩法", ["① 像素格子（直接涂模型视野）", "② 拍照上传（走完整预处理管线）"],
                horizontal=True)

if mode.startswith("①"):
    st.markdown("点亮格子=白色笔画。先用按钮清空，再一格一格涂个数字：")
    ctrls = st.columns(3)
    if ctrls[0].button("全部清空"):
        st.session_state.pop("px", None)
        st.rerun()
    # 8 行开关，改动写回 8×8 矩阵
    if "px" not in st.session_state:
        st.session_state.px = np.zeros((8, 8), dtype=np.float32)
    grid = st.container(border=True)
    changed = False
    for r in range(8):
        cols = grid.columns([1] + [0.62] * 8, gap="small")
        cols[0].markdown(f"**{r}**")
        for c in range(8):
            key = f"g{r}_{c}"
            if key not in st.session_state:
                st.session_state[key] = bool(st.session_state.px[r, c])
            v = cols[c + 1].toggle(f"{r}-{c}", key=key, label_visibility="collapsed")
            if v != bool(st.session_state.px[r, c]):
                st.session_state.px[r, c] = 1.0 if v else 0.0
                changed = True
    x = st.session_state.px
    st.divider()
    if x.sum() == 0:
        st.info("格子还空着——点亮几个试试，比如竖着点亮中间一列再顶上横两格写个 '7'")
    else:
        predict_and_show(x)
        st.caption("注意：像素格子是二值的（0 或 1），训练数据却是平滑灰度。"
                   "置信度偏低是正常的'域偏移'——这就是第3课增强实验讲的分布错位，只不过方向反过来了。")
else:
    up = st.file_uploader("上传你在白纸上的手写数字照片（黑笔写，光线均匀）",
                          type=["png", "jpg", "jpeg", "webp"])
    if up is None:
        st.info("没照片也行：手机备忘录/画图板里黑笔白底写一个，截图上传即可。")
    else:
        img = Image.open(up)
        x = canvas_to_input(img.convert("RGBA"))
        if x is None:
            st.error("整张图像是白的/太浅——找不到笔画，检查是不是反了色（应为白底黑字）")
        else:
            predict_and_show(x)

st.divider()
st.markdown(
    "**课后挑战**：① 涂一个 '1'（就是中间竖一条），看它会不会把细长竖线认成别的；"
    "② 拍照时把手也拍进去一点，看包围盒被手影撑大后会发生什么。")
