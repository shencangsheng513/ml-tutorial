# -*- coding: utf-8 -*-
"""帖子自动归版助手 —— Streamlit 应用（端口 8503）
运行：python -m streamlit run app.py --server.port 8503
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load():
    pipe = joblib.load(ROOT / "models" / "text_clf.joblib")
    report = json.loads((ROOT / "models" / "report.json").read_text(encoding="utf-8"))
    return pipe, report


pipe, report = load()
classes = report["classes"]
thr = report["confidence_threshold"]

st.set_page_config(page_title="帖子归版助手", page_icon="🗂️", layout="wide")
st.title("🗂️ 帖子自动归版助手")
st.caption(f"训练集 {report['n_train']} 帖 / CV {report['cv_accuracy']:.1%} / "
           f"测试 {report['test_accuracy']:.1%}（20newsgroups 4版面）")

text = st.text_area("把帖子正文粘贴到这里（英文）", height=220,
                    placeholder="Does anyone know how to render a sphere with ray tracing in OpenGL?...")
col1, col2 = st.columns([1, 2])
go = col1.button("归类", type="primary", disabled=not text.strip())

if go:
    proba = pipe.predict_proba([text])[0]
    order = np.argsort(proba)[::-1]
    top1, top2 = classes[order[0]], classes[order[1]]
    conf = proba[order[0]] - proba[order[1]]

    with col1:
        st.metric("判为", top1)
        st.metric("置信度 (Top1-Top2)", f"{conf:.2f}")
        n_words = len(text.split())
        if n_words < 10:
            # 第3课分片结论：<10词的帖准确率仅52%，且模型对OOD输入反而"自信"
            st.error(f"仅 {n_words} 个词，证据不足——此判定基本不可信，请补充正文或转人工")
        elif conf < thr:
            st.warning(f"置信度低于阈值 {thr:.2f}：证据不足，建议人工复核（第3课学的'拒答'）")
        else:
            st.success("高置信度")

    with col2:
        df = pd.DataFrame({"版面": classes, "概率": proba}).sort_values("概率", ascending=False)
        st.bar_chart(df.set_index("版面"))
        st.subheader("模型看到的证据词")
        vec = pipe.named_steps["tfidf"]
        W = pipe.named_steps["clf"]
        k = classes.index(top1)
        try:
            feat_names = np.array(vec.get_feature_names_out())
            w = W.coef_[k] if W.coef_.shape[0] == len(classes) else W.coef_
            idx = np.argsort(w)[-12:][::-1]
            st.write("  ".join(f"`{feat_names[i]}`" for i in idx))
        except Exception:
            st.write("(略)")

with st.expander("批量自测：几条样例帖"):
    samples = [
        "The pitcher struck out the side in the ninth inning.",
        "NASA delayed the shuttle launch again due to a fuel leak.",
        "Anyone knows a good file format for storing 3D vertex data?",
        "The goalie played out of his net and the defense collapsed.",
    ]
    p = pipe.predict_proba(samples)
    st.table(pd.DataFrame({"帖子": samples,
                           "判定": [classes[i] for i in p.argmax(1)],
                           "置信度": [(lambda s: s[-1] - s[-2])(np.sort(r)) for r in p]}))
