# -*- coding: utf-8 -*-
"""
第3课：文本误差分析 —— 把判错的 200 条帖子捞出来"验尸"
呼应项目1进阶6：模型错了 12.7%，但"错 12.7%"不是行动指南，
"错在哪种帖子上、为什么错"才是。表格数据按特征分片，文本按
长度分片 + 按混淆对看原文。
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

CATS = ["comp.graphics", "rec.sport.baseball", "rec.sport.hockey", "sci.space"]
tr = fetch_20newsgroups(subset="train", categories=CATS, shuffle=True,
                        random_state=42, remove=("headers", "footers", "quotes"))
te = fetch_20newsgroups(subset="test", categories=CATS, shuffle=True,
                        random_state=42, remove=("headers", "footers", "quotes"))
names = te.target_names

clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LinearSVC(max_iter=10000))
clf.fit(tr.data, tr.target)
pred = clf.predict(te.data)
print(f"基线：测试准确率 {accuracy_score(te.target, pred):.3f}（第2课同款模型）")

# ============================================================
# 第1步：错误集中在哪对类别之间？
# ============================================================
print("\n" + "=" * 60)
print("第1步：混淆对排行 —— 谁和谁最'像'？")
print("=" * 60)
cm = confusion_matrix(te.target, pred)
pairs = []
for i in range(len(names)):
    for j in range(len(names)):
        if i != j and cm[i, j] > 0:
            pairs.append((names[i], names[j], int(cm[i, j])))
pairs.sort(key=lambda x: -x[2])
for true_c, pred_c, n in pairs[:6]:
    print(f"  真为 {true_c:<20} 却判成 {pred_c:<20} {n:>3} 条")
n_err = len(te.data) - int(cm.trace())
print(f"\n共 {n_err} 条判错。前6对占 {sum(p[2] for p in pairs[:6])} 条"
      f"（{sum(p[2] for p in pairs[:6]) / n_err:.0%}）→ 错误不是均匀撒的，是成堆的。")

# ============================================================
# 第2步：按长度分片 —— 短帖是不是重灾区？
# ============================================================
print("\n" + "=" * 60)
print("第2步：按帖长分片看准确率（呼应项目1的分片诊断）")
print("=" * 60)
nw = np.array([len(re.findall(r"[A-Za-z']+", t)) for t in te.data])
ok = (pred == te.target)
bins = [(0, 10), (10, 30), (30, 80), (80, 200), (200, 10**9)]
rows = []
for lo, hi in bins:
    m = (nw >= lo) & (nw < hi)
    rows.append([f"{lo}-{hi if hi < 10**9 else '+'}词", int(m.sum()), ok[m].mean()])
    print(f"  {rows[-1][0]:<12} {rows[-1][1]:>4} 条  准确率 {rows[-1][2]:.3f}")
print("→ 证据太少的超短帖，神仙也分不出——这类错误不值得修模型，")
print("  上线时该做的是'置信度太低就拒答/转人工'。")

# ============================================================
# 第3步：验尸 —— 看最大混淆对的判错原文
# ============================================================
print("\n" + "=" * 60)
print("第3步：抽查最大混淆对的判错帖（每对前3条，截断展示）")
print("=" * 60)
for true_c, pred_c, _ in pairs[:3]:
    ti, pj = names.index(true_c), names.index(pred_c)
    idx = [k for k in range(len(te.data)) if te.target[k] == ti and pred[k] == pj]
    print(f"\n--- {true_c} → 误判为 {pred_c}（共{len(idx)}条）---")
    for k in idx[:3]:
        body = " ".join(te.data[k].split())
        print(f"  [{nw[k]}词] {body[:160]}")

# ============================================================
# 第4步：模型自己怎么说？—— 用 decision_function 找"心虚"样本
# ============================================================
print("\n" + "=" * 60)
print("第4步：置信度与错误率的关系（上线时的'拒答'依据）")
print("=" * 60)
dec = clf.decision_function(te.data)
margin = dec.max(axis=1) - np.sort(dec, axis=1)[:, -2]  # 第一名-第二名差距
q = pd.qcut(margin, 4, labels=["Q1最不自信", "Q2", "Q3", "Q4最自信"])
tab = pd.DataFrame({"q": q, "ok": ok}).groupby("q", observed=True)["ok"].agg(["mean", "count"])
tab.columns = ["准确率", "条数"]
print(tab.round(3).to_string())
print("→ 判错大多发生在'心虚区'。应用层只要卡一个 margin 阈值，")
print("  把 Q1 的帖子交给人工，整体错误就能砍掉一大块——模型不用改。")

# ============================================================
# 画图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

lab = [r[0] for r in rows]
axes[0].bar(lab, [r[2] for r in rows], color=["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c", "#5cb85c"])
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("准确率")
axes[0].set_title("按帖长分片的准确率")
axes[0].axhline(accuracy_score(te.target, pred), ls="--", c="gray", lw=1)

top = pairs[:8][::-1]
axes[1].barh([f"{a.split('.')[-1]}→{b.split('.')[-1]}" for a, b, _ in top],
             [n for _, _, n in top], color="#4C72B0")
axes[1].set_title("Top8 混淆对（判错条数）")

ax = axes[2]
ax.scatter(np.arange(len(margin)), margin, c=np.where(ok, "#5cb85c", "#d9534f"), s=8)
ax.axhline(np.quantile(margin, 0.25), ls="--", c="gray", lw=1)
ax.set_yscale("symlog")
ax.set_title("决策置信度(margin)：绿=判对 红=判错")
ax.set_ylabel("margin (log)")
ax.set_xlabel("测试样本")

fig.suptitle("第3课：文本误差分析", fontsize=14)
fig.tight_layout()
fig.savefig("text_errors.png", dpi=120)
print("\n图表已保存：text_errors.png")
