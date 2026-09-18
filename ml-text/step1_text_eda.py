# -*- coding: utf-8 -*-
"""
第1课：拿到文本数据 + 看文本"长什么样"的 EDA
呼应项目1第1课：建模之前先看看数据。但文本的"看看"和表格完全不同——
表格看分布/关系，文本要看：长度、类别样本量、词、以及" Metadata 泄漏"。
"""
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

CATS = ["comp.graphics", "rec.sport.baseball", "rec.sport.hockey", "sci.space"]

# ---------- 第0步：同一份缓存，两种读法 ----------
# remove=() 保留邮件的原始头部；remove=('headers',...) 把它们删掉
dirty = fetch_20newsgroups(subset="train", categories=CATS,
                           shuffle=True, random_state=42, remove=())
clean = fetch_20newsgroups(subset="train", categories=CATS,
                           shuffle=True, random_state=42,
                           remove=("headers", "footers", "quotes"))
target_names = clean.target_names
y = clean.target

print("=" * 60)
print("第0步：为什么要 remove=('headers','footers','quotes')？")
print("=" * 60)
print("\n【未清理版本】第一条样本的前 6 行：")
for line in dirty.data[0].split("\n")[:6]:
    print("   ", line[:80])
print("\n【清理版本】第一条样本的前 6 行：")
for line in clean.data[0].split("\n")[:6]:
    print("   ", line[:80])

# 泄漏的证据：直接写类别名的只有少数（112条），metadata 的泄漏更隐蔽。
# 做一次对照实验：用"脏文本"（带headers）训练 vs "干净文本"训练，
# 统一在干净的测试集上考试。
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

te = fetch_20newsgroups(subset="test", categories=CATS,
                        shuffle=True, random_state=42,
                        remove=("headers", "footers", "quotes"))
headers_only = [t.split("\n\n", 1)[0] for t in dirty.data]  # 只留"信封"，不看"信"
print(f"\n【泄漏对照实验】考试统一用干净的测试集（{len(te.data)}条，只有正文）")
for name, corpus in [("只用信封(headers)", headers_only),
                     ("信封+正文(原始脏文本)", dirty.data),
                     ("只用正文(清理后)", clean.data)]:
    clf = make_pipeline(TfidfVectorizer(max_features=30000), LogisticRegression(max_iter=2000))
    clf.fit(corpus, y)
    tr = accuracy_score(y, clf.predict(corpus))
    acc = accuracy_score(te.target, clf.predict(te.data))
    print(f"  {name:<20} 训练集自评 {tr:.3f} → 干净考卷 {acc:.3f}")
print("→ '只看信封'就能考出几分——证明 headers 里藏着能猜类别的元数据")
print("  （谁发的、什么组织、FAQ编号）。而我们的应用上线后用户只会贴正文，")
print("  这种'信封分'在真实场景拿不到，留着 headers 只会让训练指标虚高、")
print("  学偏。这就是项目1讲过的数据泄漏，在文本数据里的样子。")

# ---------- 第1步：基本信息 ----------
print("\n" + "=" * 60)
print("第1步：基本信息（对应表格EDA的 df.info()）")
print("=" * 60)
print(f"样本数：{len(clean.data)}")
print(f"类别：{target_names}")
y = clean.target
class_counts = pd.Series([target_names[i] for i in y]).value_counts()
print("\n每类样本量：")
print(class_counts.to_string())
print("→ 四类数量接近（不平衡度 <10%），暂不需要特殊处理。")

# 文本长度
lens = np.array([len(t) for t in clean.data])
words = np.array([len(re.findall(r"[A-Za-z']+", t)) for t in clean.data])
print(f"\n字符长度：中位数 {np.median(lens):.0f}，最短 {lens.min()}，最长 {lens.max()}")
print(f"词数长度：中位数 {np.median(words):.0f}，最短 {words.min()}，最长 {words.max()}")

# ---------- 第2步：每类文本长度分布 ----------
print("\n" + "=" * 60)
print("第2步：各类别的'词数'分布")
print("=" * 60)
df = pd.DataFrame({"cat": [target_names[i] for i in y], "nwords": words})
print(df.groupby("cat")["nwords"].describe()[["count", "mean", "50%", "max"]].round(0).to_string())
print("→ 各类中位数接近（58~81 词），但最长帖差到几千词——长度区分度其实很弱。")
print("  真正的信号在'用了哪些词'里，这就是下一课 TF-IDF 要做的事。")

# ---------- 第3步：各类别的特色词 ----------
STOP = set("""a an the and or but if then this that these those i you he she we they it
of in on at to for from with by as is are was were be been am do does did not no so
s t re ve ll d m o o'clock its my your our their his her there here what when how
will can could would should may might have has had i'm don't can't won't""".split())

print("\n" + "=" * 60)
print("第3步：每个类别的高频词（去掉常见停用词后）")
print("=" * 60)
top_per_cat = {}
for c in CATS:
    tokens = []
    for t, label in zip(clean.data, y):
        if target_names[label] == c:
            tokens += [w.lower() for w in re.findall(r"[A-Za-z']{3,}", t)]
    cnt = Counter(w for w in tokens if w not in STOP)
    top_per_cat[c] = cnt.most_common(10)
    print(f"\n{c}:")
    print("   ", ", ".join(f"{w}({n})" for w, n in top_per_cat[c]))
print("\n→ 看词就知道每类在聊什么：graphics 类满是 image/render，")
print("  baseball 类是 team/season/player。这些区分度高的词，")
print("  就是第2课 TF-IDF 要喂给模型的东西。")

# ---------- 第4步：翻几条原文感受一下 ----------
print("\n" + "=" * 60)
print("第4步：随机看 4 条原文（每类第一条的前 200 字符）")
print("=" * 60)
for c in CATS:
    idx = int(np.argmax(y == target_names.index(c)))
    body = " ".join(clean.data[idx].split())
    print(f"\n[{c}]  {body[:200]}...")

# ---------- 画图 ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

box_data = [df.loc[df["cat"] == c, "nwords"].clip(upper=np.percentile(df["nwords"], 98)) for c in CATS]
axes[0].boxplot(box_data, tick_labels=[c.split(".")[-1] for c in CATS])
axes[0].set_title("各类别词数分布（截断到98分位）")
axes[0].set_ylabel("词数")

# 右图：每类前8个专属高频词的水平条形图（graphics vs baseball 对比）
a, b = CATS[0], CATS[1]
wa = dict(Counter({w: n for w, n in top_per_cat[a]}))
wb = dict(Counter({w: n for w, n in top_per_cat[b]}))
only_a = sorted(set(wa) - set(wb), key=wa.get, reverse=True)[:8]
only_b = sorted(set(wb) - set(wa), key=wb.get, reverse=True)[:8]
ys = list(range(len(only_a)))
axes[1].barh([len(only_b) + i for i in ys], [wa[w] for w in only_a], color="steelblue")
axes[1].barh(ys, [wb[w] for w in only_b], color="darkorange")
axes[1].set_yticks(ys + [len(only_b) + i for i in ys], only_b + only_a)
axes[1].set_title(f"专属高频词对比：{b}（橙）vs {a}（蓝）")
axes[1].set_xlabel("出现次数")

fig.suptitle("20newsgroups 文本 EDA（4 类）", fontsize=14)
fig.tight_layout()
fig.savefig("text_eda.png", dpi=120)
print("\n图表已保存：text_eda.png")
