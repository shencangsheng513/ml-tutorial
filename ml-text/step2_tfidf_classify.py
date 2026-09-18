# -*- coding: utf-8 -*-
"""
第2课：把"词"变成数字 —— TF-IDF + 第一个文本分类器
上节课结尾问：文本怎么喂给模型？模型只吃数字。
本课回答：① 最朴素的数法（词频）为什么不够 ② TF-IDF 聪明在哪
③ LinearSVC 冲 88%+ ④ 词袋模型的天花板在哪（为 embedding 埋钩子）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

CATS = ["comp.graphics", "rec.sport.baseball", "rec.sport.hockey", "sci.space"]
tr = fetch_20newsgroups(subset="train", categories=CATS, shuffle=True,
                        random_state=42, remove=("headers", "footers", "quotes"))
te = fetch_20newsgroups(subset="test", categories=CATS, shuffle=True,
                        random_state=42, remove=("headers", "footers", "quotes"))
names = tr.target_names
print(f"训练 {len(tr.data)} 条 / 测试 {len(te.data)} 条（都是干净正文）")

# ============================================================
# 第0步：TF-IDF 到底给每个词打多少分？
# ============================================================
print("\n" + "=" * 60)
print("第0步：先看 IDF —— 为什么 'the' 不值钱、'nasa' 值钱")
print("=" * 60)
vec = TfidfVectorizer()
Xtr = vec.fit_transform(tr.data)
idf = pd.Series(vec.idf_, index=vec.get_feature_names_out())
for w in ["the", "game", "image", "nasa", "hockey", "jpeg"]:
    if w in idf.index:
        df_n = int(np.asarray(Xtr[:, vec.vocabulary_[w]].todense()).ravel().astype(bool).sum())
        print(f"  {w:<7} 出现在 {df_n:>4}/{len(tr.data)} 篇里 → idf = {idf[w]:.2f}")
print("→ idf ≈ log(总篇数/出现该词的篇数)（sklearn 加了平滑，原理相同）。")
print("  'the' 几乎篇篇都有 → idf 跌到地板，等于没有区分度；")
print("  'nasa' 只在航天帖出现 → 高分。TF-IDF = 词频 × 区分度。")

# ============================================================
# 第1步：词频 vs TF-IDF，同一裁判（LinearSVC）
# ============================================================
print("\n" + "=" * 60)
print("第1步：三种'数词'方法同台竞技")
print("=" * 60)
results = {}


def run(tag, vectorizer):
    clf = make_pipeline(vectorizer, LinearSVC(max_iter=10000))
    clf.fit(tr.data, tr.target)
    acc = accuracy_score(te.target, clf.predict(te.data))
    results[tag] = acc
    print(f"  {tag:<28} 测试准确率 {acc:.3f}")
    return clf


run("词频 CountVectorizer", CountVectorizer())
clf_tfidf = run("TF-IDF（单词）", TfidfVectorizer())
clf_bigram = run("TF-IDF（单词+双词）", TfidfVectorizer(ngram_range=(1, 2)))
print("→ 词频→TF-IDF 的提升：同样一篇帖，'the'出现50次不该比'nasa'出现")
print("  1次更有分量。双词组(bigram)再涨一点：'New Zealand' 不再被拆成")
print("  'New'+'Zealand'，'not good' 至少被当成一个整体看了。")

# ============================================================
# 第2步：细看最好模型的成绩单
# ============================================================
print("\n" + "=" * 60)
print("第2步：TF-IDF(1,2) + LinearSVC 的成绩单")
print("=" * 60)
pred = clf_bigram.predict(te.data)
print(classification_report(te.target, pred, target_names=names, digits=3))

# 每个类别权重最高的10个词（模型"眼里"每类长什么样）
vec_used = clf_bigram.named_steps["tfidfvectorizer"]
model = clf_bigram.named_steps["linearsvc"]
print("各类别的'证据词'（模型权重 top10）：")
for k, name in enumerate(names):
    top = model.coef_[k].argsort()[-10:][::-1]
    words = [vec_used.get_feature_names_out()[i] for i in top]
    print(f"  {name:<20} {' '.join(words)}")

# ============================================================
# 第3步：词袋的天花板 —— 两个致命盲区
# ============================================================
print("\n" + "=" * 60)
print("第3步：词袋模型(BOW)看不见什么？")
print("=" * 60)
probe = TfidfVectorizer(ngram_range=(1, 1))
P = probe.fit_transform([
    "This movie is good",
    "This movie is not good",
    "That film was excellent",
])
sim = (P @ P.T).toarray()
print(f"  句1 'This movie is good'    vs")
print(f"  句2 'This movie is not good'  → 余弦相似度 {sim[0, 1]:.2f}")
print(f"  句3 'That film was excellent' → 余弦相似度 {sim[0, 2]:.2f}")
print("→ 荒谬之处：意思完全相反的句1/句2 相似度 0.8+（只差一个'not'的权重），")
print("  意思完全相同的句1/句3 相似度却是 0（good≠excellent，词面不重叠）。")
print("  盲区①不看语序（词袋=把词装进袋子摇匀）②不懂同义词。")
print("  修法：让'意思相近的词'在向量空间里也相近 → 词向量/预训练模型")
print("  （BERT 那一系），这是后面课程的伏笔。")

# ============================================================
# 画图：左=方法对比条形图 右=混淆矩阵
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
bars = list(results)
axes[0].barh(bars, [results[b] for b in bars],
             color=["#999999", "#4C72B0", "#55A868"])
axes[0].axvline(0.25, ls="--", c="red", lw=1)
axes[0].text(0.255, 0.05, "瞎猜 25%", color="red", fontsize=9)
axes[0].set_xlim(0.2, 1.0)
axes[0].set_xlabel("测试准确率")
axes[0].set_title("数词方法对比（裁判都是 LinearSVC）")

ConfusionMatrixDisplay.from_predictions(
    te.target, pred, display_labels=[n.split(".")[-1] for n in names],
    cmap="Blues", ax=axes[1], colorbar=False)
axes[1].set_title("TF-IDF(1,2)+LinearSVC 混淆矩阵")
fig.suptitle("第2课：TF-IDF 文本分类", fontsize=14)
fig.tight_layout()
fig.savefig("tfidf_classify.png", dpi=120)
print("\n图表已保存：tfidf_classify.png")
