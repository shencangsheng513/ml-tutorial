# -*- coding: utf-8 -*-
"""
第6课：突破词袋 —— 用 LSA（潜语义分析）给词建立"意义坐标"
第2课结尾的两个荒谬：反义句相似度0.84、同义句相似度0.00。
本课的解法思想（也是 word2vec/BERT 的共同思想）：
  一个词的意义 ≈ 它出现的语境。把 3 万个词的 TF-IDF 表压缩到 300 维，
  经常出现在相似语境里的词（good/excellent）会被压到相近的坐标上。
这一步叫 LSA（潜在语义分析），是 2013 年 word2vec 之前的经典方法，
不需要任何新依赖，且思想与 word2vec 一脉相承。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

CATS = ["comp.graphics", "rec.sport.baseball", "rec.sport.hockey", "sci.space"]
tr = fetch_20newsgroups(subset="train", categories=CATS, shuffle=True,
                        random_state=42, remove=("headers", "footers", "quotes"))
te = fetch_20newsgroups(subset="test", categories=CATS, shuffle=True,
                        random_state=42, remove=("headers", "footers", "quotes"))

# ============================================================
# 第0步：训练"词的意义坐标系"（纯无监督，不碰标签）
# ============================================================
print("=" * 60)
print("第0步：TF-IDF(3万维) --SVD--> 300维 词坐标")
print("=" * 60)
docs = tr.data + te.data          # 语料越大语境越丰富；只用文本不用标签
vec = TfidfVectorizer(ngram_range=(1, 1), min_df=5, sublinear_tf=True)
X = vec.fit_transform(docs)
# 词的坐标：对"词×文档"矩阵做 SVD（经典 LSA 的 term 表示）。
# 注意不能直接拿"文档×词"SVD 的 components_——那是另一组轴，词间余弦失真。
svd_w = TruncatedSVD(n_components=300, random_state=42)
W = svd_w.fit_transform(X.T)      # 词 × 300 ← 每个词得到一个"意义坐标"
words = vec.get_feature_names_out()
Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
wi = {w: i for i, w in enumerate(words)}
print(f"词表 {len(words)} 个词，每个词压缩成 300 维坐标")
print(f"压缩保留的信息量（解释方差）: {svd_w.explained_variance_ratio_.sum():.1%}")

def sim(a, b):
    return float(Wn[wi[a]] @ Wn[wi[b]])

def nearest(w, k=8):
    s = Wn @ Wn[wi[w]]
    idx = np.argsort(s)[::-1][1:k + 1]
    return [(words[i], round(float(s[i]), 3)) for i in idx]

# ============================================================
# 第1步：复检第2课的两桩"悬案"
# ============================================================
print("\n" + "=" * 60)
print("第1步：同义词还形同陌路吗？（词袋 vs LSA 余弦相似度）")
print("=" * 60)
pairs = [("good", "excellent"), ("nasa", "spacecraft"), ("goalie", "netminder"),
         ("image", "picture"), ("pitcher", "batting"), ("good", "bad")]
print(f"  {'词对':<22} 词袋   LSA")
for a, b in pairs:
    if a not in wi or b not in wi:
        print(f"  {a+' ~ '+b:<22} 0.00   词表外(min_df=5)")
        continue
    print(f"  {a+' ~ '+b:<22} {0.0:.2f}   {sim(a, b):.2f}")
print("→ 词袋里不同词永远 0.00（第2课句1vs句3的荒谬根源）；LSA 至少把")
print("  '语境相似的词'拉到正分区间，同词形变体（goalie/goalies、render/renders）最高。")
print("  但注意绝对值不大（0.1~0.3）——我们只有 4 千篇帖，语境统计很稀。")
print("  同一个思想喂给 10 亿句网页，就是 word2vec 把 good~excellent 拉到 0.7+ 的原因。")
print("  （诚实说明：good~bad 得分反而更高——LSA 编码的是'语境相似'，")
print("   褒贬反义词总出现在同样句式里。这是 LSA/词袋共同的盲区，")
print("   要靠语序建模（BERT 一类）才能分开。）")

print("\n每个词的'语义邻居'：")
for w in ["nasa", "goalie", "rendering"]:
    if w in wi:
        print(f"  {w:<10} → {', '.join(f'{n}({s})' for n, s in nearest(w, 6))}")

# ============================================================
# 第2步：这份"意义坐标"对分类有帮助吗？
# ============================================================
print("\n" + "=" * 60)
print("第2步：LSA 句子向量 vs TF-IDF，同一裁判 LogisticRegression")
print("=" * 60)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
m_tfidf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                        LogisticRegression(C=5.0, max_iter=5000))
m_lsa = make_pipeline(TfidfVectorizer(ngram_range=(1, 1), min_df=5, sublinear_tf=True),
                      TruncatedSVD(300), LogisticRegression(C=5.0, max_iter=5000))
s1 = cross_val_score(m_tfidf, tr.data, tr.target, cv=cv, n_jobs=1)
s2 = cross_val_score(m_lsa, tr.data, tr.target, cv=cv, n_jobs=1)
print(f"  TF-IDF 3万维稀疏  CV {s1.mean():.4f}")
print(f"  LSA 300维稠密     CV {s2.mean():.4f}")
m_lsa.fit(tr.data, tr.target)
print(f"  LSA 真实测试集    {accuracy_score(te.target, m_lsa.predict(te.data)):.4f}")
print("→ 300 维打平 3 万维：信息主要活在'语境结构'里，维度大半是冗余。")
print("  这就是表示学习的第一个震撼：压缩不是丢信息，是提纯信息。")
print("  也为第7课埋钩子：既然 300 维稠密向量这么好，让神经网络直接")
print("  从海量文本里学出来的词向量（word2vec/GloVe/BERT）会更好。")

# ============================================================
# 画图：语义邻居 + 方法对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
targets = [w for w in ["nasa", "goalie", "rendering"] if w in wi]
labels, vals, colors = [], [], []
cmap = {"nasa": "#4C72B0", "goalie": "#55A868", "rendering": "#C44E52"}
for w in targets:
    nb = nearest(w, 6)[::-1]
    for n, s in nb:
        labels.append(f"{w} → {n}")
        vals.append(s)
        colors.append(cmap[w])
axes[0].barh(labels, vals, color=colors)
axes[0].set_yticks(range(len(labels)), labels, fontsize=8)
axes[0].set_xlim(0, 1.0)
axes[0].set_xlabel("LSA 余弦相似度")
axes[0].set_title("词的语义邻居（语境相似 = 坐标相近）")

x = np.arange(2)
axes[1].bar(["TF-IDF 3万维", "LSA 300维"], [s1.mean(), s2.mean()],
            color=["#4C72B0", "#55A868"], yerr=[s1.std(), s2.std()], capsize=5)
axes[1].set_ylim(0.7, 0.95)
axes[1].set_ylabel("5折CV 准确率")
axes[1].set_title("维度砍掉 99%，准确率没掉")

fig.suptitle("第6课：LSA —— 给词建立意义坐标", fontsize=14)
fig.tight_layout()
fig.savefig("lsa_semantics.png", dpi=120)
print("\n图表已保存：lsa_semantics.png")
