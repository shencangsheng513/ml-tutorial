# -*- coding: utf-8 -*-
"""
第7课：word2vec —— 把第6课没做成的事做成
第6课用 LSA（线性代数）给词建坐标，同义词相似度只有 0.1~0.3，
good~bad 还反超 good~excellent。本课换神经网络的正主 word2vec：
  思想：一个词 = 它的上下文预测器（跳过gram预测邻居词）。
语料仍只有 4 千篇帖（word2vec 想吃饱需要百万级句子），
所以预期：比 LSA 好，但别指望教科书级的 king-queen 类比。
本课同时是一次"给结论配得上证据"的练习。
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from gensim.models import Word2Vec

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

CATS = ["comp.graphics", "rec.sport.baseball", "rec.sport.hockey", "sci.space"]
tr = fetch_20newsgroups(subset="train", categories=CATS, shuffle=True,
                        random_state=42, remove=("headers", "footers", "quotes"))
te = fetch_20newsgroups(subset="test", categories=CATS, shuffle=True,
                        random_state=42, remove=("headers", "footers", "quotes"))

# ============================================================
# 第0步：训练（无监督：只用文字，不用标签）
# ============================================================
print("=" * 60)
print("第0步：训练 word2vec（skip-gram, 100维, 窗口5）")
print("=" * 60)
sents = [[w for w in re.findall(r"[a-z']{2,}", d.lower()) if w not in ("the", "a", "an")]
         for d in (tr.data + te.data)]
n_tok = sum(map(len, sents))
print(f"语料 {len(sents)} 篇 / {n_tok} 个词（维基百科是它的几万倍，先降低预期）")
model = Word2Vec(sents, vector_size=100, window=5, min_count=5,
                 epochs=20, seed=42, workers=2)
model.wv.key_to_index  # 触发作词表
print(f"词表 {len(model.wv)} 个词，每个词 100 维")

# ============================================================
# 第1步：复检第6课那张"词对考卷"（LSA vs word2vec）
# ============================================================
print("\n" + "=" * 60)
print("第1步：同义词考卷 —— LSA 得分 vs word2vec 得分")
print("=" * 60)
lsa_scores = {"good~excellent": 0.24, "nasa~spacecraft": 0.04, "image~picture": 0.15,
              "goalie~netminder": None, "pitcher~batting": 0.05, "good~bad": 0.26}
rows = []
for a, b in [("good", "excellent"), ("nasa", "spacecraft"), ("image", "picture"),
             ("goalie", "netminder"), ("pitcher", "batting"), ("good", "bad")]:
    key = f"{a}~{b}"
    if a in model.wv and b in model.wv:
        s = float(model.wv.similarity(a, b))
    else:
        s = np.nan
    rows.append((key, lsa_scores.get(key), s))
    lsa_txt = f"{lsa_scores.get(key):.2f}" if lsa_scores.get(key) is not None else "词表外"
    print(f"  {key:<20} LSA {lsa_txt:>6}   w2v {s if not np.isnan(s) else float('nan'):.2f}")
print("→ 小语料下 w2v 也磕磕绊绊（有的词 min_count=5 直接出局），")
print("  但方向对了：语境预测天然把'可互换的词'拉近。")

print("\n每个词的邻居（对比第6课 LSA 的邻居表）：")
for w in ["nasa", "goalie", "image"]:
    if w in model.wv:
        nb = model.wv.most_similar(w, topn=6)
        print(f"  {w:<8} → {', '.join(f'{n}({s:.2f})' for n, s in nb)}")

# ============================================================
# 第2步：词向量代数 —— 类比推理成色几何？
# ============================================================
print("\n" + "=" * 60)
print("第2步：类比推理 A 之于 B 如同 C 之于 ?")
print("=" * 60)
analogies = [
    (["nhl", "mlb"], ["hockey"]),            # nhl之于冰球 如 mlb之于?
    (["space", "graphics"], ["nasa"]),       # 航天之于图形 如 nasa之于?
    (["pitcher", "goalie"], ["baseball"]),   # 投手之于击球 如 ?(棒球→冰球)
]
for pos, neg in analogies:
    miss = [w for w in pos + neg if w not in model.wv]
    if miss:
        print(f"  {pos} - {neg}: 词 {miss} 不在词表，跳过")
        continue
    r = model.wv.most_similar(positive=pos, negative=neg, topn=3)
    print(f"  {'+'.join(pos)} - {'+'.join(neg)} → {[(n, round(s, 2)) for n, s in r]}")
print("→ 有的类比在小语料上会失败——不是思想错了，是'喂的词例不够多'。")
print("  word2vec 论文里惊艳的 king-queen 类比，用的是几十亿词的语料。")

# ============================================================
# 第3步：用词向量做分类，干得过 TF-IDF 吗？（诚实环节）
# ============================================================
print("\n" + "=" * 60)
print("第3步：平均词向量 + 逻辑回归 vs TF-IDF + 逻辑回归")
print("=" * 60)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

def sent_vec(text):
    vs = [model.wv[w] for w in re.findall(r"[a-z']{2,}", text.lower()) if w in model.wv]
    return np.mean(vs, axis=0) if vs else np.zeros(100)

Xw_tr = np.vstack([sent_vec(d) for d in tr.data])
cv = StratifiedKFold(5, shuffle=True, random_state=42)
s_w2v = cross_val_score(LogisticRegression(max_iter=3000), Xw_tr, tr.target, cv=cv)
from sklearn.pipeline import make_pipeline
m_tf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                     LogisticRegression(C=5.0, max_iter=5000))
s_tf = cross_val_score(m_tf, tr.data, tr.target, cv=cv)
print(f"  平均词向量(100维稠密)  CV {s_w2v.mean():.4f}")
print(f"  TF-IDF(3万维稀疏)      CV {s_tf.mean():.4f}")
print("→ 震撼反转：精心学的词向量，简单平均后反而输给老老实实的 TF-IDF！")
print("  原因：平均丢掉了词序和'哪个词重要'（一篇帖里 nasa 出现1次和")
print("  出现20次被平均抹平了）。TF-IDF 的'计数+加权'恰好保住了这些。")
print("  词向量真正的用法是 BERT 那类'上下文嵌入'——同一个词在不同句子里")
print("  有不同向量，平均与语序问题一起解决。这是本项目的终点站方向。")

# ============================================================
# 画图：词对对比 + 分类对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ks = [r[0] for r in rows if not np.isnan(r[2])]
w2v_s = [r[2] for r in rows if not np.isnan(r[2])]
lsa_s = [r[1] if r[1] is not None else 0 for r in rows if not np.isnan(r[2])]
x = np.arange(len(ks))
axes[0].bar(x - 0.2, lsa_s, 0.4, label="LSA", color="#999999")
axes[0].bar(x + 0.2, w2v_s, 0.4, label="word2vec", color="#55A868")
axes[0].set_xticks(x, ks, rotation=25, fontsize=9)
axes[0].axhline(0, lw=0.5, c="k")
axes[0].set_ylabel("余弦相似度")
axes[0].set_title("词对相似度：LSA vs word2vec")
axes[0].legend()

axes[1].bar(["TF-IDF 3万维", "平均词向量 100维"], [s_tf.mean(), s_w2v.mean()],
            color=["#4C72B0", "#d9534f"], yerr=[s_tf.std(), s_w2v.std()], capsize=5)
axes[1].set_ylim(0.4, 1.0)
axes[1].set_ylabel("5折CV 准确率")
axes[1].set_title("诚实反转：平均词向量输给 TF-IDF")
fig.suptitle("第7课：word2vec 实测", fontsize=14)
fig.tight_layout()
fig.savefig("word2vec.png", dpi=120)
print("\n图表已保存：word2vec.png")
