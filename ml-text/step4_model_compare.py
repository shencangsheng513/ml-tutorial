# -*- coding: utf-8 -*-
"""
第4课：模型对比 + 交叉验证调参 —— 文本任务的"老将"和"新兵"
呼应项目1进阶1/2：同样用 5折CV 当裁判。但文本有个反直觉的结论：
项目1里大杀四方的随机森林/XGBoost，在高维稀疏的文本空间里反而打不过
最朴素的线性模型。为什么？本课用实验说话。
另一个呼应：Vectorizer 必须放进 Pipeline——否则 idf 是在全量数据上算的，
CV 的验证折"提前见过"自己的统计量，又是一次泄漏。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def tfidf(**kw):
    return TfidfVectorizer(ngram_range=(1, 2), **kw)


# ============================================================
# 第1步：五个模型同台（5折CV，vectorizer 都在折内 fit）
# ============================================================
print("=" * 60)
print("第1步：模型对比 —— 线性模型 vs 朴素贝叶斯 vs 随机森林")
print("=" * 60)
models = {
    "MultinomialNB": make_pipeline(tfidf(), MultinomialNB()),
    "ComplementNB": make_pipeline(tfidf(), ComplementNB()),
    "LogisticRegression": make_pipeline(tfidf(), LogisticRegression(max_iter=2000)),
    "LinearSVC": make_pipeline(tfidf(), LinearSVC(max_iter=10000)),
    "RandomForest(SVD100)": make_pipeline(tfidf(min_df=5), TruncatedSVD(100),
                                          RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=42)),
}
cv_scores = {}
for name, m in models.items():
    s = cross_val_score(m, tr.data, tr.target, cv=cv, n_jobs=1)
    cv_scores[name] = s
    print(f"  {name:<22} CV {s.mean():.3f} ± {s.std():.3f}")
print("→ 结论①：文本是高维稀疏问题（几万维、每篇只碰到几百维），")
print("  线性可分性很好，所以简单线性模型就是主力；树模型要把空间切成")
print("  一个个盒子，在高维稀疏空间反而切不动（RF 还得先 SVD 降维才能跑）。")
print("→ 结论②：NB 假设'词与词独立'——第2课刚证明这假设很荒谬")
print("  （'New' 和 'Zealand' 显然相关），但它照样接近满分。")
print("  好模型不要求假设是真的，只要'错得有用'（Box: 所有模型都是错的）。")

# ============================================================
# 第2步：调 Vectorizer 的旋钮（比调模型更值钱）
# ============================================================
print("\n" + "=" * 60)
print("第2步：文本任务的特征工程 = 调'怎么数词'")
print("=" * 60)
grid = []
for min_df in [1, 2, 5]:
    for sublinear in [False, True]:
        m = make_pipeline(tfidf(min_df=min_df, sublinear_tf=sublinear), LinearSVC())
        s = cross_val_score(m, tr.data, tr.target, cv=cv, n_jobs=1)
        grid.append((min_df, sublinear, s.mean()))
        print(f"  min_df={min_df:<2} sublinear_tf={str(sublinear):<6} CV {s.mean():.4f}")
best_md, best_sub, _ = max(grid, key=lambda r: r[2])
print(f"→ min_df 过滤'只在一两篇出现过的词'（噪声维度）；")
print(f"  sublinear_tf 把词频换成 1+log(tf)，防止复读机帖刷权重。")
print(f"  最优组合: min_df={best_md}, sublinear_tf={best_sub}")

# ============================================================
# 第3步：调 C（正则强度）—— 项目1老朋友，换个战场
# ============================================================
print("\n" + "=" * 60)
print(f"第3步：LinearSVC 的 C 值扫描（固定 min_df={best_md}, sublinear_tf={best_sub}）")
print("=" * 60)
c_grid = [0.01, 0.1, 0.5, 1.0, 5.0]
c_scores = []
for C in c_grid:
    m = make_pipeline(tfidf(min_df=best_md, sublinear_tf=best_sub), LinearSVC(C=C, max_iter=10000))
    s = cross_val_score(m, tr.data, tr.target, cv=cv, n_jobs=1)
    c_scores.append(s.mean())
    print(f"  C={C:<5} CV {s.mean():.4f}")
best_C = c_grid[int(np.argmax(c_scores))]

# ============================================================
# 第4步：总决赛 —— 调参后的模型 vs 第2课基线，都上真实考卷
# ============================================================
print("\n" + "=" * 60)
print("第4步：真实测试集上的总决赛")
print("=" * 60)
final = make_pipeline(tfidf(min_df=best_md, sublinear_tf=best_sub), LinearSVC(C=best_C, max_iter=10000))
final.fit(tr.data, tr.target)
acc_base = accuracy_score(te.target, make_pipeline(tfidf(), LinearSVC(max_iter=10000))
                          .fit(tr.data, tr.target).predict(te.data))
acc_final = accuracy_score(te.target, final.predict(te.data))
print(f"  第2课基线            测试 {acc_base:.3f}")
print(f"  调参后(C={best_C}, min_df={best_md}, sublinear) 测试 {acc_final:.3f}")
print("→ 提升不大？正常。第2课 0.873 已接近这套表示的上限，剩下的错误是")
print("  第3课验过尸的'证据不足帖'和'词汇邻居帖'——调参修不了表示的天花板。")

# ============================================================
# 画图
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].boxplot(list(cv_scores.values()), tick_labels=list(cv_scores))
axes[0].set_ylabel("5折CV 准确率")
axes[0].set_title("模型对比（箱体=折间波动）")
axes[0].tick_params(axis="x", rotation=20)
axes[0].axhline(0.25, ls="--", c="red", lw=1)

axes[1].plot([str(c) for c in c_grid], c_scores, "o-", color="#4C72B0")
axes[1].set_xlabel("C（越大正则越弱，越敢贴合训练数据）")
axes[1].set_ylabel("5折CV 准确率")
axes[1].set_title("C 值扫描曲线")
fig.suptitle("第4课：模型对比与调参", fontsize=14)
fig.tight_layout()
fig.savefig("model_compare.png", dpi=120)
print("\n图表已保存：model_compare.png")
