# -*- coding: utf-8 -*-
"""step9-B 审计（零成本：ask() 全走 llm_cache.json，不发新请求）
问题：为什么拼 4 个 LLM 特征让 0.790 → 0.677？
查三件事：1) 特征-类别相关性 训练侧 vs 测试侧 是否错位
         2) 3折CV(训练侧内部) 能否提前预警这个坑
         3) 把 C=5.0 调小/去掉特征幅度，损失是否收窄（判断是不是"稠密特征压倒稀疏词频"）
"""
import json
import importlib.util
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# 只借用 step9 的缓存 ask()，不跑它主流程：手动装配等价函数
sys.argv = [sys.argv[0], "--audit"]  # 防呆
spec = importlib.util.spec_from_file_location("s9", Path(__file__).with_name("step9_llm_annotate.py"))

CATS = ["comp.graphics", "rec.sport.baseball", "rec.sport.hockey", "sci.space"]
tr = fetch_20newsgroups(subset="train", categories=CATS,
                        remove=("headers", "footers", "quotes"), random_state=42)
te = fetch_20newsgroups(subset="test", categories=CATS,
                        remove=("headers", "footers", "quotes"), random_state=42)
FEATS = ["asking_help", "has_snippet", "rules_chat", "has_math"]
SHORT = {"comp.graphics": "graphics", "rec.sport.baseball": "baseball",
         "rec.sport.hockey": "hockey", "sci.space": "space"}


def trim(t, n):
    return t[:n]


def make_prompt(t):
    return ("分析帖子，输出 JSON（只输出JSON）：" + json.dumps({f: "0或1" for f in FEATS}, ensure_ascii=False)
            + "\nasking_help=作者在求助; has_snippet=含代码/配置/邮件头片段; rules_chat=讨论规则赛制; has_math=含公式或数值计算\n---\n"
            + trim(t, 900) + "\n---")


cache = json.loads(Path(__file__).with_name("llm_cache.json").read_text(encoding="utf-8"))


def feat_vec(t):
    import hashlib
    k = hashlib.md5(make_prompt(t).encode("utf-8")).hexdigest()
    raw = cache.get(k)
    if raw is None:
        return None
    s, e = raw.find("{"), raw.rfind("}")
    try:
        j = json.loads(raw[s:e + 1])
        return [int(bool(j.get(f, 0))) for f in FEATS]
    except Exception:
        return [0] * 4


N_B = 300
Xtr = [feat_vec(t) for t in tr.data[:N_B]]
Xte = [feat_vec(t) for t in te.data[:N_B]]
miss_tr = sum(v is None for v in Xtr)
print(f"缓存命中：训练 {N_B-miss_tr}/{N_B}（未命中说明这些调用当时没跑成）")
Xtr = np.array([v if v else [0]*4 for v in Xtr], float)
Xte = np.array([v if v else [0]*4 for v in Xte], float)

# 1) 特征-类别错位：每个特征在各类的均值，训练 vs 测试并排
ytr, yte = tr.target[:N_B], te.target[:N_B]
print("\n[1] 特征出现率（按类）：训练侧 / 测试侧 —— 差距大=特征含义在两半数据里不一致")
for fi, f in enumerate(FEATS):
    row = []
    for ci in range(4):
        a = Xtr[ytr == ci, fi].mean() if (ytr == ci).sum() else np.nan
        b = Xte[yte == ci, fi].mean() if (yte == ci).sum() else np.nan
        row.append(f"{SHORT[CATS[ci]][:4]}:{a:.2f}/{b:.2f}")
    print(f"  {f:12}", "  ".join(row))

# 2) 内部CV能否预警（训练集3折，同一批特征）
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
A = tf.fit_transform(tr.data[:N_B])
lr = LogisticRegression(C=5.0, max_iter=5000)
cv_plain = cross_val_score(lr, A, ytr, cv=3, scoring="accuracy").mean()
J = sp.hstack([A, sp.csr_matrix(Xtr)])
cv_join = cross_val_score(lr, J, ytr, cv=3, scoring="accuracy").mean()
print(f"\n[2] 训练侧内部3折CV：纯TF-IDF {cv_plain:.4f}  vs  加LLM特征 {cv_join:.4f}")
Ate = tf.transform(te.data[:N_B])
print(f"    （对照测试集口径：0.7900 vs 0.6767 —— CV 与外考是否同向）")

# 3) 幅度假说：LLM特征缩到0.2权重再拼
for scale in [1.0, 0.5, 0.2]:
    Js = sp.hstack([A, sp.csr_matrix(Xtr * scale)])
    lr.fit(Js, ytr)
    acc = accuracy_score(yte, lr.predict(sp.hstack([Ate, sp.csr_matrix(Xte * scale)])))
    print(f"[3] 特征权重×{scale:.1f} -> 测试 {acc:.4f}")

# 4) 数据体检：前300篇的类别分布（fetch默认shuffle，但确认一下）
print("\n[4] 训练前300类别分布:", np.bincount(ytr), " 测试前300:", np.bincount(yte))
