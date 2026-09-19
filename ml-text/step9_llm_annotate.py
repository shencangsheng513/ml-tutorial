# -*- coding: utf-8 -*-
"""
文本第9课：大模型 API 当"外挂"——标注、特征、数据三用法
前面八课全是"我们教模型"，这一课反过来：花点小钱调用云端 LLM，
看它能给老管线带来什么。三个实验（都控量省钱，全跑约 1-2 元）：
  A) Zero-shot 分类器：LLM 直接读帖报类别 —— 它值 0.870 吗？
  B) LLM 特征工厂：让 LLM 抽取结构化信号（领域词表算不出来的高阶特征），
     拼进 TF-IDF，看老模型能涨几点 —— "接回本项目管线"
  C) LLM 改写增强：让 LLM 换一种说法复写训练帖（语义不变的"平移增强"），
     呼应 CV第3课：这个变换匹配真实分布吗？——先猜再测
用法（需要 key）：
  set DASHSCOPE_API_KEY=sk-xxx
  C:\\Users\\shenh\\miniconda3\\python.exe step9_llm_annotate.py
"""
import json
import os
import sys
import time
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
if not KEY:  # 首选百炼 CLI 官方存储（bl auth login 写入），key 不经过命令行/仓库
    bl_cfg = Path.home() / ".bailian" / "config.json"
    if bl_cfg.exists():
        _c = json.loads(bl_cfg.read_text(encoding="utf-8"))
        KEY = _c.get("api_key")
        if _c.get("base_url"):
            BASE = _c["base_url"].rstrip("/") + "/compatible-mode/v1"
if not KEY:
    kf = Path(__file__).with_name(".dashscope_key")  # 兜底：本地密钥文件（已 gitignore）
    if kf.exists():
        KEY = kf.read_text(encoding="utf-8").strip()
if not KEY:
    sys.exit("缺少 API key：运行 bl auth login --api-key sk-... 或 set DASHSCOPE_API_KEY=sk-...")
MODEL = os.environ.get("LLM_MODEL", "qwen-plus")

from openai import OpenAI
client = OpenAI(api_key=KEY, base_url=BASE)

CATS = ["comp.graphics", "rec.sport.baseball", "rec.sport.hockey", "sci.space"]
SHORT = {"comp.graphics": "graphics", "rec.sport.baseball": "baseball",
         "rec.sport.hockey": "hockey", "sci.space": "space"}
tr = fetch_20newsgroups(subset="train", categories=CATS,
                        remove=("headers", "footers", "quotes"), random_state=42)
te = fetch_20newsgroups(subset="test", categories=CATS,
                        remove=("headers", "footers", "quotes"), random_state=42)
cat_names = [SHORT[c] for c in CATS]


def ask(prompt, tries=3):
    """带本地缓存：同一 prompt 只烧一次钱，崩了重跑不重复计费"""
    import hashlib
    key = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    cache_f = Path(__file__).with_name("llm_cache.json")
    if not hasattr(ask, "_cache"):
        ask._cache = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.exists() else {}
    if key in ask._cache:
        return ask._cache[key]
    for k in range(tries):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=640)
            ans = r.choices[0].message.content
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 + 3 * k)
            continue
        ask._cache[key] = ans
        if len(ask._cache) % 25 == 0:
            cache_f.write_text(json.dumps(ask._cache, ensure_ascii=False), encoding="utf-8")
        return ans
    return ""


def trim(text, n=1600):
    return text[:n]


# ============================================================
# 实验A：Zero-shot —— LLM 裸考我们的考卷（抽150篇，150次调用）
# ============================================================
print("=" * 60)
print(f"A) {MODEL} zero-shot 分类（150篇抽考）")
print("=" * 60)
idx = list(range(0, len(te.data), max(1, len(te.data) // 150)))[:150]
preds, fails = [], 0
for n, i in enumerate(idx):
    q = (f"下面帖子属于哪一类？只回答一个词：{','.join(cat_names)}。\n\n"
         f"---\n{trim(te.data[i])}\n---")
    ans = ask(q).lower()
    hit = [c for c in cat_names if c in ans]
    if len(hit) == 1:
        preds.append(CATS.index(f"{[c for c in CATS if SHORT[c]==hit[0]][0]}"))
    else:
        preds.append(-1); fails += 1
ok = [p == t for p, t in zip(preds, te.target[idx])]
acc_zs = sum(ok) / len(ok)
print(f"LLM zero-shot 准确率 {acc_zs:.3f}（无法判定 {fails} 篇按错计）")
print(f"我们的 TF-IDF+LR      0.870 —— 提示：LLM 没看过一张训练帖\n")

# ============================================================
# 实验B：特征工厂 —— LLM 抽 4 个高阶信号，拼进老管线
# ============================================================
print("=" * 60)
print("B) LLM 特征工厂：抽'工具类'信号拼回 TF-IDF")
print("=" * 60)
# 让 LLM 判断：是否求助帖 / 是否含代码或命令行 / 体育规则讨论 / 数学公式
FEATS = ["asking_help", "has_snippet", "rules_chat", "has_math"]
prompt_b = ("分析帖子，输出 JSON（只输出JSON）：" + json.dumps({f: "0或1" for f in FEATS}, ensure_ascii=False)
            + "\nasking_help=作者在求助; has_snippet=含代码/配置/邮件头片段; rules_chat=讨论规则赛制; has_math=含公式或数值计算\n---\n__TEXT__\n---")


def llm_features(texts, n=None):
    out = []
    for t in (texts if n is None else texts[:n]):
        raw = ask(prompt_b.replace("__TEXT__", trim(t, 900)))
        s = raw.find("{"); e = raw.rfind("}")
        try:
            j = json.loads(raw[s:e + 1])
            out.append([int(bool(j.get(f, 0))) for f in FEATS])
        except Exception:
            out.append([0] * len(FEATS))
    return out


import numpy as np
N_B = 300   # 成本闸：只给前300篇训练帖+300篇测试帖造特征
t0 = time.time()
Xtr_llm = np.array(llm_features(tr.data, N_B))
print(f"  训练侧 {N_B} 篇完成，{time.time()-t0:.0f}s")
Xte_llm = np.array(llm_features(te.data, N_B))
print(f"  测试侧完成，总耗时 {(time.time()-t0)/60:.1f} 分钟")

# 直接在矩阵层面比：纯TF-IDF vs TF-IDF+LLM特征
import scipy.sparse as sp
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
Atr = tf.fit_transform(tr.data[:N_B]); Ate = tf.transform(te.data[:N_B])
lr = LogisticRegression(C=5.0, max_iter=5000)
X_join_tr = sp.hstack([Atr, sp.csr_matrix(Xtr_llm.astype(float))])
X_join_te = sp.hstack([Ate, sp.csr_matrix(Xte_llm.astype(float))])
lr.fit(X_join_tr, tr.target[:N_B])
acc_join = accuracy_score(te.target[:N_B], lr.predict(X_join_te))
lr.fit(Atr, tr.target[:N_B])
acc_plain = accuracy_score(te.target[:N_B], lr.predict(Ate))
print(f"  同样{N_B}篇：纯TF-IDF {acc_plain:.4f}  vs  TF-IDF+LLM特征 {acc_join:.4f}")
print(f"  单看特征增益：{dict(zip(FEATS, Xtr_llm.mean(0).round(3)))}（训练侧出现率）\n")

# ============================================================
# 实验C：LLM 同义改写增强 —— 语义不变，词汇全变（先猜再测）
# ============================================================
print("=" * 60)
print("C) LLM 改写增强：把训练集最小的类各改写 60 篇（4×60=240 次调用）")
print("=" * 60)
rng = np.random.default_rng(42)
aug_texts, aug_labels = [], []
for ci, c in enumerate(CATS):
    sub = [t for t, l in zip(tr.data, tr.target) if l == ci]
    pick = [sub[j] for j in rng.choice(len(sub), 60, replace=False)]
    for t in pick:
        rew = ask("把下面帖子用另一种写法复述一遍，保留全部技术细节和主题，"
                  "不要添加新信息，长度相近，直接输出正文：\n---\n" + trim(t, 900) + "\n---")
        if 40 < len(rew) < 4000:
            aug_texts.append(rew); aug_labels.append(ci)
print(f"  产出 {len(aug_texts)} 篇改写帖")
if len(aug_texts) >= 100:
    Xa = sp.vstack([Atr, tf.transform(aug_texts)])
    ya = np.concatenate([tr.target[:N_B], aug_labels])
    lr2 = LogisticRegression(C=5.0, max_iter=5000).fit(Xa, ya)
    acc_aug = accuracy_score(te.target[:N_B], lr2.predict(Ate))
    print(f"  原300篇          测试 {acc_plain:.4f}")
    print(f"  300篇+LLM改写     测试 {acc_aug:.4f}  Δ={100*(acc_aug-acc_plain):+.1f}pp")
    print("  CV第3课教训迁移成功了吗？改写=匹配'部署时用户会换说法'的真实变化，"
          "与随机平移像素的'造假变化'本质不同——数字见真章。")
print("\n全部完成。把输出记进复盘文档『收官四件套』之后。")
Path("llm_cache.json").write_text(json.dumps(ask._cache, ensure_ascii=False), encoding="utf-8")
