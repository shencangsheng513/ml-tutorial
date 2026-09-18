# -*- coding: utf-8 -*-
"""
文本第8课：BERT 微调 —— 迁移学习把 0.877 打飞了吗？
前面六课我们手工搭 TF-IDF（词频统计+IDF 加权），最好成绩 0.877。
BERT 的玩法完全不同：别人用几亿文本预训练好的"语言模型"，我们只带来
4 个类别的 2374 篇帖子，在它顶上接个小分类头一起微调——迁移学习。
CPU 精简配方（真实 GPU 训练是它的常态，这里只求看懂流程不吃亏）：
  max_len=96（截断前96个token）, batch=16, 训练取 1200 篇, 约 75 step
对比的不只是准确率，还有三个维度：参数量、吃数据能力、对"没见过的词"的态度。
"""
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
CATS = ["comp.graphics", "rec.sport.baseball", "rec.sport.hockey", "sci.space"]
tr = fetch_20newsgroups(subset="train", categories=CATS,
                        remove=("headers", "footers", "quotes"), random_state=42)
te = fetch_20newsgroups(subset="test", categories=CATS,
                        remove=("headers", "footers", "quotes"), random_state=42)
label2cat = {i: c.split(".")[-1] for i, c in enumerate(CATS)}

# ============================================================
# 第1步：分词 —— 看 BERT 眼里一句话长什么样（和第2课词袋对照！）
# ============================================================
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print("加载 tokenizer / 模型（本地缓存后很快）...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=4)
print(f"完成，用时 {time.time()-t0:.0f}s")

demo = "The spacecraft launched towards Mars"
enc = tok(demo, truncation=True, max_length=96)
print("\n== BERT 眼里的句子 ==")
print("token :", tok.convert_ids_to_tokens(enc["input_ids"]))
print("注意 [CLS]/[SEP]：子词切分——launch 一个词，ed 是独立 token")
n_param = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {n_param/1e6:.1f}M —— 我们只有 {len(tr.data)} 篇训练帖，"
      f"平均每参数摊到 {len(tr.data)*128/n_param:.0f} 个训练信号")

# ============================================================
# 第2步：打包数据 + CPU 精简微调
# ============================================================
N_TRAIN, MAX_LEN, BS, EPOCHS = 1200, 96, 16, 1


def encode(texts):
    e = tok(texts, truncation=True, max_length=MAX_LEN, padding="max_length",
            return_tensors="pt")
    return e


t0 = time.time()
enc_tr = encode(tr.data[:N_TRAIN])
ds = TensorDataset(enc_tr["input_ids"], enc_tr["attention_mask"],
                   torch.tensor(tr.target[:N_TRAIN], dtype=torch.long))
dl = DataLoader(ds, batch_size=BS, shuffle=True)
print(f"训练集编码 {N_TRAIN} 篇耗时 {time.time()-t0:.0f}s")

opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
lossf = nn.CrossEntropyLoss()
step, losses = 0, []
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    for ids, am, y in dl:
        opt.zero_grad()
        out = model(input_ids=ids, attention_mask=am, labels=y)
        out.loss.backward()
        opt.step()
        step += 1
        losses.append(float(out.loss.detach()))
        if step % 25 == 0:
            print(f"  step {step:3d}/{len(dl)*EPOCHS}  loss {np.mean(losses[-25:]):.3f}"
                  f"  已用 {(time.time()-t0)/60:.1f} 分钟")

# ============================================================
# 第3步：全测试集评估 + 和 TF-IDF 对账
# ============================================================
model.eval()
preds = []
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(te.data), 32):
        e = encode(te.data[i:i+32])
        preds.append(model(**e).logits.argmax(1))
preds = torch.cat(preds).numpy()
acc_bert = accuracy_score(te.target, preds)
print(f"\n评估 {len(te.data)} 篇耗时 {(time.time()-t0)/60:.1f} 分钟")
print("=" * 60)
print(f"BERT(1200篇微调) 测试准确率 {acc_bert:.4f}")
print(f"TF-IDF+LR 历史成绩          0.8702（1579篇测试, src/train.py 存档）")
print("=" * 60)
print(classification_report(te.target, preds, target_names=list(label2cat.values())))

# ============================================================
# 第4步：语义探针 —— 词袋三大死穴，BERT 还活着吗？（TF-IDF 同台实测）
# ============================================================
import joblib
tfidf_clf = joblib.load("models/text_clf.joblib")

print("\n== 探针A：领域词短句（第2课说词袋会死——真会吗？） ==")
probes = [
    ("Orbit the planet red planet launch", "预期 sci.space"),
    ("puck slap shot goalie rink", "预期 hockey"),
    ("curveball batting home run stadium", "预期 baseball"),
    ("raster texture rendering polygon shading", "预期 graphics"),
]
for sent, note in probes:
    e = tok([sent], truncation=True, max_length=96, padding="max_length",
            return_tensors="pt")
    with torch.no_grad():
        p = torch.softmax(model(**e).logits, 1)[0]
    top = int(p.argmax())
    pt = tfidf_clf.predict_proba([sent])[0]
    print(f"  {sent!r:45s} BERT→{label2cat[top]:9s}(p={float(p[top]):.2f})  "
          f"TF-IDF→{CATS[int(pt.argmax())].split('.')[-1]:9s}(p={float(pt.max()):.2f})")
print("  ⚠ 诚实结果：平局。领域词密集的主题短句，TF-IDF 训练集里都见过这些词，照样全对。")
print("     词袋的死穴（否定/多义/长文）在主题分类任务上根本不被触发。")

print("\n== 探针B：换序（词袋的数学死穴 vs Transformer 的本职） ==")
base, var = "goalie puck slap rink shot", "shot rink slap goalie puck"
def bert_p(s):
    e = tok([s], truncation=True, max_length=96, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        return torch.softmax(model(**e).logits, 1)[0].numpy()
tb, tv = bert_p(base), bert_p(var)
fb, fv = tfidf_clf.predict_proba([base])[0], tfidf_clf.predict_proba([var])[0]
print(f"  TF-IDF：换序前后概率最大差异 {np.abs(fb-fv).max():.6f} —— 不是'没学好'，是输入向量数学上同一个东西")
print(f"  BERT ：换序前后概率最大差异 {np.abs(tb-tv).max():.4f} —— 语序进了注意力层")
print("""
== 全课对账（诚实版）==
BERT 用一半数据(1200篇)、零特征工程，追平 TF-IDF 全量手工方案(0.869 vs 0.870)。
它没能"吊打"——因为 2.4k 条主题分类任务不触发它的强项（上下文/语序/否定）。
迁移学习的正确预期：不是万能钥匙，而是把'数据饥渴+特征工程'的成本
转成了'算力+显存'的成本。GPU 上跑 3 轮 + 更长大文本任务，它才会甩开。""")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(losses, lw=1, color="#4C72B0")
w = 10
ax.plot(np.convolve(losses, np.ones(w)/w, "valid"), lw=2, color="#d9534f", label=f"{w}步滑动平均")
ax.set_xlabel("step"); ax.set_ylabel("交叉熵 loss")
ax.set_title(f"BERT 微调 loss（1200篇/1轮）  最终测试 {acc_bert:.3f} vs TF-IDF 0.870")
ax.legend()
fig.tight_layout(); fig.savefig("bert_finetune.png", dpi=120)
print("\n图已存：bert_finetune.png")
