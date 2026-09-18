# -*- coding: utf-8 -*-
"""
MLflow 实验跟踪：把散落在各课的"跑完就忘"的训练，升级成可查账的实验记录。
为什么需要：我们文本课调过 min_df×sublinear×C 一堆组合，图像课换过 LR/SVM/MLP/CNN，
全靠 print 和记忆。真实团队用 MLflow 记三样东西：参数(怎么配的)、指标(效果如何)、
产物(模型文件/图)。之后一行代码排序对比所有实验。
用法：
  C:\\Users\\shenh\\miniconda3\\python.exe mlflow-lab/track_projects.py   # 记录实验
  C:\\Users\\shenh\\miniconda3\\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow-lab/mlflow.db --port 5000   # 看账本
"""
import sys
from pathlib import Path

import numpy as np
import mlflow
from sklearn.datasets import load_digits
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, LinearSVC

ROOT = Path(__file__).resolve().parent.parent
LAB = Path(__file__).resolve().parent
mlflow.set_tracking_uri(f"sqlite:///{(LAB / 'mlflow.db').as_posix()}")
_client = mlflow.MlflowClient()


def use_exp(name):
    """建实验并指定产物目录（MLflow 3：artifact_location 只能在 create 时给）"""
    if _client.get_experiment_by_name(name) is None:
        _client.create_experiment(name, artifact_location=str(LAB / "artifacts"))
    mlflow.set_experiment(name)


rng = np.random.default_rng(42)

# ============================================================
# 实验一：文本分类（20newsgroups 4类）—— 三个历史选手重跑并记账
# ============================================================
from sklearn.datasets import fetch_20newsgroups
cats = ["comp.graphics", "rec.sport.baseball", "rec.sport.hockey", "sci.space"]
tr = fetch_20newsgroups(subset="train", categories=cats,
                        remove=("headers", "footers", "quotes"), random_state=42)
te = fetch_20newsgroups(subset="test", categories=cats,
                        remove=("headers", "footers", "quotes"), random_state=42)

use_exp("text-20newsgroups")
text_runs = [
    ("ComplementNB", make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True), ComplementNB())),
    ("LogisticRegression C=5", make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True), LogisticRegression(C=5.0, max_iter=5000))),
    ("LinearSVC", make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True), LinearSVC())),
]
for name, pipe in text_runs:
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"ngram": "(1,2)", "min_df": 1, "sublinear_tf": True})
        cv = cross_val_score(pipe, tr.data, tr.target, cv=3, scoring="accuracy", n_jobs=3)
        pipe.fit(tr.data, tr.target)
        test = accuracy_score(te.target, pipe.predict(te.data))
        mlflow.log_metrics({"cv_mean": cv.mean(), "cv_std": cv.std(), "test_acc": test})
        mlflow.log_artifact(str(ROOT / "ml-text" / "config.json"), "config")
        print(f"[text] {name:24s} CV {cv.mean():.4f}  test {test:.4f}")

# ============================================================
# 实验二：图像分类（digits）—— 含 PyTorch CNN，模型文件也入库
# ============================================================
d = load_digits()
X, y = d.data / 16.0, d.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

use_exp("digits-cv")
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

img_runs = [
    ("SVM-rbf", SVC(gamma="scale")),
    ("MLP-64", MLPClassifier((64,), max_iter=2000, random_state=42)),
    ("RF-200", RandomForestClassifier(200, random_state=42)),
]
for name, clf in img_runs:
    with mlflow.start_run(run_name=name):
        cv = cross_val_score(clf, Xtr, ytr, cv=3, scoring="accuracy", n_jobs=3)
        clf.fit(Xtr, ytr)
        test = accuracy_score(yte, clf.predict(Xte))
        mlflow.log_params({"train_n": len(Xtr), "test_n": len(Xte)})
        mlflow.log_metrics({"cv_mean": cv.mean(), "test_acc": test})
        print(f"[img]  {name:24s} CV {cv.mean():.4f}  test {test:.4f}")

# PyTorch CNN：结构同 ml-vision 第3课，训完把 state_dict 作为产物入库
import tempfile

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 2 * 2, 10))

    def forward(self, x):
        return self.net(x)


torch.manual_seed(42)
with mlflow.start_run(run_name="CNN-torch"):
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss()
    xt = torch.tensor(Xtr.reshape(-1, 8, 8), dtype=torch.float32).unsqueeze(1)
    yt = torch.tensor(ytr, dtype=torch.long)
    for ep in range(80):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = lossf(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte.reshape(-1, 8, 8), dtype=torch.float32).unsqueeze(1)).argmax(1).numpy()
    test = float(accuracy_score(yte, pred))
    mlflow.log_params({"epochs": 80, "lr": 3e-3, "arch": "2conv+fc"})
    mlflow.log_metrics({"test_acc": test})
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cnn_digits.pt"
        torch.save(model.state_dict(), p)
        mlflow.log_artifact(str(p), "model")
    print(f"[img]  {'CNN-torch':24s} --          test {test:.4f}")

# ============================================================
# 收官：一行代码对比全部实验（MLflow 的核心价值）
# ============================================================
df = mlflow.search_runs(experiment_names=["text-20newsgroups", "digits-cv"])
cols = ["tags.mlflow.runName", "metrics.cv_mean", "metrics.test_acc"]
print("\n== 全部实验排行 ==")
print(df[cols].sort_values("metrics.test_acc", ascending=False).to_string(index=False))
