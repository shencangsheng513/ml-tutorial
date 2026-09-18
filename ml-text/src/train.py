# -*- coding: utf-8 -*-
"""
第5课：工程化封装 —— python -m src.train 一键训练
用法：在 ml-text 目录下运行
    PYTHONIOENCODING=utf-8 python -m src.train
产出：models/text_clf.joblib + models/report.json
呼应项目2：训练/服务共用同一个 build_pipeline()，杜绝 train/serve skew。
一个细节：'转人工'的 margin 阈值不拿测试集定（那又是偷看考卷），
用 cross_val_predict 的折外预测来定——模型训练时没见过的预测分布。
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent
cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def build_pipeline():
    v = cfg["vectorizer"]
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=tuple(v["ngram_range"]),
                                  min_df=v["min_df"], sublinear_tf=v["sublinear_tf"])),
        # 应用要显示置信度 → 需要概率。LinearSVC 不给概率，这里用同为线性模型、
        # 概率直接的 LogisticRegression（第4课它比 SVC 低约2个点，换来上线可用的置信度，值）
        ("clf", LogisticRegression(C=cfg["model"]["C"], max_iter=cfg["model"]["max_iter"])),
    ])


def load_data(subset):
    return fetch_20newsgroups(
        subset=subset, categories=cfg["categories"], shuffle=True,
        random_state=cfg["random_state"], remove=tuple(cfg["remove"]))


def main():
    tr, te = load_data("train"), load_data("test")
    names = tr.target_names
    pipe = build_pipeline()

    cv = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=cfg["random_state"])
    proba_oof = cross_val_predict(pipe, tr.data, tr.target, cv=cv,
                                  method="predict_proba", n_jobs=1)
    cv_acc = accuracy_score(tr.target, proba_oof.argmax(1))

    # 折外置信度 = 第一名概率 - 第二名概率（应用里显示的就是它）
    srt = np.sort(proba_oof, axis=1)
    conf_oof = srt[:, -1] - srt[:, -2]
    conf_thr = float(np.quantile(conf_oof, 0.25))  # 最没底气的25%的门槛

    pipe.fit(tr.data, tr.target)
    test_acc = accuracy_score(te.target, pipe.predict(te.data))

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    joblib.dump(pipe, out / "text_clf.joblib")
    report = {
        "cv_accuracy": round(float(cv_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "n_train": len(tr.data), "n_test": len(te.data),
        "classes": list(names),
        "confidence_threshold": round(conf_thr, 4),
        "note": "confidence=Top1-Top2概率差；低于阈值应用会提示转人工",
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"CV  {cv_acc:.4f} | 测试 {test_acc:.4f}")
    print(f"转人工置信度阈值（折外25分位）: {conf_thr:.3f}")
    print("已保存 models/text_clf.joblib 与 models/report.json")


if __name__ == "__main__":
    main()
