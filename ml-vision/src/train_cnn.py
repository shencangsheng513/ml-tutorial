# -*- coding: utf-8 -*-
"""
CV第4课·上半场：把第3课的小CNN正式"训练+存档"，产出给手绘识别App用。
工程化三件事：
  ① 训练逻辑与实验脚本分离（src/ 目录，可重复运行）
  ② 模型存盘 + 口径文档：App 预处理必须和这里一模一样（0-1灰度、8×8、居中）
  ③ 存一份报告 json：测试指标写进档案，上线后能对账
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent
torch.manual_seed(42)
rng = np.random.default_rng(42)

d = load_digits()
X, y = d.data / 16.0, d.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=42, stratify=y)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 2 * 2, 10))

    def forward(self, x):
        return self.net(x)


model = SmallCNN()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
lossf = nn.CrossEntropyLoss()
xt = torch.tensor(Xtr.reshape(-1, 8, 8), dtype=torch.float32).unsqueeze(1)
yt = torch.tensor(ytr, dtype=torch.long)
B = 64
for ep in range(120):
    model.train()
    perm = torch.randperm(len(xt))
    for i in range(0, len(xt), B):
        idx = perm[i:i + B]
        opt.zero_grad()
        loss = lossf(model(xt[idx]), yt[idx])
        loss.backward()
        opt.step()

model.eval()
xe = torch.tensor(Xte.reshape(-1, 8, 8), dtype=torch.float32).unsqueeze(1)
with torch.no_grad():
    pred = model(xe).argmax(1).numpy()
acc = accuracy_score(yte, pred)

Path(ROOT / "models").mkdir(exist_ok=True)
torch.save(model.state_dict(), ROOT / "models" / "cnn_digits.pt")
(ROOT / "models" / "train_report.json").write_text(json.dumps({
    "model": "SmallCNN 2conv+fc", "epochs": 120, "test_acc": round(float(acc), 4),
    "preprocess": "灰度→白字黑底→裁包围盒→正方形padding→缩到8x8→除以255→[0,1]",
    "input_shape": "(1,1,8,8) float32",
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"训练完成：测试准确率 {acc:.4f}")
print("模型已存：models/cnn_digits.pt")
