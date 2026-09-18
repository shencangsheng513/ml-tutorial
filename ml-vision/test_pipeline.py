# -*- coding: utf-8 -*-
"""单测：canvas_to_input 管线过一遍全部450张测试图，验证'画笔→模型'链路信息不丢"""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from src.preprocess import canvas_to_input

d = load_digits()
X, y = d.data / 16.0, d.target
_, Xte, _, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

class SmallCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Flatten(), torch.nn.Linear(16 * 2 * 2, 10))
    def forward(self, x):
        return self.net(x)

m = SmallCNN()
m.load_state_dict(torch.load("models/cnn_digits.pt", map_location="cpu", weights_only=True))
m.eval()

preds = []
for img01, lab in zip(Xte.reshape(-1, 8, 8), yte):
    # 模拟用户在 320px 画布上"画"这个字：黑笔白底，先放大到 40×40 再随机偏移放置
    big = np.zeros((320, 320), dtype=np.uint8) + 255
    stroke = (img01 * 255).astype(np.uint8)
    up = np.asarray(Image.fromarray(255 - stroke).resize((160, 160), Image.LANCZOS))
    big[80:240, 80:240] = up                        # 白底黑字放中间
    x = canvas_to_input(Image.fromarray(big).convert("RGBA"))
    with torch.no_grad():
        preds.append(int(m(torch.tensor(x).unsqueeze(0).unsqueeze(0)).argmax(1)[0]))
preds = np.array(preds)
print(f"经'画布管线'绕行一圈后的准确率: {accuracy_score(yte, preds):.4f}（直接喂=0.9800）")
cm = confusion_matrix(yte, preds)
bad = [(i, j, cm[i, j]) for i in range(10) for j in range(10) if i != j and cm[i, j] > 0]
print("错到哪儿:", sorted(bad, key=lambda t: -t[2])[:5])
