# 进阶3：分类任务 —— 逻辑回归 + ROC 曲线
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
features = [c for c in train_df.columns if c != "MedHouseVal"]
X_train, X_test = train_df[features], test_df[features]

# ---------------------------------------------------------------
# 1) 造分类目标：房价 > 3.0（即 $30万）记为"豪宅"=1
#    第一件事永远是：看类别比例！
# ---------------------------------------------------------------
y_train = (train_df["MedHouseVal"] > 3.0).astype(int)
y_test = (test_df["MedHouseVal"] > 3.0).astype(int)
print(f"1) 类别分布: 豪宅 {y_train.mean()*100:.1f}%  普通 {100-y_train.mean()*100:.1f}%")
print("   ⚠️ 如果无脑全预测'普通'，准确率就有 77%！所以准确率是骗人指标")

# ---------------------------------------------------------------
# 2) 两个模型
#    逻辑回归：必须标准化（它像找一条带权重的分界线，量纲不齐会歪）
#    随机森林：不需要标准化（它是切分数据，不在乎尺度）
#    Pipeline 把"标准化+模型"捆在一起，交叉验证时不会泄漏
# ---------------------------------------------------------------
lr = Pipeline([("scaler", StandardScaler()),
               ("clf", LogisticRegression(max_iter=1000))])
rf = RandomForestClassifier(n_estimators=200, max_depth=16, random_state=42)

print("\n2) 5折交叉验证的 ROC-AUC:")
for name, m in [("逻辑回归", lr), ("随机森林", rf)]:
    s = cross_val_score(m, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"   {name:<6} AUC = {s.mean():.4f} ± {s.std():.4f}")

# ---------------------------------------------------------------
# 3) 详细成绩单：选阈值 0.5 时的混淆矩阵 + precision/recall
# ---------------------------------------------------------------
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
print("\n3) 随机森林在测试集上的报告 (阈值=0.5):")
print(classification_report(y_test, pred, target_names=["普通", "豪宅"], digits=3))
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print(f"   混淆矩阵: 真阴性={tn} 假警报={fp} 漏报={fn} 命中={tp}")
print(f"   精确率=命中/(命中+假警报)={tp/(tp+fp):.3f} ← 预测'豪宅'里多少是真的")
print(f"   召回率=命中/(命中+漏报)={tp/(tp+fn):.3f} ← 真豪宅里抓住了多少")

# ---------------------------------------------------------------
# 4) ROC 曲线：不指定阈值，扫描所有阈值的 真正例率 vs 假正例率
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for name, m, color in [("LogisticRegression", lr, "tab:blue"),
                       ("RandomForest", rf, "tab:orange")]:
    fpr, tpr, _ = roc_curve(y_test, m.predict_proba(X_test)[:, 1])
    axes[0].plot(fpr, tpr, color=color, lw=2,
                 label=f"{name} AUC={auc(fpr, tpr):.3f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="random guess")
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

# 逻辑回归系数 = 每个特征把"变豪宅"的对数几率推高/压低多少
coefs = pd.Series(lr.named_steps["clf"].coef_[0], index=features).sort_values()
axes[1].barh(coefs.index, coefs.values, color=np.where(coefs > 0, "g", "r"))
axes[1].set_title("LogisticRegression coefficients (after scaling)")
axes[1].axvline(0, color="k", lw=0.8); axes[1].grid(alpha=0.3)

fig.tight_layout(); fig.savefig("classification_roc.png", dpi=110)
print("\n4) 图已保存 classification_roc.png")
print("   系数图: 绿=推向豪宅 红=推向普通; 注意 LogIncome 一柱擎天")
