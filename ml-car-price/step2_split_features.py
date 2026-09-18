# 第2课：数据划分 + 特征工程
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/california_housing.csv")

# ---------------------------------------------------------------
# 原则1：先划分，再处理
# 测试集模拟"将来才见到的数据"，必须在处理前锁进保险箱
# ---------------------------------------------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print("训练集:", train_df.shape, " 测试集:", test_df.shape)

# ---------------------------------------------------------------
# 原则2：特征工程来自领域直觉，而不是算法
# 从"租客看房"的角度构造新特征：
# ---------------------------------------------------------------
def add_features(d):
    d = d.copy()
    # 卧室占比：卧室占房间总数比例过高 => 可能是群租房，价值低
    d["BedroomRatio"] = d["AveBedrms"] / d["AveRooms"]
    # 每套房人数（比 AveOccup 更稳，AveOccup 有极端离群值）
    d["PeoplePerRoom"] = d["AveOccup"] / d["AveRooms"]
    # 收入对数：收入分布右偏，取对数后更接近线性关系
    d["LogIncome"] = np.log1p(d["MedInc"])
    return d

train_df = add_features(train_df)
test_df = add_features(test_df)

# ---------------------------------------------------------------
# 原则3：处理只在训练集上"学习"，再应用到测试集
# 先看离群值有多夸张：
# ---------------------------------------------------------------
print("\nAveOccup 分位数（训练集）:")
print(train_df["AveOccup"].quantile([0.5, 0.95, 0.99, 1.0]).round(1))

# 用训练集的分位数裁剪极端值（clip），测试集用同样的界限
lo, hi = train_df["AveOccup"].quantile([0.01, 0.99])
for d in (train_df, test_df):
    d["AveOccup"] = d["AveOccup"].clip(lo, hi)
print(f"\n裁剪后 AveOccup 范围: [{lo:.1f}, {hi:.1f}]（上下界只由训练集决定）")

# 整理成 sklearn 需要的 X / y 格式
features = [c for c in train_df.columns if c != "MedHouseVal"]
X_train, y_train = train_df[features], train_df["MedHouseVal"]
X_test, y_test = test_df[features], test_df["MedHouseVal"]

# 保存给下一课用
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
print("\n特征列:", features)
print("已保存 data/train.csv 和 data/test.csv")
