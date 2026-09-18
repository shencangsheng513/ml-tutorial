# 第1课：数据探索 (EDA)
import pandas as pd
import numpy as np

df = pd.read_csv("data/california_housing.csv")

# 1) 数据长什么样：行数、列数
print("=" * 50)
print("1. 数据形状 (行, 列):", df.shape)

# 2) 每一列的类型和缺失情况
print("=" * 50)
print("2. 列类型与非空数量:")
print(df.info())

# 3) 数值列的统计概览
print("=" * 50)
print("3. 统计概览:")
print(df.describe().round(2))

# 4) 相关性：哪些特征和目标 MedHouseVal 关系最强？
print("=" * 50)
print("4. 各特征与房价(MedHouseVal)的相关系数:")
corr = df.corr(numeric_only=True)["MedHouseVal"].drop("MedHouseVal")
print(corr.sort_values(ascending=False).round(3))
