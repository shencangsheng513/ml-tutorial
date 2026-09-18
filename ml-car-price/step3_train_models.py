# 第3课：训练三个模型 + 对比评估
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

features = [c for c in train_df.columns if c != "MedHouseVal"]
X_train, y_train = train_df[features], train_df["MedHouseVal"]
X_test, y_test = test_df[features], test_df["MedHouseVal"]

def evaluate(name, model):
    # 训练集误差 vs 测试集误差 —— 两者的差距就是"过拟合程度"
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    rmse_tr = np.sqrt(mean_squared_error(y_train, train_pred))
    rmse_te = np.sqrt(mean_squared_error(y_test, test_pred))
    mae_te = mean_absolute_error(y_test, test_pred)
    print(f"{name:<14} 训练RMSE={rmse_tr:.3f}  测试RMSE={rmse_te:.3f}  "
          f"测试MAE={mae_te:.3f}  过拟合差距={rmse_te-rmse_tr:+.3f}")
    return test_pred

print("RMSE 单位 = 10万美元；0.50 即平均误差约 $50,000")
print("-" * 78)

# 1) 线性回归：假设房价 = 特征的加权和
lr = LinearRegression().fit(X_train, y_train)
evaluate("线性回归", lr)

# 2) 决策树：不限制深度（放任它死记硬背）
dt = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
evaluate("决策树(无限深)", dt)

# 3) 随机森林：100棵各自限制深度的树取平均
rf = RandomForestRegressor(n_estimators=200, max_depth=16,
                           random_state=42, n_jobs=-1).fit(X_train, y_train)
pred_rf = evaluate("随机森林", rf)

print("-" * 78)
# 换算成美元，直观感受误差
rmse = np.sqrt(mean_squared_error(y_test, pred_rf))
print(f"随机森林误差约 ±${rmse*100000:,.0f}（1990年加州房价中位数约 $14万）")

# 保存最好的模型，第5课预测实战用
joblib.dump({"model": rf, "features": features}, "model_rf.joblib")
print("最佳模型已保存: model_rf.joblib")
