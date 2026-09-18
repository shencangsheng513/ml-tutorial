# 进阶2：Bagging vs Boosting —— 随机森林 vs XGBoost
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
features = [c for c in train_df.columns if c != "MedHouseVal"]
X_train, y_train = train_df[features], train_df["MedHouseVal"]
X_test, y_test = test_df[features], test_df["MedHouseVal"]

def cv_rmse(model):
    s = cross_val_score(model, X_train, y_train, cv=5,
                        scoring="neg_root_mean_squared_error", n_jobs=-1)
    return -s.mean()

# ---------------------------------------------------------------
# 1) 随机森林（上一课的冠军，已调优）
# ---------------------------------------------------------------
rf = RandomForestRegressor(n_estimators=200, max_depth=24, random_state=42, n_jobs=-1)
print(f"随机森林   CV-RMSE = {cv_rmse(rf):.4f}")

# ---------------------------------------------------------------
# 2) XGBoost：boosting。关键参数是 n_estimators(树棵数) 和 learning_rate(每步走多远)
#    经验：learning_rate 小 → 需要更多树，但泛化更好
# ---------------------------------------------------------------
xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                   subsample=0.8, colsample_bytree=0.8,
                   random_state=42, n_jobs=-1)
print(f"XGBoost    CV-RMSE = {cv_rmse(xgb):.4f}")

# ---------------------------------------------------------------
# 3) 最终在测试集上对比（只跑一次，防作弊）
# ---------------------------------------------------------------
print("\n训练后在测试集(保险箱数据)上的最终成绩:")
rf.fit(X_train, y_train); xgb.fit(X_train, y_train)
for name, m in [("随机森林", rf), ("XGBoost", xgb)]:
    p = m.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, p))
    mae = mean_absolute_error(y_test, p)
    print(f"  {name:<8} 测试RMSE={rmse:.4f}  测试MAE={mae:.4f}")

# 保存更好者
p_rf = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
p_xgb = np.sqrt(mean_squared_error(y_test, xgb.predict(X_test)))
winner, wname = (rf, "随机森林") if p_rf <= p_xgb else (xgb, "XGBoost")
joblib.dump({"model": winner, "features": features}, "model_best.joblib")
print(f"\n>>> 冠军：{wname}，已保存 model_best.joblib（进阶4 上线用）")
