# 特征工程封装为 sklearn Transformer：fit 只接触训练数据，泄漏在结构上不可能
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RAW_COLS = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]
OUT_COLS = RAW_COLS + ["BedroomRatio", "PeoplePerRoom", "LogIncome",
                       "DistCoast", "DistSF", "DistLA"]

# 加州海岸线近似：纬度 -> 海岸经度（线性插值），把"是否沿海"变成一维数字
_COAST_LAT = [32.53, 33.70, 34.45, 36.20, 37.80, 39.00, 40.50, 41.95]
_COAST_LON = [-117.13, -118.40, -120.47, -121.80, -123.00, -123.70, -124.10, -124.35]
_SF = (37.78, -122.42)
_LA = (34.05, -118.24)


def _km_per_deg_lon(lat):
    return 111.0 * np.cos(np.radians(lat))


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, clip_quantiles=(0.01, 0.99)):
        self.clip_quantiles = clip_quantiles

    def fit(self, X, y=None):
        # 裁剪边界只从"喂进来的这份数据"学习 —— 放进 Pipeline 后就是每个训练折本身
        self.clip_lo_, self.clip_hi_ = X["AveOccup"].quantile(self.clip_quantiles)
        return self

    def transform(self, X):
        X = X.copy()
        # 与 step2 完全一致的顺序：先派生（用未裁剪值），再裁剪
        X["BedroomRatio"] = X["AveBedrms"] / X["AveRooms"]
        X["PeoplePerRoom"] = X["AveOccup"] / X["AveRooms"]
        X["LogIncome"] = np.log1p(X["MedInc"])
        X["AveOccup"] = X["AveOccup"].clip(self.clip_lo_, self.clip_hi_)

        lat = X["Latitude"].values
        lon = X["Longitude"].values
        km_lon = _km_per_deg_lon(lat)
        coast_lon = np.interp(lat, _COAST_LAT, _COAST_LON)
        X["DistCoast"] = np.abs(coast_lon - lon) * km_lon
        for name, (clat, clon) in {"DistSF": _SF, "DistLA": _LA}.items():
            dx = (lon - clon) * km_lon
            dy = (lat - clat) * 111.0
            X[name] = np.sqrt(dx ** 2 + dy ** 2)
        return X[OUT_COLS]
