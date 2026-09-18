# 从 config 组装完整 Pipeline：原始8列进 → 预测出，中间无人工环节
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.features import FeatureEngineer, RAW_COLS


def build_pipeline(cfg):
    fe = FeatureEngineer(clip_quantiles=tuple(cfg["clip_quantiles"]))
    model = XGBRegressor(**cfg["model"])
    return Pipeline([("features", fe), ("model", model)])


def load_raw(cfg):
    import pandas as pd
    df = pd.read_csv(cfg["data_path"])
    return df[RAW_COLS], df[cfg["target"]]
