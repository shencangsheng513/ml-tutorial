# 时序预测器：直接式多步（每个步数一个模型，目标=相对锚点的累计变化）
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATS = ["lag1", "lag2", "lag3", "lag6", "lag12", "roll3", "roll12", "month", "t"]


def _feat_row(v, dates, i):
    w = v[i-11:i+1]
    return {"lag1": w[-1], "lag2": w[-2], "lag3": w[-3], "lag6": w[-6],
            "lag12": w[0], "roll3": w[-3:].mean(), "roll12": w.mean(),
            "month": dates[i].month % 12 + 1, "t": i + 1}


class SalesForecaster:
    def __init__(self, horizons=(1, 2, 3, 4, 5, 6), **model_kw):
        self.horizons = list(horizons)
        self.model_kw = model_kw

    def fit(self, series: pd.Series):
        v, idx = series.values, series.index
        anchors = list(range(11, len(v)))
        X = pd.DataFrame([_feat_row(v, idx, i) for i in anchors], index=anchors)
        self.models_ = {}
        for h in self.horizons:
            tgt = {i: v[i+h] - v[i] for i in anchors if i + h < len(v)}
            if len(tgt) < 24:
                continue
            m = RandomForestRegressor(**self.model_kw).fit(X.loc[sorted(tgt)], pd.Series(tgt))
            self.models_[h] = m
        return self

    def predict(self, history: pd.Series):
        """history: 截至最近已知月。返回 {h: 预测值}"""
        v, idx = history.values, history.index
        row = pd.DataFrame([_feat_row(v, idx, len(v) - 1)])
        base = v[-1]
        return {h: base + float(m.predict(row)[0]) for h, m in self.models_.items()}
