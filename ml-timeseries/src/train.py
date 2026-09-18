# 训练入口: python -m src.train —— 按年滚动评估 + 保存最终模型
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.forecaster import SalesForecaster


def main():
    cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
    s = pd.read_csv(cfg["data_path"], parse_dates=["date"], index_col="date")["sales"]
    v, idx = s.values, s.index
    hs = cfg["horizons"]
    kw = cfg["model"]

    # walk-forward：每年 1 月和 7 月各设原点，测 6 步
    origins = [d for d in idx if d.year in cfg["eval_years"] and d.month in (1, 7)]
    errs = {h: [] for h in hs}
    for o in origins:
        oi = idx.get_loc(o)
        f = SalesForecaster(horizons=hs, **kw).fit(s.iloc[:oi])
        preds = f.predict(s.iloc[:oi])
        for h in hs:
            if h in preds:
                errs[h].append(preds[h] - v[oi + h - 1])

    rmse = {h: round(float(np.sqrt(np.mean(np.square(e)))), 2) for h, e in errs.items()}
    naive = []
    for o in origins:
        oi = idx.get_loc(o)
        naive += [v[oi-1] - v[oi + h - 1] for h in hs]
    report = {
        "origins": len(origins),
        "rmse_by_horizon": rmse,
        "overall_rmse": round(float(np.sqrt(np.mean(np.square(
            np.concatenate([np.array(errs[h]) for h in hs]))))), 2),
        "naive_overall_rmse": round(float(np.sqrt(np.mean(np.square(naive)))), 2),
        "config": cfg,
    }

    final = SalesForecaster(horizons=hs, **kw).fit(s)
    out = Path(cfg["artifacts_dir"]); out.mkdir(exist_ok=True)
    joblib.dump(final, out / "forecaster.joblib")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"原点 {report['origins']} 个 | 总RMSE={report['overall_rmse']} "
          f"vs Naive={report['naive_overall_rmse']} 万元")
    print("分步数 RMSE:", rmse)
    print(f"工件: {out}/forecaster.joblib, {out}/report.json")


if __name__ == "__main__":
    main()
