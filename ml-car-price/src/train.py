# 训练入口: python -m src.train
# 一份代码同时是"实验"和"产线"：CV → 终训 → 测试集评估 → 保存工件
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

from src.pipeline import build_pipeline, load_raw


def main():
    cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
    X, y = load_raw(cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"])

    pipe = build_pipeline(cfg)
    cv = cross_val_score(pipe, X_train, y_train, cv=cfg["cv_folds"],
                         scoring="neg_root_mean_squared_error", n_jobs=1)
    pipe.fit(X_train, y_train)
    rmse = np.sqrt(mean_squared_error(y_test, pipe.predict(X_test)))

    report = {
        "cv_rmse_mean": round(-cv.mean(), 4),
        "cv_rmse_std": round(cv.std(), 4),
        "test_rmse": round(rmse, 4),
        "config": cfg,
    }
    out = Path(cfg["artifacts_dir"]); out.mkdir(exist_ok=True)
    joblib.dump(pipe, out / "model.pipeline.joblib")
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"CV-RMSE  = {-cv.mean():.4f} ± {cv.std():.4f}")
    print(f"Test-RMSE= {rmse:.4f}")
    print(f"工件已保存: {out}/model.pipeline.joblib, {out}/report.json")


if __name__ == "__main__":
    main()
