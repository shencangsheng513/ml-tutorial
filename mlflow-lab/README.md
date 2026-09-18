# mlflow-lab · 实验跟踪

把散落在各课的"跑完就忘"训练统一入账：参数/指标/产物三件套。

- 2 个实验 8 条 run（文本 3 + 图像 5，含 CNN state_dict 产物）
- `python track_projects.py` 重跑入账；`mlflow ui --backend-store-uri sqlite:///mlflow-lab/mlflow.db --port 5000` 查账
- 一行 `mlflow.search_runs()` 重排全部历史：SVM-rbf 0.991 居首
- 踩坑记录：MLflow 3 弃用文件后端（需迁移 sqlite）；Windows 无 job backend（UI 部分功能降级）
