# 机器学习实操课程作品集

> 从 0 到上线：4 个完整项目 + 5 个工程实验，全部亲手实现并诚实复盘。
> 环境：Python / scikit-learn / XGBoost / PyTorch / Streamlit / MLflow / FastAPI / DashScope（CPU 可跑）

## 项目全景

| 项目 | 数据 | 成绩 | 最值得看的一课 |
|------|------|------|----------------|
| [ml-car-price](ml-car-price/) 房价回归 | California Housing 20640×9 | RMSE 0.4229 | 域知识特征 > 调参；置换重要性纠错 |
| [ml-timeseries](ml-timeseries/) 销量预测 | 合成120月+真实CO2 | RMSE 7.09 vs 朴素11.67 | 树模型不会外推→差分目标；滚动验证 |
| [ml-text](ml-text/) 新闻组分类 | 20newsgroups 4类 | 87.0%（BERT持平） | 元数据泄漏实测；词袋死穴的边界 |
| [ml-vision](ml-vision/) 数字识别 | digits + MNIST | CNN 98.0% / MNIST 正名 | 数据增强证伪；架构要匹配挑战 |
| [kaggle-sim](kaggle-sim/) 比赛机制 | Titanic 887行 | 公榜0.972 私榜0.826 | submit登山=背答案实锤 |
| [mlflow-lab](mlflow-lab/) 实验跟踪 | 8 runs 入库 | — | search_runs 一行重排历史 |
| [ml-deploy](ml-deploy/) 模型服务 | ml-text 的 87.0% 分类器 | pytest 14/14 · 延迟~17ms | 护栏/日志/healthz+ready 前置到服务层 |
| [ml-rag](ml-rag/) 知识库问答 | 仓库 9 篇 md → 76 块（生产索引随语料重建） | v7.1 端到端 RAG 8.19 vs 裸考 4.19（p=7.5e-6） | v8~v12 五轮消融：四条修法全部实测封死，负结果才是产出 |
| [ml-compress](ml-compress/) 模型压缩 | ml-vision 的 MNIST CNN（106k 参数）· 两课 | int8 静态 3.7x 体积、精度 0.9798→0.9801；第2课查明"默认 qconfig 即逐通道"，4x 变 3.7x 的 9.1% 差额是元数据+序列化 | 收益上限可以不碰后端先算出来；体积/精度跨运行字节级可复现，墙钟延迟跨运行漂 4.3 倍——两类指标不能同信 |

## 跨项目方法论（踩出来的）

1. **泄漏是头号公敌**：时间泄漏/元数据泄漏/评估泄漏，每个项目用对照实验量化一次
2. **指标反常 = 流程警报**，先审计后庆祝（时序课因此发现多步预测标签泄漏）
3. **误差分析决定投资方向**：分片、混淆对、置信度分层收益 > 换模型
4. **train/serve 一致性**：一套预处理从训练用到线上（增强不对齐分布时是毒药，有实验）
5. **上线三件套**：置信度阈值 + 输入护栏 + 拒答转人工
6. **评估工具也是被测对象**：RAG 首轮输给裸考，查出的问题是提示词和判卷官自己；judge 全给满分是量表饱和，不是完美

## 快速开始

```bash
pip install -r requirements.txt
# 每个项目目录内按 README 运行；App 均为 Streamlit，各带 launch_*.bat
```

- 3 个 Streamlit 应用端口：8502（时序）/ 8503（文本）/ 8504（手绘数字）
- MLflow UI：`python -m mlflow ui --backend-store-uri sqlite:///mlflow-lab/mlflow.db --port 5000`

## 复盘与简历

完整反思见 [项目复盘与简历写法.md](项目复盘与简历写法.md)（含 STAR 条目）。
