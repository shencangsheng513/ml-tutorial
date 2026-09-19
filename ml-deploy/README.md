# ml-deploy — 生产级模型推理服务

把文本分类项目（`ml-text/`，TF-IDF+LR Pipeline，测试 87.7%）包装成一个可部署、可测试、可维护的 HTTP 服务。

## 它长什么样

```
POST /v1/classify          {"texts": ["..."]}  ->  每篇的类别+四类概率+置信度
GET  /healthz /ready       存活/就绪探针（容器编排用）
GET  /docs                 FastAPI 自动生成的 OpenAPI 文档
```

## 工程化要点（相对"能跑"的 Streamlit 版升级在哪）

| 维度 | 做法 | 为什么 |
|---|---|---|
| 配置 | pydantic-settings，全部 `DEPLOY_*` 环境变量注入 | 12-Factor：镜像不变，环境变配置变 |
| 数据契约 | pydantic schema：空列表/超64篇/非字符串 → 422，**进模型前拦截** | 模型不该为脏输入背锅 |
| 护栏 | <10字符直接 `rejected`+原因 | 第3课实证：<10词帖准确率仅52%，拒答好过自信地错 |
| 加载 | lifespan 启动时 load 一次，进程内复用 | joblib load 14MB 要秒级，不能每请求来一遍 |
| 日志 | JSON 结构化 + 请求中间件（path/status/latency） | 机器可解析才能进日志系统告警 |
| 测试 | 7 个 pytest：契约/护栏/确定性；CI 用微型替身模型，**不携带14MB二进制** | 模型工件走 MLflow/卷挂载，测试测的是接口 |
| 打包 | Dockerfile：模型卷挂载 + HEALTHCHECK | 与真实部署一致的 artifact 管理姿势 |

## 本地跑

```bash
# 依赖装在项目自带 venv（不动 conda 全局）
D:\ML_Projects\ml-tutorial\.venv\Scripts\python.exe -m pip install -e "ml-deploy[dev]"
# 测试 + lint
python -m pytest ml-deploy && python -m ruff check ml-deploy/src ml-deploy/tests
# 起服务（真模型）
ml-deploy\run_server.bat   # 或 uvicorn deploy.service:app --app-dir ml-deploy/src
curl http://127.0.0.1:8000/v1/classify -d "{\"texts\":[\"The goalie made a glove save in overtime\"]}" -H "Content-Type: application/json"
```

## 冒烟实测（真模型）
三帖批测：GPU 帖→graphics 0.71 / 冰球帖→hockey 0.77 / 望远镜帖→space 0.55，端到端 17ms。

## 压测（`scripts/loadtest.py`，本地回环，单 uvicorn worker，全零错误）

| 并发×批大小 | 吞吐 | p50 | p99 |
|---|---|---|---|
| 1×1 | ~98 条/秒 | 5.7ms | 28.9ms |
| 8×1 | 510–680 条/秒（两轮波动） | 10.7ms | 45.2ms |
| 32×1 | ~565 条/秒 | 51.8ms | 93.7ms |
| 8×16（批请求） | **~1741 条/秒** | 70.2ms | 101.3ms |

三个结论：① 单 worker 在并发 8 附近就摸到吞吐天花板（GIL + CPU 推理），32 并发只换来排队（p50 从 10ms 涨到 52ms，RPS 不涨）——**要容量就加 worker/副本，不是加并发**；② 批处理是最便宜的扩容：客户端攒 16 条一发，同并发下吞吐比单发再 ×3（565→1741 条/秒），代价只是 p99 从 94ms 涨到 101ms；③ 两轮 8 并发差 30%（509 vs 683），本机有后台任务干扰——**单点数字不可信，要看趋势和拐点**。原始数据：`loadtest_results.json`。

## 上云清单（免费层，待账号）
1. **首选 Hugging Face Spaces（Docker SDK）**：14MB 模型可直接烤进 repo（与 CI 用替身模型的原则不冲突——那是测接口，这是发服务）；免费 CPU 0+1，健康检查映射到 `/healthz`。
2. 备选 Render/Railway 免费档：需把模型上传到卷或对象存储，Dockerfile 的 volume 挂载方案正好复用。
3. 上云后要改的三件事：`uvicorn --workers 2`（或平台侧多副本）、加 `Authorization` 中间件（现在裸奔，仅限本机假设）、把结构化日志接到平台日志流。
4. 预期：免费 CPU 档性能 ≈ 本地压测的 1/3–1/5，p99 SLO 按 300ms 定，别拿本机数字对外承诺。

## CI
`.github/workflows/deploy-ci.yml`：push 触及 `ml-deploy/**` 时自动装依赖 → ruff → pytest。

## Docker
```bash
docker build -t ml-deploy ml-deploy/
docker run -v <绝对路径>/ml-tutorial/ml-text/models:/models:ro -p 8000:8000 ml-deploy
```
