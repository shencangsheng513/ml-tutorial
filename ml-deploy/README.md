# ml-deploy — 生产级模型推理服务

把文本分类项目（`ml-text/`，TF-IDF+LR Pipeline，测试 87.0%，口径见 `ml-text/models/report.json`）包装成一个可部署、可测试、可维护的 HTTP 服务。

## 它长什么样

```
POST /v1/classify          {"texts": ["..."]}  ->  每篇的类别+四类概率+置信度
GET  /healthz /ready       存活/就绪探针（容器编排用，永不开鉴权）
GET  /docs                 FastAPI 自动生成的 OpenAPI 文档
```

鉴权：设 `DEPLOY_API_TOKEN` 后，除探针外的全部路径要求 `Authorization: Bearer <token>`（恒定时间比对），不设则关闭——本地开发零摩擦，上云一行开关。

**confidence 口径（对接收方必读）**：同一个模型、两个入口，`confidence` 含义不同——Streamlit 应用（`ml-text/app.py`）显示的是 **Top1−Top2 差值**，`ml-text/models/report.json` 的 `confidence_threshold`（0.207）就是按这个差值定的；本服务的 `confidence` 是 **Top1 概率**。四分类下 Top1 概率恒 ≥0.25，拿 0.207 去比服务的 `confidence` 永远不会触发转人工。要在服务侧复刻"低置信转人工"，请用响应里完整的 `probabilities` 自己算差值。

## 工程化要点（相对"能跑"的 Streamlit 版升级在哪）

| 维度 | 做法 | 为什么 |
|---|---|---|
| 配置 | pydantic-settings，全部 `DEPLOY_*` 环境变量注入 | 12-Factor：镜像不变，环境变配置变 |
| 数据契约 | pydantic schema：空列表/超64篇/非字符串 → 422，**进模型前拦截** | 模型不该为脏输入背锅 |
| 护栏 | <10字符直接 `rejected`+原因 | 第3课实证：<10词帖准确率仅52%，拒答好过自信地错 |
| 加载 | lifespan 启动时 load 一次，进程内复用 | joblib load 14MB 要秒级，不能每请求来一遍 |
| 日志 | JSON 结构化 + 请求中间件（path/status/latency） | 机器可解析才能进日志系统告警 |
| 测试 | 14 个 pytest（鉴权 7 + 服务 7）：契约/护栏/确定性；CI 用微型替身模型，**不携带14MB二进制** | 模型工件走 MLflow/卷挂载，测试测的是接口 |
| 打包 | Dockerfile：模型卷挂载 + HEALTHCHECK | 与真实部署一致的 artifact 管理姿势 |
| 版本锁 | `scikit-learn==1.9.1`（原来是 `>=1.5`） | joblib 的 pickle 里带着训练版本号，运行时不一致就 `InconsistentVersionWarning`；`>=` 让镜像每次重建都可能装到更新版，等于把慢雷埋进 CI。**训练版本 = 锁版本 = 工件版本**，三者必须同源 |

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

## 上云实战（2026-09-20 已完成 ✅）
**Live：https://news-classifier-service-jsfq.onrender.com**（Render 免费档，Docker 部署，独立仓库 [news-classifier-service](https://github.com/shencangsheng513/news-classifier-service)：模型烤进镜像，自包含 Dockerfile）

1. ~~首选 HF Spaces~~ **改道记录**：2026 起 HF 对 Docker/Gradio Space 免费档收费（402，需 $9/月 PRO，公开的也不行），只有 Static 免费 → 转 Render 免费档，首建 2m01s 上线
2. 鉴权/入口按设计生效：`DEPLOY_API_TOKEN` 走 Render 环境变量注入（不落 repo），`deploy-serve` 读 `DEPLOY_PORT` 监听 7860
3. **踩坑实录——健康检查必须配 `/healthz`**：Render 默认探 `HEAD /`，被 Bearer 中间件打 401，平台误判服务不健康触发重启。修法：Settings → Health Check Path = `/healthz`。这就是 `OPEN_PATHS` 白名单存在的实战理由
4. 冒烟三查全绿：无钥匙 `/healthz` 200（model_loaded true）→ 无钥匙 `/v1/classify` 401 → 带钥匙 200 + 真实预测，公网延迟 ~183ms（**此为单样本，已被第 5 条实测推翻，别拿它当数**）
5. 免费档脾气——**2026-09-20/21 实测版**（`scripts/cloud_probe.py`，只打公开 `/healthz` 不需要 token；原始数据 `cloud_probe_results.json` / `cloud_probe_warm.json`）：
   - **冷启动：实测唤醒 33.4 秒**（空闲 >15 分钟后的第一发请求，含平台排队 + 容器起 + 模型载入；结局 200 + model_loaded true，唤醒可靠）。
   - **热服务：最便宜的 `/healthz` 端点连测 25 发，公网 p50 684ms / p95 829ms / max 1149ms**，两轮共 50+ 发全 200。
   - 结论：之前"≈ 本地 1/3–1/5、按 p99 300ms 对外承诺"是外推，**免费档兑现不了**——光是探活端点 p99 就接近 1 秒，预测接口只会更慢。对外承诺改口径：**p99 ≈ 1.2s + 首请求 ~35s 冷启动免责声明**；要低延迟只剩两条路：升付费实例，或在客户端侧做保活心跳。这是"单点数字不可信"（本地压测第 ③ 条）在生产环境的重演：183ms 那发恰好打在刚热身的窗口上。
6. **已知告警 → 本机已修完，线上待重建**：`InconsistentVersionWarning` 根因是模型用 sklearn 1.9.0 pickle、镜像按 `>=1.5` 装到了 1.9.1。修法分三刀：`pyproject.toml` 锁 `==1.9.1`；用 1.9.1 重跑 `python -m src.train` 重新生成工件（指标逐位相同：CV 0.8808 / 测试 0.8702 / 阈值 0.207，说明补丁版本没动行为）；新工件换进服务仓库 `model/`。行为一致性证据：新旧工件对全部 1579 篇测试集 **0 条预测不一致、概率最大绝对差 0.0**，加载零告警，14 个离线测试在 `-W error::UserWarning` 下全绿。**线上只剩一步：push 触发 Render 重建**（会短暂重启，需现网确认）。
7. 注意：uvicorn 多进程在 Windows 本机有 socket 继承毛病（实测偶发 WinError 10022），**多 worker 是为 Linux 容器准备的**，本机冒烟用单 worker 即可。
8. **环境漂移实录（2026-09-22）**：conda env `mt` 里**缺 `pyproject.toml` 已声明的 `pydantic-settings`** ⇒ `tests/test_auth.py` 在**收集期**就 ImportError（`Interrupted: 1 error during collection`），整套 14 个用例一个都没跑——这种失败长得像"测试坏了"，其实是环境坏了。`pip install pydantic-settings`（实得 2.15.0）后 **14 passed**。同日现读复核两条：该 env 的 `scikit-learn` 已是 **1.9.1**（与锁版一致），加载 `../ml-text/models/text_clf.joblib` 在把 warning 升级成 error 的设置下**零告警** ⇒ 第 6 条那把锁在本机侧已闭合，只剩现网重建。另：CI 与本 README 的 lint 口径只有 `src`/`tests`，**`scripts/` 不在其中**——那里三条裸 `except Exception` / `try-except-pass` 是我今天额外拿严格命令扫出来的（`cloud_probe.py`，收窄成 `OSError` + 一个"解析不出 JSON 就返回 None"的小 helper，行为不变：拿不到 `model_loaded` 时保留上一发），**不是 CI 红，别记成修了个失败**。

## CI
`.github/workflows/deploy-ci.yml`：push 触及 `ml-deploy/**` 时自动装依赖 → ruff → pytest。

## Docker
> 状态如实说：本机 Docker Desktop 引擎无头起不来（无登录会话时进程自己退了），**下面的 build/run 未在本机验证**；你回来双击打开引擎后照抄即可，"验收四查"就是给那一刻准备的。`deploy-serve` 入口与鉴权逻辑本身已在 Windows 本机起服务冒烟通过（healthz 200 / 无token 401 / 带token 200）。同款 Dockerfile 配方已在 Render 云端构建验证通过（见上面"上云实战"，模型改为烤进镜像），本机的卷挂载版仍未本地实测。
>
> 2026-09-20 用锁版后的新工件重做本机回环冒烟（单 worker，`127.0.0.1:8123`）：`/healthz` 200 且 `model_loaded: true`；无 token 打 `/v1/classify` → 401；带 token 三条真实帖子 → 200（hockey 0.9234 等），单请求 23.6ms；**进程日志零 warning**（旧工件在此必报 `InconsistentVersionWarning`），14 个离线测试 `-W error::UserWarning` 全绿。Docker 引擎当天仍没起来，下面四查照旧待你双击开引擎后执行。

```bash
docker build -t ml-deploy ml-deploy/
docker run -d --name ml-deploy -v <绝对路径>/ml-tutorial/ml-text/models:/models:ro \
  -e DEPLOY_API_TOKEN=<自己现编一个> -e DEPLOY_WORKERS=2 -p 8000:8000 ml-deploy
# 验收四查：
curl http://localhost:8000/healthz                                  # 200
curl -X POST localhost:8000/v1/classify -d '{"texts":["x y z 1 2 3"]}' -H 'Content-Type: application/json'          # 401
curl -X POST localhost:8000/v1/classify -d '{"texts":["The goalie made a glove save in overtime"]}' \
  -H 'Content-Type: application/json' -H "Authorization: Bearer <自己现编一个>"                                      # 200 hockey
docker logs ml-deploy   # 应看到 2 个 worker 的 JSON 访问日志（含上面那条 401）
```
