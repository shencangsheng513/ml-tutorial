# 运行时不打包模型（.joblib 在 .gitignore）：挂载卷注入，与真实 MLOps 一致
FROM python:3.12-slim

WORKDIR /srv/app
COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir .

ENV DEPLOY_MODEL_PATH=/models/text_clf.joblib
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

# 挂载: docker run -v <绝对路径>/ml-text/models:/models:ro -p 8000:8000 ml-deploy
CMD ["uvicorn", "deploy.service:app", "--host", "0.0.0.0", "--port", "8000"]
