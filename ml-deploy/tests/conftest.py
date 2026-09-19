"""CI 不携带 14MB 模型文件（*.joblib 在 .gitignore），用微型管线造同接口替身。"""

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

LABELS = ["graphics", "baseball", "hockey", "space"]
TINY_DOCS = [
    ("rendering shader gpu rasterization pixels", 0), ("opengl texture mesh vertex", 0),
    ("pitcher batting home run inning", 1), ("team season game baseball", 1),
    ("puck goalie skating hockey rink", 2), ("power play slap shot hockey", 2),
    ("rocket orbit telescope satellite", 3), ("nasa mars spacecraft launch", 3),
]


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory):
    X, y = [d for d, _ in TINY_DOCS], [l for _, l in TINY_DOCS]
    pipe = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=200))
    pipe.fit(X, y)
    p = tmp_path_factory.mktemp("model") / "tiny.joblib"
    joblib.dump(pipe, p)
    return p


@pytest.fixture(scope="session")
def client(tiny_model_path):
    import os
    os.environ["DEPLOY_MODEL_PATH"] = str(tiny_model_path)
    from deploy.config import get_settings
    get_settings.cache_clear()
    from fastapi.testclient import TestClient

    from deploy.service import app
    with TestClient(app) as c:  # with 语句触发 lifespan：模型真实加载一次
        yield c
