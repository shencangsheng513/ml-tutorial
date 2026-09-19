"""鉴权中间件与 deploy-serve 入口的离线测试（微型模型替身，不需要真 token/Docker）。"""

import pytest
from fastapi.testclient import TestClient

from deploy.config import get_settings

FAKE_TOKEN = "pytest-not-a-real-token"


@pytest.fixture
def auth_client(tiny_model_path, monkeypatch):
    monkeypatch.setenv("DEPLOY_API_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("DEPLOY_MODEL_PATH", str(tiny_model_path))
    get_settings.cache_clear()
    from deploy.service import app
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


BODY = {"texts": ["The pitcher threw a fastball in the inning"]}


def test_missing_header_rejected(auth_client):
    assert auth_client.post("/v1/classify", json=BODY).status_code == 401


def test_wrong_token_rejected(auth_client):
    h = {"Authorization": "Bearer " + "x" * len(FAKE_TOKEN)}
    assert auth_client.post("/v1/classify", json=BODY, headers=h).status_code == 401


def test_valid_bearer_passes(auth_client):
    h = {"Authorization": f"Bearer {FAKE_TOKEN}"}
    r = auth_client.post("/v1/classify", json=BODY, headers=h)
    assert r.status_code == 200 and r.json()["predictions"][0]["label"] == "baseball"


def test_health_endpoints_stay_open(auth_client):
    assert auth_client.get("/healthz").status_code == 200
    assert auth_client.get("/ready").status_code == 200


def test_auth_off_when_token_unset(client):
    # 默认（conftest 未设 token）：不鉴权也能用，向后兼容本地开发
    assert client.post("/v1/classify", json=BODY).status_code == 200


def test_serve_reads_port_workers_from_env(monkeypatch):
    from deploy import server
    monkeypatch.setenv("DEPLOY_PORT", "7860")
    monkeypatch.setenv("DEPLOY_WORKERS", "2")
    get_settings.cache_clear()
    calls = {}
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **k: calls.update(k))
    server.main([])
    assert calls["port"] == 7860 and calls["workers"] == 2
    get_settings.cache_clear()


def test_serve_cli_overrides_env(monkeypatch):
    from deploy import server
    monkeypatch.setenv("DEPLOY_WORKERS", "2")
    get_settings.cache_clear()
    calls = {}
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **k: calls.update(k))
    server.main(["--port", "9000", "--workers", "4"])
    assert calls["port"] == 9000 and calls["workers"] == 4
    get_settings.cache_clear()
