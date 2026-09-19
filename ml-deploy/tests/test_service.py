def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_ready(client):
    assert client.get("/ready").json()["status"] == "ready"


def test_classify_happy_path(client):
    body = {"texts": ["The pitcher threw a fastball in the inning",
                      "Launching a rocket into orbit around mars"]}
    r = client.post("/v1/classify", json=body)
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert preds[0]["label"] == "baseball"
    assert preds[1]["label"] == "space"
    assert preds[0]["confidence"] > 0.3


def test_rejects_empty_batch(client):
    assert client.post("/v1/classify", json={"texts": []}).status_code == 422


def test_rejects_oversized_batch(client):
    body = {"texts": ["some text here"] * 65}
    assert client.post("/v1/classify", json=body).status_code == 422


def test_length_guardrail(client):
    r = client.post("/v1/classify", json={"texts": ["hi"]})
    p = r.json()["predictions"][0]
    assert p["label"] == "rejected" and p["note"]


def test_deterministic_predictions(client):
    body = {"texts": ["gpu shader rendering", "gpu shader rendering"]}
    a = client.post("/v1/classify", json=body).json()["predictions"]
    assert a[0] == a[1]
