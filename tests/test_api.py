from fastapi.testclient import TestClient

from app.main import app


def test_health():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}


def test_debug_paths():
    client = TestClient(app)
    payload = client.get("/debug/paths").json()
    assert "uploads" in payload
    assert "outputs" in payload
    assert "db" in payload
