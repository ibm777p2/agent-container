"""CORS on the AG-UI server: localhost (any port) and the configured web origins only."""

import pytest
from fastapi.testclient import TestClient

from property_intelligence_rag import server


@pytest.fixture
def client(monkeypatch):
    # `require_token` reads the module global at request time, so this works however the
    # module was first imported (other test files import it without a token set).
    monkeypatch.setattr(server, "AGENT_TOKEN", "test-token")
    return TestClient(server.app)


def _preflight(client, origin):
    return client.options(
        "/agent",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:8765",
        "http://127.0.0.1:3000",
        "http://localhost",
    ],
)
def test_preflight_allows_known_origins(client, origin):
    resp = _preflight(client, origin)
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == origin
    assert "POST" in resp.headers["access-control-allow-methods"]
    allowed = resp.headers["access-control-allow-headers"].lower()
    assert "authorization" in allowed and "content-type" in allowed


@pytest.mark.parametrize("origin", ["https://evil.example", "http://localhost.evil.example:8765"])
def test_preflight_rejects_unknown_origins(client, origin):
    resp = _preflight(client, origin)
    assert "access-control-allow-origin" not in resp.headers


def test_health_carries_cors_header(client):
    resp = client.get("/agent/health", headers={"Origin": "http://localhost:5173"})
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_unauthorized_still_carries_cors_header(client):
    resp = client.post("/agent", json={}, headers={"Origin": "http://localhost:5173"})
    assert resp.status_code == 401
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"
