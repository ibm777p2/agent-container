from fastapi.testclient import TestClient

from property_intelligence_rag import server


def test_health():
    response = TestClient(server.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_direct_property_endpoint(monkeypatch):
    monkeypatch.setattr(
        server,
        "get_property_data",
        lambda address, refresh=False: {"address": address, "refresh": refresh},
    )
    response = TestClient(server.app).post(
        "/tools/property", json={"address": "123 Main St", "refresh": True}
    )
    assert response.status_code == 200
    assert response.json() == {"address": "123 Main St", "refresh": True}
