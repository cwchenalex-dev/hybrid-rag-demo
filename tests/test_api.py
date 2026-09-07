from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_rejects_empty_query():
    response = client.post(
        "/chat",
        json={"query": ""},
    )

    assert response.status_code == 422


def test_chat_rejects_whitespace_only_query():
    response = client.post(
        "/chat",
        json={"query": "   "},
    )

    assert response.status_code == 422


def test_chat_trims_query(monkeypatch):
    def fake_query(value: str):
        assert value == "refund status"

        return {
            "answer": "test",
            "sources": [],
        }

    monkeypatch.setattr(
        "app.routes.query",
        fake_query,
    )

    response = client.post(
        "/chat",
        json={"query": "  refund status  "},
    )

    assert response.status_code == 200


def test_chat_rejects_query_over_max_length():
    response = client.post(
        "/chat",
        json={"query": "x" * 2001},
    )

    assert response.status_code == 422