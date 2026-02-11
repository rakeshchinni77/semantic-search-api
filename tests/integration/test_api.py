import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    """
    Ensure /health returns 200 and correct body.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_valid_query():
    """
    Ensure /search returns 200 and correct structure.
    """
    response = client.post(
        "/search",
        json={"query": "machine learning"}
    )

    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

    first_item = data[0]
    assert "id" in first_item
    assert "text_snippet" in first_item
    assert "score" in first_item


def test_search_invalid_query():
    """
    Ensure /search returns 400 for invalid query.
    """
    response = client.post(
        "/search",
        json={"query": ""}
    )

    assert response.status_code == 400


def test_search_missing_query_field():
    """
    Ensure /search returns 422 when query field missing.
    """
    response = client.post(
        "/search",
        json={}
    )

    assert response.status_code == 422
