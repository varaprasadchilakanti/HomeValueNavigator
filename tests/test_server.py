from __future__ import annotations

from pathlib import Path

import pytest
from flask.testing import FlaskClient

from homevalue.server import create_app


@pytest.fixture
def client(model_path: Path) -> FlaskClient:
    return create_app(model_path, allowed_origin="").test_client()


def test_locations_are_listed(client: FlaskClient) -> None:
    response = client.get("/api/locations")
    assert response.status_code == 200
    assert response.get_json() == {"locations": ["Hebbal", "Whitefield", "other"]}


def test_a_valid_request_gets_a_price(client: FlaskClient) -> None:
    body = {"location": "Whitefield", "total_sqft": 1000, "bhk": 2, "bath": 2}
    response = client.post("/api/predict", json=body)
    assert response.status_code == 200
    assert response.get_json() == {
        "estimated_price": 75.0,
        "typical_error": 12.5,
        "unit": "lakh INR",
    }


def test_an_unknown_location_is_a_422_with_the_reason(client: FlaskClient) -> None:
    body = {"location": "Atlantis", "total_sqft": 1000, "bhk": 2, "bath": 2}
    response = client.post("/api/predict", json=body)
    assert response.status_code == 422
    assert "not a location the model was trained on" in response.get_json()["error"]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"not json", id="not-json"),
        pytest.param(b"[1, 2]", id="a-list"),
        pytest.param(b'"text"', id="a-string"),
    ],
)
def test_a_body_that_is_not_an_object_is_a_422(client: FlaskClient, payload: bytes) -> None:
    response = client.post("/api/predict", data=payload, content_type="application/json")
    assert response.status_code == 422
    assert "JSON object" in response.get_json()["error"]


def test_no_origin_is_allowed_unless_one_is_set(model_path: Path) -> None:
    closed = create_app(model_path, allowed_origin="").test_client()
    assert "Access-Control-Allow-Origin" not in closed.get("/api/locations").headers
    opened = create_app(model_path, allowed_origin="https://example.org").test_client()
    headers = opened.get("/api/locations").headers
    assert headers["Access-Control-Allow-Origin"] == "https://example.org"


def test_the_origin_is_read_from_the_environment(
    model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOMEVALUE_ALLOWED_ORIGIN", "https://example.org")
    response = create_app(model_path).test_client().get("/api/locations")
    assert response.headers["Access-Control-Allow-Origin"] == "https://example.org"
