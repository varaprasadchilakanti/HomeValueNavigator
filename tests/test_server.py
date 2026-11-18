# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest
from flask.testing import FlaskClient

from homevalue.server import create_app

VALID = {"location": "Whitefield", "total_sqft": 1000, "bhk": 2, "bath": 2}


@pytest.fixture
def client(model_path: Path) -> FlaskClient:
    return create_app(model_path, allowed_origin="").test_client()


def test_locations_are_listed_without_the_pooled_bucket(client: FlaskClient) -> None:
    response = client.get("/api/locations")
    assert response.status_code == 200
    assert response.get_json() == {"locations": ["Hebbal", "Whitefield"]}


def test_a_valid_request_gets_a_price_and_its_likely_range(client: FlaskClient) -> None:
    response = client.post("/api/predict", json=VALID)
    assert response.status_code == 200
    assert response.get_json() == {
        "location": "Whitefield",
        "estimated_price": 75.0,
        "likely_low": 60.0,
        "likely_high": 90.0,
        "unit": "lakh INR",
    }


def test_a_refusal_is_a_422_problem_naming_the_field(client: FlaskClient) -> None:
    response = client.post("/api/predict", json=VALID | {"location": "Atlantis"})
    assert response.status_code == 422
    assert response.mimetype == "application/problem+json"
    body = response.get_json(force=True)
    assert body["status"] == 422
    assert body["field"] == "location"
    assert "not a listed location" in body["detail"]


@pytest.mark.parametrize(
    ("data", "content_type", "status", "detail"),
    [
        pytest.param(b"not json", "application/json", 400, "JSON object", id="not-json"),
        pytest.param(b"[1, 2]", "application/json", 400, "JSON object", id="a-list"),
        pytest.param(b"\xff\xfe{}", "application/json", 400, "JSON object", id="not-utf-8"),
        pytest.param(
            b"location=Hebbal",
            "application/x-www-form-urlencoded",
            415,
            "application/json",
            id="a-form",
        ),
    ],
)
def test_a_malformed_body_gets_the_matching_status(
    client: FlaskClient, data: bytes, content_type: str, status: int, detail: str
) -> None:
    response = client.post("/api/predict", data=data, content_type=content_type)
    assert response.status_code == status
    assert response.mimetype == "application/problem+json"
    assert detail in response.get_json(force=True)["detail"]


def test_an_oversized_body_is_a_413(client: FlaskClient) -> None:
    response = client.post("/api/predict", json=VALID | {"padding": "x" * 2000})
    assert response.status_code == 413
    assert response.mimetype == "application/problem+json"


def test_an_unknown_path_is_a_404_problem_not_an_html_page(client: FlaskClient) -> None:
    response = client.get("/nowhere")
    assert response.status_code == 404
    assert response.mimetype == "application/problem+json"


def test_no_origin_is_allowed_unless_one_is_set(model_path: Path) -> None:
    closed = create_app(model_path, allowed_origin="").test_client()
    assert "Access-Control-Allow-Origin" not in closed.get("/api/locations").headers
    opened = create_app(model_path, allowed_origin="https://example.org").test_client()
    headers = opened.get("/api/locations").headers
    assert headers["Access-Control-Allow-Origin"] == "https://example.org"


def test_a_wildcard_origin_is_refused_at_start_up(model_path: Path) -> None:
    with pytest.raises(ValueError, match="not a wildcard"):
        create_app(model_path, allowed_origin="*")


def test_settings_are_read_from_the_environment(
    model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOMEVALUE_MODEL", str(model_path))
    monkeypatch.setenv("HOMEVALUE_ALLOWED_ORIGIN", "https://example.org")
    response = create_app().test_client().get("/api/locations")
    assert response.get_json() == {"locations": ["Hebbal", "Whitefield"]}
    assert response.headers["Access-Control-Allow-Origin"] == "https://example.org"


def test_a_fault_of_the_server_is_a_500_problem_not_a_stack_trace(model_path: Path) -> None:
    app = create_app(model_path, allowed_origin="")

    @app.get("/fault")
    def fault() -> str:
        raise RuntimeError("internal detail")

    response = app.test_client().get("/fault")
    assert response.status_code == 500
    assert response.mimetype == "application/problem+json"
    assert "internal detail" not in response.get_data(as_text=True)
