# SPDX-License-Identifier: MIT
from __future__ import annotations

import io
from pathlib import Path

import pytest
from flask.testing import FlaskClient

from homevalue.server import create_app

VALID = {"location": "Whitefield", "total_sqft": 1000, "bhk": 2, "bath": 2}


@pytest.fixture
def client(model_path: Path) -> FlaskClient:
    return create_app(model_path).test_client()


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


def test_a_body_without_a_length_is_a_411(client: FlaskClient) -> None:
    """A chunked body has no length to check against the limit, so it is refused."""
    response = client.post(
        "/api/predict",
        input_stream=io.BytesIO(b"{}"),
        content_type="application/json",
        headers={"Transfer-Encoding": "chunked"},
    )
    assert response.status_code == 411
    assert response.mimetype == "application/problem+json"


def test_responses_forbid_content_sniffing(client: FlaskClient) -> None:
    assert client.get("/api/locations").headers["X-Content-Type-Options"] == "nosniff"


def test_the_model_path_is_read_from_the_environment(
    model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOMEVALUE_MODEL", str(model_path))
    response = create_app().test_client().get("/api/locations")
    assert response.get_json() == {"locations": ["Hebbal", "Whitefield"]}


def test_a_fault_of_the_server_is_a_500_problem_not_a_stack_trace(model_path: Path) -> None:
    app = create_app(model_path)

    @app.get("/fault")
    def fault() -> str:
        raise RuntimeError("internal detail")

    response = app.test_client().get("/fault")
    assert response.status_code == 500
    assert response.mimetype == "application/problem+json"
    assert "internal detail" not in response.get_data(as_text=True)
