# SPDX-License-Identifier: MIT
"""The HTTP API: list the locations, and estimate a price.

    GET  /api/locations  -> 200 {"locations": [...]}
    POST /api/predict    {"location": "Whitefield", "total_sqft": 1200, "bhk": 2, "bath": 2}
                         -> 200 {"estimated_price": 85.4, "likely_low": 72.1,
                                 "likely_high": 98.7, "unit": "lakh INR"}

Every error is an RFC 9457 problem document (application/problem+json): 400 for a body
that is not JSON, 415 for another content type, 422 for a request the model cannot
answer, with "field" naming the input at fault.

Configuration comes from the environment:
    HOMEVALUE_MODEL           path to model.json (default: the repository's model/model.json)
    HOMEVALUE_ALLOWED_ORIGIN  the one origin allowed to call cross-origin (default: none)
"""

from __future__ import annotations

import os
from http import HTTPStatus
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import HTTPException

from homevalue.model import InvalidInputError, PriceModel

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "model" / "model.json"
MAX_BODY_BYTES = 1024


def create_app(model_path: Path | None = None, allowed_origin: str | None = None) -> Flask:
    """Build the app around one model; arguments left as None are read from the environment."""
    path = model_path or Path(os.environ.get("HOMEVALUE_MODEL", DEFAULT_MODEL))
    origin = _origin(allowed_origin, os.environ.get("HOMEVALUE_ALLOWED_ORIGIN", ""))
    model = PriceModel.from_json(path)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES

    @app.get("/api/locations")
    def locations() -> Response:
        return jsonify(locations=model.location_names())

    @app.post("/api/predict")
    def predict() -> Response:
        if not request.is_json:
            return problem(415, "Send the body as application/json.")
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return problem(400, "The body must be a JSON object.")
        try:
            args = (body.get(k) for k in ("location", "total_sqft", "bhk", "bath"))
            price = model.predict(*args)
        except InvalidInputError as refused:
            return problem(422, str(refused), field=refused.field)
        spread = price * model.median_error
        return jsonify(
            estimated_price=price,
            likely_low=round(price - spread, 2),
            likely_high=round(price + spread, 2),
            unit="lakh INR",
        )

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException) -> Response:
        return problem(error.code or 500, error.description or "")

    @app.after_request
    def cross_origin(response: Response) -> Response:
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Vary"] = "Origin"
        return response

    return app


def problem(status: int, detail: str, **extra: Any) -> Response:
    """An RFC 9457 problem document."""
    body = {"type": "about:blank", "title": HTTPStatus(status).phrase, "status": status}
    body |= {"detail": detail, **{k: v for k, v in extra.items() if v is not None}}
    response = jsonify(body)
    response.status_code = status
    response.mimetype = "application/problem+json"
    return response


def _origin(argument: str | None, setting: str) -> str:
    """One exact origin, or none; a wildcard is refused at start-up, not served."""
    value = argument if argument is not None else setting
    if "*" in value:
        raise ValueError("HOMEVALUE_ALLOWED_ORIGIN must name one origin, not a wildcard.")
    return value
