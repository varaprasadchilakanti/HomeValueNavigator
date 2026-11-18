# SPDX-License-Identifier: MIT
"""The HTTP API: list the locations, and estimate a price.

    GET  /api/locations  -> 200 {"locations": [...]}
    POST /api/predict    {"location": "Whitefield", "total_sqft": 1200, "bhk": 2, "bath": 2}
                         -> 200 {"location": "Whitefield", "estimated_price": 61.83,
                                 "likely_low": 52.09, "likely_high": 71.57, "unit": "lakh INR"}

The response names the location it matched, so a caller sees what was priced. Every
error is an RFC 9457 problem document (application/problem+json): 400 for a body that
is not a JSON object, 413 for one over 1 KiB, 415 for another content type, 422 for a
request the model cannot answer, with "field" naming the input at fault, and 500, logged,
for a fault of the server's own.

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
            location=model.location_name(body.get("location")),
            estimated_price=price,
            likely_low=round(price - spread, 2),
            likely_high=round(price + spread, 2),
            unit="lakh INR",
        )

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException) -> Response:
        return problem(error.code or 500, error.description or "")

    @app.errorhandler(Exception)
    def server_fault(error: Exception) -> Response:
        app.logger.exception("Unhandled error", exc_info=error)
        return problem(500, "The server failed to answer. Try again; if it persists, report it.")

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
