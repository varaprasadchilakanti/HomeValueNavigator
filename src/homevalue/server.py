# SPDX-License-Identifier: MIT
"""The HTTP API: list the locations, and estimate a price.

    GET  /api/locations  -> 200 {"locations": [...]}
    POST /api/predict    {"location": "Whitefield", "total_sqft": 1200, "bhk": 2, "bath": 2}
                         -> 200 {"location": "Whitefield", "estimated_price": 62.12,
                                 "likely_low": 52.01, "likely_high": 72.23, "unit": "lakh INR"}

The response names the location it matched, so a caller sees what was priced. Every
error is an RFC 9457 problem document (application/problem+json): 400 for a body that
is not a JSON object, 411 for one sent without a length, 413 for one over 1 KiB, 415 for
another content type, 422 for a request the model cannot answer, with "field" naming the
input at fault, and 500, logged, for a fault of the server's own.

Configuration comes from the environment:
    HOMEVALUE_MODEL  path to model.json (default: model/model.json in a checkout)
"""

from __future__ import annotations

import os
from http import HTTPStatus
from pathlib import Path
from typing import Any

from flask import Blueprint, Flask, Response, current_app, jsonify, request
from werkzeug.exceptions import HTTPException

from homevalue.model import InvalidInputError, PriceModel

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "model" / "model.json"
MAX_BODY_BYTES = 1024
FIELDS = ("location", "total_sqft", "bhk", "bath")
_MODEL = "homevalue.model"  # the key the model is kept under in app.extensions

api = Blueprint("api", __name__, url_prefix="/api")


def create_app(model_path: Path | None = None) -> Flask:
    """Build the app around one model; with no path, HOMEVALUE_MODEL names it."""
    path = model_path or Path(os.environ.get("HOMEVALUE_MODEL", DEFAULT_MODEL))
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
    app.extensions[_MODEL] = PriceModel.from_json(path)
    app.register_blueprint(api)
    app.register_error_handler(HTTPException, _http_error)
    app.register_error_handler(Exception, _server_fault)
    app.after_request(_no_sniffing)
    return app


@api.get("/locations")
def locations() -> Response:
    """The names a price can be asked for."""
    return jsonify(locations=_model().location_names())


@api.post("/predict")
def predict() -> Response:
    """An estimate and its typical range, or a problem document saying why not."""
    if not request.is_json:
        return problem(415, "Send the body as application/json.")
    if request.content_length is None:
        return problem(
            411, f"Send a Content-Length; the body may be at most {MAX_BODY_BYTES} bytes."
        )
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return problem(400, "The body must be a JSON object.")
    try:
        estimate = _model().estimate(*(body.get(k) for k in FIELDS))
    except InvalidInputError as refused:
        return problem(422, str(refused), field=refused.field)
    return jsonify(
        location=estimate.location,
        estimated_price=estimate.price,
        likely_low=estimate.low,
        likely_high=estimate.high,
        unit="lakh INR",
    )


def problem(status: int, detail: str, **extra: Any) -> Response:
    """An RFC 9457 problem document."""
    body = {"type": "about:blank", "title": HTTPStatus(status).phrase, "status": status}
    body |= {"detail": detail, **{k: v for k, v in extra.items() if v is not None}}
    response = jsonify(body)
    response.status_code = status
    response.mimetype = "application/problem+json"
    return response


def _model() -> PriceModel:
    model: PriceModel = current_app.extensions[_MODEL]
    return model


def _http_error(error: HTTPException) -> Response:
    return problem(error.code or 500, error.description or "")


def _server_fault(error: Exception) -> Response:
    current_app.logger.exception("Unhandled error", exc_info=error)
    return problem(500, "The server failed to answer. Try again; if it persists, report it.")


def _no_sniffing(response: Response) -> Response:
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response
