"""The HTTP API: list the locations, and estimate a price.

    uv run flask --app homevalue.server run

    GET  /api/locations  -> {"locations": [...]}
    POST /api/predict    {"location": "...", "total_sqft": 1200, "bhk": 2, "bath": 2}
                         -> {"estimated_price": 85.4, "typical_error": 19.5,
                             "unit": "lakh INR"}
                         or 422 {"error": "<why it was refused>"}

A cross-origin caller is allowed only from the origin named in HOMEVALUE_ALLOWED_ORIGIN;
unset, none is. The 2024 server allowed every origin.
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, Response, jsonify, request

from homevalue.model import InvalidInputError, PriceModel

MODEL = Path(__file__).resolve().parents[2] / "model" / "model.json"
ORIGIN_SETTING = "HOMEVALUE_ALLOWED_ORIGIN"


def create_app(model_path: Path = MODEL, allowed_origin: str | None = None) -> Flask:
    """Build the app around one model. The origin defaults to the environment setting."""
    model = PriceModel.from_json(model_path)
    origin = allowed_origin if allowed_origin is not None else os.environ.get(ORIGIN_SETTING)
    app = Flask(__name__)

    @app.get("/api/locations")
    def locations() -> Response:
        return jsonify(locations=model.location_names())

    @app.post("/api/predict")
    def predict() -> tuple[Response, int]:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error="The request body must be a JSON object."), 422
        try:
            price = model.predict(
                body.get("location"), body.get("total_sqft"), body.get("bhk"), body.get("bath")
            )
        except InvalidInputError as refused:
            return jsonify(error=str(refused)), 422
        return jsonify(
            estimated_price=price, typical_error=model.typical_error, unit="lakh INR"
        ), 200

    @app.after_request
    def cross_origin(response: Response) -> Response:
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Vary"] = "Origin"
        return response

    return app
