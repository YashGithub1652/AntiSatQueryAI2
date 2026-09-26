from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict

import httpx


logger = logging.getLogger(__name__)


GEOCHAT_URL = os.getenv(
    "SATQUERY_GEOCHAT_URL",
    "http://127.0.0.1:8100",
)


def image_bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


def run_geochat(
    image_bytes: bytes,
    query: str,
    timeout: float = 300.0,
) -> Dict[str, Any]:

    payload = {
        "image_base64": image_bytes_to_base64(
            image_bytes
        ),
        "query": query,
        "max_new_tokens": 256,
    }

    try:

        with httpx.Client(
            timeout=timeout
        ) as client:

            response = client.post(
                f"{GEOCHAT_URL}/v1/vqa",
                json=payload,
            )

        response.raise_for_status()

        data = response.json()

        if not data.get("success"):
            return {
                "success": False,
                "answer": "",
                "model": "GeoChat-7B",
                "scientific": False,
                "fallback": False,
                "error": data.get(
                    "error",
                    "GeoChat inference failed.",
                ),
            }

        return {
            "success": True,
            "answer": data["answer"],
            "model": data.get(
                "model",
                "GeoChat-7B",
            ),
            "scientific": bool(
                data.get("scientific", True)
            ),
            "fallback": False,
            "latency_sec": data.get(
                "latency_sec",
                0.0,
            ),
            "error": None,
        }

    except Exception as exc:

        logger.exception(
            "GeoChat service request failed."
        )

        return {
            "success": False,
            "answer": "",
            "model": "GeoChat-7B",
            "scientific": False,
            "fallback": False,
            "error": str(exc),
        }
