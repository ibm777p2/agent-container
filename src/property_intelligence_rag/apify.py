"""Minimal Apify REST client used by the property-data tool."""

from __future__ import annotations

from typing import Any

import requests

from property_intelligence_rag import config


class ApifyError(RuntimeError):
    """Base error for an Actor run that could not produce usable data."""


class ApifyConfigurationError(ApifyError):
    pass


class ApifyTimeoutError(ApifyError):
    pass


def run_actor(
    actor_id: str,
    payload: dict[str, Any],
    *,
    token: str | None = None,
    timeout_seconds: int | None = None,
) -> list[dict[str, Any]]:
    """Run an Actor synchronously and return its default dataset items."""
    api_token = token or config.APIFY_TOKEN
    if not api_token:
        raise ApifyConfigurationError("APIFY_TOKEN is not configured")
    normalized_id = actor_id.replace("/", "~")
    url = f"https://api.apify.com/v2/acts/{normalized_id}/run-sync-get-dataset-items"
    try:
        response = requests.post(
            url,
            params={"clean": "true", "format": "json"},
            headers={"Authorization": f"Bearer {api_token}"},
            json=payload,
            timeout=timeout_seconds or config.APIFY_TIMEOUT_SECONDS,
        )
    except requests.Timeout as error:
        raise ApifyTimeoutError(f"Actor {actor_id} timed out") from error
    except requests.RequestException as error:
        raise ApifyError(f"Actor {actor_id} request failed: {error}") from error

    if not response.ok:
        detail = response.text.strip().replace("\n", " ")[:300]
        raise ApifyError(
            f"Actor {actor_id} returned HTTP {response.status_code}: {detail}"
        )
    try:
        items = response.json()
    except requests.JSONDecodeError as error:
        raise ApifyError(f"Actor {actor_id} returned invalid JSON") from error
    if not isinstance(items, list):
        raise ApifyError(f"Actor {actor_id} returned an unexpected response shape")
    return [item for item in items if isinstance(item, dict)]
