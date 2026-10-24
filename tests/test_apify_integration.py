"""Opt-in paid/network integration test.

Run with PI_RUN_LIVE_APIFY_TEST=1, APIFY_TOKEN, and optionally
PI_LIVE_TEST_ADDRESS set. It is skipped during normal test runs.
"""

import os

import pytest

from property_intelligence_rag.property_cache import PropertyCache
from property_intelligence_rag.property_data import get_property_data


@pytest.mark.skipif(
    os.getenv("PI_RUN_LIVE_APIFY_TEST") != "1",
    reason="live Apify test is opt-in and incurs external usage",
)
def test_live_property_retrieval(tmp_path):
    result = get_property_data(
        os.getenv(
            "PI_LIVE_TEST_ADDRESS", "52 Timber Ridge Dr, Staten Island, NY 10306"
        ),
        refresh=True,
        cache=PropertyCache(tmp_path / "live.sqlite3"),
    )
    assert result["subject"]["id"]
    assert result["subject"]["address"]["formatted"]
    assert result["market_summary"]["label"] == "current comparable market snapshot"
