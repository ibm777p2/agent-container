import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from property_intelligence_rag.property_cache import PropertyCache
from property_intelligence_rag.property_data import (
    PropertyDataError,
    build_market_summary,
    get_property_data,
    normalize_zillow_record,
    select_comparables,
)


FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text())


def actor_fixture(calls, fail_operation=None):
    mapping = {
        "buy-detail": load("zillow_subject.json"),
        "buy": load("zillow_active.json"),
        "sold": load("zillow_sold.json"),
        "rent": load("zillow_rentals.json"),
    }

    def runner(actor_id, payload):
        calls.append(payload)
        operation = payload["operation"]
        if operation == fail_operation:
            raise RuntimeError("fixture failure")
        key = "buy-detail" if payload.get("fetchDetails") else operation
        return mapping[key]

    return runner


@pytest.fixture
def cache(tmp_path):
    return PropertyCache(tmp_path / "property.sqlite3")


def test_property_retrieval_market_summary_and_cache(cache):
    calls = []
    now = datetime(2026, 9, 26, tzinfo=timezone.utc)
    result = get_property_data(
        "123 Main St, Austin, TX 78704",
        cache=cache,
        actor_runner=actor_fixture(calls),
        now=now,
    )
    assert len(calls) == 4
    assert result["subject"]["id"] == "100"
    assert [item["id"] for item in result["active_comps"]] == ["101", "102"]
    assert len(result["sold_comps"]) == 3
    assert len(result["rental_comps"]) == 3
    assert result["market_summary"]["active"]["median_asking_price"] == 600000
    assert result["market_summary"]["sold"]["median_sale_price"] == 570000
    assert result["market_summary"]["rentals"]["median_asking_rent"] == 3200
    assert (
        result["market_summary"]["subject_comparison"][
            "asking_price_vs_sold_median_pct"
        ]
        == 5.26
    )
    assert result["cache"]["status"] == "miss"

    cached = get_property_data(
        "123 main st Austin TX 78704",
        cache=cache,
        actor_runner=lambda *_: pytest.fail("cache should avoid actor"),
        now=now,
    )
    assert cached["cache"]["status"] == "hit"


def test_refresh_bypasses_fresh_cache(cache):
    cache.put("123 Main St", {"cached": True}, ttl_seconds=60)
    calls = []
    result = get_property_data(
        "123 Main St",
        refresh=True,
        cache=cache,
        actor_runner=actor_fixture(calls),
        now=datetime(2026, 9, 26, tzinfo=timezone.utc),
    )
    assert result["subject"]["id"] == "100"
    assert len(calls) == 4


def test_optional_failure_returns_partial_response(cache):
    result = get_property_data(
        "123 Main St",
        cache=cache,
        actor_runner=actor_fixture([], fail_operation="rent"),
        now=datetime(2026, 9, 26, tzinfo=timezone.utc),
    )
    assert result["rental_comps"] == []
    assert result["unavailable_sections"] == ["rental_comps"]
    assert result["subject"]["id"] == "100"


def test_sparse_sold_results_widen_to_twelve_months(cache):
    calls = []

    def runner(actor_id, payload):
        calls.append(payload)
        if payload.get("fetchDetails"):
            return load("zillow_subject.json")
        if payload["operation"] == "buy":
            return load("zillow_active.json")
        if payload["operation"] == "rent":
            return load("zillow_rentals.json")
        sold = load("zillow_sold.json")
        return sold[:2] if payload["maxSoldDate"] == "6m" else sold

    result = get_property_data(
        "123 Main St",
        cache=cache,
        actor_runner=runner,
        now=datetime(2026, 9, 26, tzinfo=timezone.utc),
    )
    assert len(calls) == 5
    assert len(result["sold_comps"]) == 3
    assert "sold_comps search widened from six to twelve months" in result["warnings"]


def test_subject_failure_uses_stale_cache(cache):
    cache.put(
        "123 Main St", {"warnings": [], "subject": {"id": "cached"}}, ttl_seconds=-1
    )
    result = get_property_data(
        "123 Main St",
        cache=cache,
        actor_runner=lambda *_: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    assert result["subject"]["id"] == "cached"
    assert result["cache"]["status"] == "stale"


def test_subject_failure_without_fallback_raises(cache, monkeypatch):
    from property_intelligence_rag import config

    monkeypatch.setattr(config, "DEMO_FALLBACK", False)
    with pytest.raises(PropertyDataError, match="Subject property retrieval failed"):
        get_property_data(
            "123 Main St",
            cache=cache,
            actor_runner=lambda *_: (_ for _ in ()).throw(RuntimeError("offline")),
        )


def test_missing_numbers_stay_null():
    record = normalize_zillow_record(
        {"zpid": 1, "address": {"streetAddress": "1 Test St"}}, "buy"
    )
    assert record["price"] is None
    assert record["price_per_sqft"] is None
    summary = build_market_summary(record, [], [], [])
    assert summary["active"]["median_asking_price"] is None
    assert summary["subject_comparison"]["asking_price_vs_active_median_pct"] is None


def test_comparable_filtering_enforces_type_size_beds_and_age():
    subject = normalize_zillow_record(load("zillow_subject.json")[0], "buy")
    sold = [normalize_zillow_record(item, "sold") for item in load("zillow_sold.json")]
    selected = select_comparables(
        subject,
        sold,
        operation="sold",
        now=datetime(2026, 9, 26, tzinfo=timezone.utc),
    )
    assert {item["id"] for item in selected} == {"201", "202", "203"}
