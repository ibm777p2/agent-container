"""Retrieve, normalize, compare, and cache live residential property data."""

from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable

from property_intelligence_rag import config
from property_intelligence_rag.apify import run_actor
from property_intelligence_rag.property_cache import (
    PropertyCache,
    isoformat,
    normalize_address,
    utc_now,
)


ActorRunner = Callable[[str, dict[str, Any]], list[dict[str, Any]]]
MAX_COMPS = 10
MIN_COMPS = 3


class PropertyDataError(RuntimeError):
    pass


def _dig(value: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _first(*values: Any) -> Any:
    return next((value for value in values if value not in (None, "", [])), None)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    cleaned = re.sub(r"[^\d.-]", "", str(value))
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        # Zillow commonly uses milliseconds for sold timestamps.
        seconds = value / 1000 if value > 10_000_000_000 else value
        try:
            return datetime.fromtimestamp(seconds, timezone.utc).date().isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    for candidate in (text, text[:10]):
        try:
            return (
                datetime.fromisoformat(candidate.replace("Z", "+00:00"))
                .date()
                .isoformat()
            )
        except ValueError:
            continue
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return match.group(0) if match else None


def _address(raw: dict[str, Any]) -> dict[str, Any]:
    value = raw.get("address")
    if not isinstance(value, dict):
        value = {}
    street = _first(
        value.get("streetAddress"), raw.get("streetAddress"), raw.get("addressStreet")
    )
    city = _first(value.get("city"), raw.get("city"), raw.get("addressCity"))
    state = _first(value.get("state"), raw.get("state"), raw.get("addressState"))
    postal_code = _first(
        value.get("zipcode"),
        value.get("postalCode"),
        raw.get("zipcode"),
        raw.get("zipCode"),
    )
    formatted = _first(raw.get("formattedAddress"), raw.get("addressString"))
    if not formatted:
        pieces = [str(piece) for piece in (street, city, state, postal_code) if piece]
        formatted = ", ".join(pieces)
    if not formatted and isinstance(raw.get("address"), str):
        formatted = raw["address"]
    return {
        "street": street,
        "city": city,
        "state": state,
        "postal_code": str(postal_code) if postal_code is not None else None,
        "formatted": formatted or None,
    }


def _photos(raw: dict[str, Any]) -> list[str]:
    candidates = (
        _first(
            _dig(raw, "media", "allPropertyPhotos", "highResolution"),
            raw.get("photos"),
            [raw.get("imgSrc")] if raw.get("imgSrc") else None,
        )
        or []
    )
    result = []
    for item in candidates:
        url = item.get("url") if isinstance(item, dict) else item
        if isinstance(url, str) and url.startswith("http") and url not in result:
            result.append(url)
    return result[:20]


def normalize_zillow_record(raw: dict[str, Any], operation: str) -> dict[str, Any]:
    details = raw.get("_details") if isinstance(raw.get("_details"), dict) else {}
    reso = (
        details.get("resoFacts") if isinstance(details.get("resoFacts"), dict) else {}
    )
    home_info = _dig(raw, "hdpData", "homeInfo") or {}
    address = _address(raw)

    generic_price = _first(
        _dig(raw, "price", "value"), raw.get("unformattedPrice"), raw.get("price")
    )
    sold_price = _first(
        raw.get("soldPrice"), raw.get("lastSoldPrice"), home_info.get("price")
    )
    rent = _first(raw.get("baseRent"), raw.get("minBaseRent"), generic_price)
    if operation == "sold":
        price = _number(sold_price or generic_price)
    elif operation == "rent":
        price = _number(rent)
    else:
        price = _number(generic_price or home_info.get("price"))

    sqft = _number(
        _first(
            raw.get("livingArea"),
            raw.get("livingAreaValue"),
            home_info.get("livingArea"),
        )
    )
    price_per_sqft = _number(
        _first(
            _dig(raw, "price", "pricePerSquareFoot"),
            raw.get("pricePerSqft"),
            reso.get("pricePerSquareFoot"),
        )
    )
    if price_per_sqft is None and price is not None and sqft:
        price_per_sqft = price / sqft

    location = raw.get("location") if isinstance(raw.get("location"), dict) else {}
    latitude = _number(
        _first(location.get("latitude"), raw.get("latitude"), home_info.get("latitude"))
    )
    longitude = _number(
        _first(
            location.get("longitude"), raw.get("longitude"), home_info.get("longitude")
        )
    )
    variable = (
        raw.get("variableData") if isinstance(raw.get("variableData"), dict) else {}
    )
    sold_date = _date(
        _first(
            raw.get("soldDate"),
            raw.get("dateSold"),
            variable.get("data"),
            variable.get("text"),
        )
    )

    property_type = _first(
        raw.get("homeType"),
        raw.get("propertyType"),
        home_info.get("homeType"),
        reso.get("homeType"),
    )
    estimates = raw.get("estimates") if isinstance(raw.get("estimates"), dict) else {}
    result = {
        "id": str(_first(raw.get("zpid"), raw.get("id"), raw.get("palsId"), ""))
        or None,
        "source": "zillow",
        "source_url": _first(raw.get("url"), raw.get("hdpUrl"), raw.get("detailUrl")),
        "operation": operation,
        "address": address,
        "latitude": latitude,
        "longitude": longitude,
        "status": _first(
            _dig(raw, "listing", "listingStatus"),
            raw.get("statusType"),
            raw.get("statusText"),
        ),
        "price": price,
        "price_per_sqft": round(price_per_sqft, 2)
        if price_per_sqft is not None
        else None,
        "bedrooms": _number(
            _first(raw.get("bedrooms"), raw.get("beds"), home_info.get("bedrooms"))
        ),
        "bathrooms": _number(
            _first(raw.get("bathrooms"), raw.get("baths"), home_info.get("bathrooms"))
        ),
        "living_area_sqft": sqft,
        "lot_area": _number(
            _first(
                _dig(raw, "lotSizeWithUnit", "lotSize"),
                details.get("lotAreaValue"),
                reso.get("lotSize"),
            )
        ),
        "lot_area_unit": _first(
            _dig(raw, "lotSizeWithUnit", "lotSizeUnit"), details.get("lotAreaUnits")
        ),
        "year_built": _number(_first(raw.get("yearBuilt"), reso.get("yearBuilt"))),
        "property_type": str(property_type) if property_type else None,
        "days_on_market": _number(
            _first(raw.get("daysOnZillow"), home_info.get("daysOnZillow"))
        ),
        "sold_date": sold_date,
        "zestimate": _number(
            _first(
                estimates.get("zestimate"),
                details.get("zestimate"),
                raw.get("zestimate"),
            )
        ),
        "rent_estimate": _number(
            _first(
                estimates.get("rentZestimate"),
                details.get("rentZestimate"),
                raw.get("rentZestimate"),
            )
        ),
        "hoa_monthly": _number(_first(reso.get("hoaFee"), raw.get("hoaFee"))),
        "parcel_id": _first(details.get("parcelId"), reso.get("parcelNumber")),
        "photos": _photos(raw),
        "description": details.get("description") or raw.get("description"),
        "price_history": details.get("priceHistory") or raw.get("priceHistory") or [],
        "tax_history": details.get("taxHistory") or raw.get("taxHistory") or [],
    }
    return result


def _property_type(value: str | None) -> str:
    text = re.sub(r"[^a-z]", "", (value or "").casefold())
    aliases = {
        "singlefamily": "house",
        "house": "house",
        "townhouse": "townhouse",
        "townhome": "townhouse",
        "condo": "condo",
        "condominium": "condo",
        "multifamily": "multifamily",
        "apartment": "apartment",
        "manufactured": "manufactured",
    }
    return aliases.get(text, text)


def _distance_miles(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    if None in (
        a.get("latitude"),
        a.get("longitude"),
        b.get("latitude"),
        b.get("longitude"),
    ):
        return None
    lat1, lon1, lat2, lon2 = map(
        math.radians, (a["latitude"], a["longitude"], b["latitude"], b["longitude"])
    )
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 3958.8 * 2 * math.asin(math.sqrt(value))


def _months_ago(date_text: str | None, now: datetime) -> float | None:
    if not date_text:
        return None
    try:
        value = datetime.fromisoformat(date_text).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return max(0.0, (now - value).days / 30.44)


def select_comparables(
    subject: dict[str, Any],
    records: Iterable[dict[str, Any]],
    *,
    operation: str,
    limit: int = MAX_COMPS,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Filter and rank comparables with transparent deterministic rules."""
    current = now or utc_now()
    subject_type = _property_type(subject.get("property_type"))
    subject_zip = _dig(subject, "address", "postal_code")
    subject_sqft = subject.get("living_area_sqft")
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for record in records:
        if subject.get("id") and record.get("id") == subject.get("id"):
            continue
        if record.get("price") is None:
            continue
        record_type = _property_type(record.get("property_type"))
        if subject_type and record_type and record_type != subject_type:
            continue
        bed_difference = None
        if subject.get("bedrooms") is not None and record.get("bedrooms") is not None:
            bed_difference = abs(subject["bedrooms"] - record["bedrooms"])
            if bed_difference > 1:
                continue
        bath_difference = None
        if subject.get("bathrooms") is not None and record.get("bathrooms") is not None:
            bath_difference = abs(subject["bathrooms"] - record["bathrooms"])
            if bath_difference > 1:
                continue
        size_difference = None
        if subject_sqft and record.get("living_area_sqft"):
            size_difference = (
                abs(record["living_area_sqft"] - subject_sqft) / subject_sqft
            )
            if size_difference > 0.25:
                continue
        age_months = _months_ago(record.get("sold_date"), current)
        if operation == "sold" and age_months is not None and age_months > 12:
            continue
        distance = _distance_miles(subject, record)
        same_zip = bool(
            subject_zip and _dig(record, "address", "postal_code") == subject_zip
        )
        completeness = sum(
            record.get(field) is not None
            for field in (
                "latitude",
                "longitude",
                "living_area_sqft",
                "price_per_sqft",
                "days_on_market",
            )
        )
        if operation == "sold":
            completeness += int(record.get("sold_date") is not None)
        score = (
            0 if same_zip else 1,
            size_difference if size_difference is not None else 1.0,
            bed_difference if bed_difference is not None else 9,
            bath_difference if bath_difference is not None else 9,
            distance if distance is not None else 9999,
            -completeness,
            str(record.get("id") or ""),
        )
        enriched = dict(record)
        enriched["distance_miles"] = (
            round(distance, 2) if distance is not None else None
        )
        enriched["similarity"] = {
            "same_postal_code": same_zip,
            "size_difference_pct": round(size_difference * 100, 1)
            if size_difference is not None
            else None,
            "bedroom_difference": bed_difference,
            "bathroom_difference": bath_difference,
        }
        candidates.append((score, enriched))
    return [
        record for _, record in sorted(candidates, key=lambda item: item[0])[:limit]
    ]


def _median(records: list[dict[str, Any]], field: str) -> float | None:
    values = [
        record[field]
        for record in records
        if isinstance(record.get(field), (int, float))
    ]
    return round(float(median(values)), 2) if values else None


def _range(records: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    values = [
        float(record[field])
        for record in records
        if isinstance(record.get(field), (int, float))
    ]
    return (
        {"min": min(values), "max": max(values)}
        if values
        else {"min": None, "max": None}
    )


def _percent_difference(value: float | None, reference: float | None) -> float | None:
    if value is None or reference in (None, 0):
        return None
    return round((value - reference) / reference * 100, 2)


def build_market_summary(
    subject: dict[str, Any],
    active: list[dict[str, Any]],
    sold: list[dict[str, Any]],
    rentals: list[dict[str, Any]],
) -> dict[str, Any]:
    active_median = _median(active, "price")
    sold_median = _median(sold, "price")
    rental_median = _median(rentals, "price")
    sold_dates = sorted(
        record["sold_date"] for record in sold if record.get("sold_date")
    )
    return {
        "label": "current comparable market snapshot",
        "active": {
            "comparable_count": len(active),
            "median_asking_price": active_median,
            "median_asking_price_per_sqft": _median(active, "price_per_sqft"),
            "median_days_on_market": _median(active, "days_on_market"),
            "asking_price_range": _range(active, "price"),
        },
        "sold": {
            "comparable_count": len(sold),
            "median_sale_price": sold_median,
            "median_sale_price_per_sqft": _median(sold, "price_per_sqft"),
            "median_days_on_market": _median(sold, "days_on_market"),
            "sale_price_range": _range(sold, "price"),
            "date_range": {
                "from": sold_dates[0] if sold_dates else None,
                "to": sold_dates[-1] if sold_dates else None,
            },
        },
        "rentals": {
            "comparable_count": len(rentals),
            "median_asking_rent": rental_median,
            "median_asking_rent_per_sqft": _median(rentals, "price_per_sqft"),
            "asking_rent_range": _range(rentals, "price"),
        },
        "subject_comparison": {
            "asking_price_vs_active_median_pct": _percent_difference(
                subject.get("price"), active_median
            ),
            "asking_price_vs_sold_median_pct": _percent_difference(
                subject.get("price"), sold_median
            ),
            "price_per_sqft_vs_sold_median_pct": _percent_difference(
                subject.get("price_per_sqft"), _median(sold, "price_per_sqft")
            ),
            "rent_estimate_vs_rental_median_pct": _percent_difference(
                subject.get("rent_estimate"), rental_median
            ),
        },
    }


def _select_subject(address: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise PropertyDataError("No subject property was returned for that address")
    target = normalize_address(address)
    ranked = []
    for record in records:
        formatted = _dig(record, "address", "formatted") or ""
        similarity = SequenceMatcher(
            None, target, normalize_address(formatted) if formatted else ""
        ).ratio()
        ranked.append((similarity, record))
    return max(ranked, key=lambda item: item[0])[1]


def _home_type_filter(property_type: str | None) -> list[str] | None:
    mapping = {
        "house": "houses",
        "townhouse": "townhomes",
        "condo": "condos",
        "multifamily": "multifamily",
        "apartment": "apartments",
        "manufactured": "manufactured",
    }
    result = mapping.get(_property_type(property_type))
    return [result] if result else None


def _fallback_payload(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def get_property_data(
    address: str,
    refresh: bool = False,
    *,
    cache: PropertyCache | None = None,
    actor_runner: ActorRunner = run_actor,
    now: datetime | None = None,
) -> dict[str, Any]:
    normalized = normalize_address(address)
    cache_store = cache or PropertyCache(config.PROPERTY_CACHE_PATH)
    if not refresh:
        cached = cache_store.get(address)
        if cached:
            result = dict(cached.payload)
            result["cache"] = {"status": "hit", "stale": False}
            return result
    stale = cache_store.get(address, allow_stale=True)
    current = (now or utc_now()).astimezone(timezone.utc)

    try:
        subject_raw = actor_runner(
            config.ZILLOW_ACTOR,
            {
                "location": address.strip(),
                "locationSelectionMode": "similarity",
                "operation": "buy",
                "sortBy": "relevance",
                "maxItems": 1,
                "fetchDetails": True,
            },
        )
        subject_records = [normalize_zillow_record(item, "buy") for item in subject_raw]
        subject = _select_subject(normalized, subject_records)
    except Exception as error:
        if stale:
            result = dict(stale.payload)
            result.setdefault("warnings", []).append(
                f"Live refresh failed; serving stale cache: {error}"
            )
            result["cache"] = {"status": "stale", "stale": True}
            return result
        if config.DEMO_FALLBACK:
            fallback = _fallback_payload(config.DEMO_FALLBACK_PATH)
            if fallback:
                fallback = dict(fallback)
                fallback.setdefault("warnings", []).append(
                    f"Using demo fallback: {error}"
                )
                fallback["cache"] = {"status": "demo-fallback", "stale": False}
                return fallback
        if isinstance(error, PropertyDataError):
            raise
        raise PropertyDataError(
            f"Subject property retrieval failed: {error}"
        ) from error

    area = _first(
        _dig(subject, "address", "postal_code"),
        ", ".join(
            part
            for part in (
                _dig(subject, "address", "city"),
                _dig(subject, "address", "state"),
            )
            if part
        ),
        address,
    )
    home_types = _home_type_filter(subject.get("property_type"))
    base: dict[str, Any] = {"location": area, "maxItems": 60}
    if home_types:
        base["homeTypes"] = home_types
    requests_by_section = {
        "active_comps": {**base, "operation": "buy", "sortBy": "newest"},
        "sold_comps": {
            **base,
            "operation": "sold",
            "sortBy": "recentlyChanged",
            "maxSoldDate": "6m",
        },
        "rental_comps": {**base, "operation": "rent", "sortBy": "newest"},
    }
    normalized_by_section: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []
    unavailable: list[str] = []
    operation_by_section = {
        "active_comps": "buy",
        "sold_comps": "sold",
        "rental_comps": "rent",
    }
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(actor_runner, config.ZILLOW_ACTOR, payload): section
            for section, payload in requests_by_section.items()
        }
        for future in as_completed(futures):
            section = futures[future]
            try:
                records = future.result()
                normalized_records = [
                    normalize_zillow_record(item, operation_by_section[section])
                    for item in records
                ]
                normalized_by_section[section] = select_comparables(
                    subject,
                    normalized_records,
                    operation=operation_by_section[section],
                    now=current,
                )
            except Exception as error:
                normalized_by_section[section] = []
                unavailable.append(section)
                warnings.append(f"{section} unavailable: {error}")

    # Widen recent sales only when the initial six-month window is sparse, and
    # disclose the wider evidence window in the response.
    if (
        "sold_comps" not in unavailable
        and len(normalized_by_section["sold_comps"]) < MIN_COMPS
    ):
        try:
            wider_raw = actor_runner(
                config.ZILLOW_ACTOR,
                {
                    **base,
                    "operation": "sold",
                    "sortBy": "recentlyChanged",
                    "maxSoldDate": "1y",
                },
            )
            wider = select_comparables(
                subject,
                [normalize_zillow_record(item, "sold") for item in wider_raw],
                operation="sold",
                now=current,
            )
            if len(wider) > len(normalized_by_section["sold_comps"]):
                normalized_by_section["sold_comps"] = wider
                warnings.append("sold_comps search widened from six to twelve months")
        except Exception as error:
            warnings.append(f"sold_comps twelve-month fallback unavailable: {error}")

    for section in requests_by_section:
        values = normalized_by_section.get(section, [])
        if section not in unavailable and len(values) < MIN_COMPS:
            warnings.append(f"{section} has only {len(values)} comparable records")

    retrieved_at = isoformat(current)
    expires_at = isoformat(
        current + timedelta(seconds=config.PROPERTY_CACHE_TTL_SECONDS)
    )
    active = normalized_by_section["active_comps"]
    sold = normalized_by_section["sold_comps"]
    rentals = normalized_by_section["rental_comps"]
    result = {
        "query": {"address": address.strip(), "normalized_address": normalized},
        "subject": subject,
        "active_comps": active,
        "sold_comps": sold,
        "rental_comps": rentals,
        "market_summary": build_market_summary(subject, active, sold, rentals),
        "sources": [
            {
                "provider": "live listing data",
                "actor_id": config.ZILLOW_ACTOR,
                "retrieved_at": retrieved_at,
            }
        ],
        "retrieved_at": retrieved_at,
        "expires_at": expires_at,
        "warnings": warnings,
        "unavailable_sections": sorted(unavailable),
        "cache": {"status": "miss", "stale": False},
    }
    cache_store.put(
        address,
        result,
        ttl_seconds=config.PROPERTY_CACHE_TTL_SECONDS,
        now=current,
    )
    return result
