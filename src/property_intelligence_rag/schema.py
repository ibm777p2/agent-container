"""Normalize scraped listing records into a common shape and into Documents.

Scrapers rarely agree on field names, so `FIELD_ALIASES` maps whatever the
source emits onto our canonical keys. Add aliases here as new sources show up.
"""

import re
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "listing_id", "mls", "mls_id", "property_id"),
    "title": ("title", "name", "headline"),
    "address": ("address", "street_address", "full_address"),
    "city": ("city", "municipality", "locality"),
    "state": ("state", "region", "province"),
    "postal_code": ("postal_code", "zip", "zipcode", "zip_code"),
    "price": ("price", "list_price", "asking_price", "rent"),
    "bedrooms": ("bedrooms", "beds", "bed"),
    "bathrooms": ("bathrooms", "baths", "bath"),
    "sqft": ("sqft", "square_feet", "living_area", "area"),
    "property_type": ("property_type", "type", "home_type"),
    "status": ("status", "listing_status"),
    "description": ("description", "remarks", "details", "body"),
    "url": ("url", "link", "listing_url", "source_url"),
    "avg_rent_zip": ("avg_rent_zip", "avg_rent", "average_rent", "zip_avg_rent"),
    "for_sale": ("for_sale", "available_for_sale", "is_for_sale"),
}


class Property(BaseModel):
    id: str
    title: str | None = None
    address: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    price: float | None = None
    bedrooms: float | None = None
    bathrooms: float | None = None
    sqft: float | None = None
    property_type: str | None = None
    status: str | None = None
    description: str | None = None
    url: str | None = None
    avg_rent_zip: float | None = None  # average monthly rent in this listing's zip code
    for_sale: bool | None = None
    extra: dict[str, Any] = {}

    @classmethod
    def from_raw(cls, raw: dict[str, Any], fallback_id: str) -> "Property":
        lowered = {k.lower().strip(): v for k, v in raw.items()}
        values: dict[str, Any] = {}
        used: set[str] = set()
        for field, aliases in FIELD_ALIASES.items():
            for alias in aliases:
                if lowered.get(alias) not in (None, ""):
                    values[field] = lowered[alias]
                    used.add(alias)
                    break
        for field in ("price", "bedrooms", "bathrooms", "sqft", "avg_rent_zip"):
            if field in values:
                values[field] = _to_number(values[field])
        if "for_sale" in values:
            values["for_sale"] = _to_bool(values["for_sale"])
        values["id"] = str(values.get("id") or fallback_id)
        values["extra"] = {k: v for k, v in lowered.items() if k not in used}
        return cls(**values)

    def to_document(self) -> Document:
        """One listing -> one Document, so retrieval never splits a listing apart."""
        lines = [
            self.title or self.address or f"Listing {self.id}",
            f"Address: {', '.join(p for p in (self.address, self.city, self.state, self.postal_code) if p)}",
            f"{'Monthly rent' if self.for_sale is False else 'Price'}: {_fmt_money(self.price)}",
            f"Beds/Baths: {_fmt(self.bedrooms)} bd / {_fmt(self.bathrooms)} ba",
            f"Size: {_fmt(self.sqft)} sqft",
            f"Type: {self.property_type or 'unknown'}  Status: {self.status or 'unknown'}",
            f"For sale: {_fmt_bool(self.for_sale)}",
            f"Average rent in zip {self.postal_code or '?'}: {_fmt_money(self.avg_rent_zip)}/mo",
        ]
        if self.description:
            lines.append(f"Description: {self.description}")
        for key, value in self.extra.items():
            lines.append(f"{key}: {value}")
        metadata = self.model_dump(exclude={"description", "extra"}, exclude_none=True)
        return Document(page_content="\n".join(lines), metadata=metadata)


def _to_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^\d.]", "", str(value))
    return float(cleaned) if cleaned else None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    return None


def _fmt_bool(value: bool | None) -> str:
    return "unknown" if value is None else ("yes" if value else "no")


def _fmt(value: float | None) -> str:
    return "?" if value is None else f"{value:g}"


def _fmt_money(value: float | None) -> str:
    return "unknown" if value is None else f"${value:,.0f}"
