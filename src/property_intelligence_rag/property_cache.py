"""Tiny SQLite-backed cache for normalized property-data responses."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS property_cache (
    cache_key TEXT PRIMARY KEY,
    address TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
)
"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def normalize_address(address: str) -> str:
    """Normalize an address for cache identity, not for postal validation."""
    if not isinstance(address, str) or not address.strip():
        raise ValueError("address is required")
    normalized = address.casefold().strip()
    normalized = re.sub(r"[^\w\s#-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def address_cache_key(address: str) -> str:
    value = f"property-data:v1:{normalize_address(address)}".encode()
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    payload: dict[str, Any]
    retrieved_at: datetime
    expires_at: datetime

    @property
    def is_stale(self) -> bool:
        return self.expires_at <= utc_now()


class PropertyCache:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(SCHEMA)

    def get(self, address: str, *, allow_stale: bool = False) -> CacheEntry | None:
        key = address_cache_key(address)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json, retrieved_at, expires_at FROM property_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        try:
            entry = CacheEntry(
                payload=json.loads(row["payload_json"]),
                retrieved_at=parse_timestamp(row["retrieved_at"]),
                expires_at=parse_timestamp(row["expires_at"]),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            self.delete(address)
            return None
        return entry if allow_stale or not entry.is_stale else None

    def put(
        self,
        address: str,
        payload: dict[str, Any],
        *,
        ttl_seconds: int,
        now: datetime | None = None,
    ) -> CacheEntry:
        retrieved_at = (now or utc_now()).astimezone(timezone.utc)
        expires_at = retrieved_at + timedelta(seconds=ttl_seconds)
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO property_cache (cache_key, address, payload_json, retrieved_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    address = excluded.address,
                    payload_json = excluded.payload_json,
                    retrieved_at = excluded.retrieved_at,
                    expires_at = excluded.expires_at
                """,
                (
                    address_cache_key(address),
                    address.strip(),
                    encoded,
                    isoformat(retrieved_at),
                    isoformat(expires_at),
                ),
            )
        return CacheEntry(payload, retrieved_at, expires_at)

    def delete(self, address: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM property_cache WHERE cache_key = ?",
                (address_cache_key(address),),
            )
