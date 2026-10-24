from property_intelligence_rag.property_cache import PropertyCache, address_cache_key


def test_address_variants_share_a_cache_key():
    assert address_cache_key("123 Main St., Austin, TX") == address_cache_key(
        "  123 main st Austin TX  "
    )


def test_cache_fresh_expired_and_delete(tmp_path):
    cache = PropertyCache(tmp_path / "cache.sqlite3")
    cache.put("123 Main St", {"ok": True}, ttl_seconds=60)
    assert cache.get("123 main st").payload == {"ok": True}

    cache.put("456 Main St", {"old": True}, ttl_seconds=-1)
    assert cache.get("456 Main St") is None
    assert cache.get("456 Main St", allow_stale=True).is_stale

    cache.delete("123 Main St")
    assert cache.get("123 Main St", allow_stale=True) is None
