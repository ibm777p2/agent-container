"""Tools for live address research and semantic listing search."""

import json
from functools import lru_cache

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from property_intelligence_rag import config
from property_intelligence_rag.ingest import build_index, get_embeddings
from property_intelligence_rag.property_data import (
    PropertyDataError,
    get_property_data as fetch_property_data,
)


@tool
def get_property_data(address: str, refresh: bool = False) -> str:
    """Retrieve current evidence for one residential street address.

    Returns subject property facts, similar active listings, recent sold
    comparables, rental comparables, data freshness, and a deterministic current
    comparable market snapshot. Set refresh only when the user explicitly asks
    for fresh data or a cached response is known to be stale. When the lookup fails,
    returns {"error": ...} instead.
    """
    try:
        result = fetch_property_data(address, refresh=refresh)
    except (PropertyDataError, ValueError) as error:
        # Hand the failure to the agent as the tool's answer so it can tell the user;
        # raised, it would abort the whole run and cut the chat's stream mid-reply.
        result = {"address": address, "error": f"Property lookup failed: {error}"}
    return json.dumps(result, ensure_ascii=False)


@lru_cache(maxsize=1)
def _get_store() -> InMemoryVectorStore:
    if not config.INDEX_PATH.exists():
        # First search on a fresh instance (e.g. a Vercel cold start): embed data/raw/ now.
        build_index(config.RAW_DIR, config.INDEX_PATH)
    return InMemoryVectorStore.load(str(config.INDEX_PATH), embedding=get_embeddings())


@tool
def search_properties(
    query: str,
    city: str | None = None,
    postal_code: str | None = None,
    max_price: float | None = None,
    min_price: float | None = None,
    min_bedrooms: float | None = None,
    property_type: str | None = None,
    for_sale: bool | None = None,
) -> str:
    """Search scraped property listings. Use `query` for the free-text part of the
    request (features, neighborhood, style) and the other arguments for hard
    constraints the user stated explicitly. Call it once per zip code or city
    when the user asks about several. Set `for_sale=True` only when the user
    wants homes available to buy. Each result includes the average monthly rent
    for its zip code."""

    def matches(doc: Document) -> bool:
        m = doc.metadata
        if city and (m.get("city") or "").lower() != city.lower():
            return False
        if postal_code and m.get("postal_code") != postal_code:
            return False
        if (
            property_type
            and property_type.lower() not in (m.get("property_type") or "").lower()
        ):
            return False
        price = m.get("price")
        if max_price is not None and (price is None or price > max_price):
            return False
        if min_price is not None and (price is None or price < min_price):
            return False
        if for_sale is not None and m.get("for_sale") is not for_sale:
            return False
        beds = m.get("bedrooms")
        if min_bedrooms is not None and (beds is None or beds < min_bedrooms):
            return False
        return True

    docs = _get_store().similarity_search(query, k=config.RETRIEVER_K, filter=matches)
    if not docs:
        return "No listings matched."
    return "\n\n---\n\n".join(
        f"[{d.metadata['id']}] {d.page_content}\nURL: {d.metadata.get('url', 'n/a')}"
        for d in docs
    )
