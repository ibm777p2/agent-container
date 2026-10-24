import json

import pytest

from property_intelligence_rag import config
from property_intelligence_rag.schema import Property


def test_from_raw_maps_aliases_and_numbers():
    prop = Property.from_raw(
        {"MLS": "X1", "list_price": "$1,250,000", "Beds": "4", "zip": "78704", "pool": True,
         "avg_rent": "$2,450", "for_sale": "yes"},
        "fallback",
    )
    assert prop.id == "X1"
    assert prop.price == 1_250_000
    assert prop.bedrooms == 4
    assert prop.postal_code == "78704"
    assert prop.avg_rent_zip == 2450
    assert prop.for_sale is True
    assert prop.extra == {"pool": True}
    assert "Price: $1,250,000" in prop.to_document().page_content


@pytest.fixture
def index(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.json").write_text(json.dumps([
        {"id": "1", "city": "Austin", "price": 300000, "beds": 2, "type": "Condo", "for_sale": False},
        {"id": "2", "city": "Austin", "price": 700000, "beds": 4, "type": "Single Family", "for_sale": True},
        {"id": "3", "city": "Dallas", "price": 400000, "beds": 3, "type": "Single Family"},
    ]))
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "fake")
    monkeypatch.setattr(config, "INDEX_PATH", tmp_path / "index.json")

    from property_intelligence_rag.ingest import build_index
    from property_intelligence_rag.retriever import _get_store

    _get_store.cache_clear()
    assert build_index(raw, config.INDEX_PATH) == 3
    yield
    _get_store.cache_clear()


def test_search_properties_applies_filters(index):
    from property_intelligence_rag.retriever import search_properties

    out = search_properties.invoke({"query": "house", "city": "austin", "max_price": 500000})
    assert "[1]" in out and "[2]" not in out and "[3]" not in out

    out = search_properties.invoke({"query": "house", "min_bedrooms": 3, "property_type": "single"})
    assert "[1]" not in out and "[2]" in out and "[3]" in out

    out = search_properties.invoke({"query": "house", "for_sale": True})
    assert "[1]" not in out and "[2]" in out and "[3]" not in out


def test_screen_context_reaches_prompt_and_question_is_latest_turn():
    from langchain_core.messages import AIMessage, HumanMessage

    from property_intelligence_rag.graph import _question, _system_prompt

    state = {
        "messages": [HumanMessage("first"), AIMessage("answer"), HumanMessage("second")],
        "ag-ui": {"context": [{"description": "Screen", "value": {"zip": "94610"}}]},
    }
    assert _question(state) == "second"
    prompt = _system_prompt(state)
    assert "<screen>" in prompt and 'Screen:\n{"zip": "94610"}' in prompt
    assert "<screen>" not in _system_prompt({"messages": state["messages"]})
