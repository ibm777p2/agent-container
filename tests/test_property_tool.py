import json


def test_property_tool_returns_json(monkeypatch):
    from property_intelligence_rag import retriever

    monkeypatch.setattr(
        retriever,
        "fetch_property_data",
        lambda address, refresh=False: {"address": address, "refresh": refresh},
    )
    value = retriever.get_property_data.invoke(
        {"address": "123 Main St", "refresh": True}
    )
    assert json.loads(value) == {"address": "123 Main St", "refresh": True}


def test_property_tool_reports_lookup_failure(monkeypatch):
    from property_intelligence_rag import retriever
    from property_intelligence_rag.property_data import PropertyDataError

    def fail(address, refresh=False):
        raise PropertyDataError("Subject property retrieval failed: APIFY_TOKEN is not configured")

    monkeypatch.setattr(retriever, "fetch_property_data", fail)
    value = json.loads(retriever.get_property_data.invoke({"address": "1 Nowhere St"}))
    assert value["address"] == "1 Nowhere St"
    assert "APIFY_TOKEN is not configured" in value["error"]
