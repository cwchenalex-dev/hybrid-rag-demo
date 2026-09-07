from app import indexer


def test_bm25_returns_relevant_section(monkeypatch):
    shipping = indexer.Section(
        id="shipping.md#standard-shipping",
        file="shipping.md",
        heading="Standard Shipping",
        heading_path=["Shipping", "Standard Shipping"],
        content="Standard shipping takes 3-5 business days.",
        tokens=indexer.tokenize(
            "Shipping Standard Shipping "
            "Standard shipping takes 3-5 business days."
        ),
    )

    refund = indexer.Section(
        id="refund.md#refund-policy",
        file="refund.md",
        heading="Refund Policy",
        heading_path=["Refund Policy"],
        content="Approved refunds are processed within 5-7 business days.",
        tokens=indexer.tokenize(
            "Refund Policy "
            "Approved refunds are processed within 5-7 business days."
        ),
    )

    monkeypatch.setattr(
        indexer,
        "sections",
        [shipping, refund],
    )

    indexer.rebuild_bm25_stats()

    results = indexer.search_bm25(
        "standard shipping",
        top_k=1,
        min_score=0.0,
    )

    assert results
    assert results[0][0].id == "shipping.md#standard-shipping"