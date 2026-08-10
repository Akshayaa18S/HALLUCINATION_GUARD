import asyncio
import pytest
from retrieval.wikipedia_retriever import WikipediaRetriever, compute_entity_similarity, classify_page_entity_type, infer_expected_entity_type
from knowledge_base.relation_registry import RelationRegistry


def test_entity_similarity_scoring():
    sim_exact = compute_entity_similarity("Apple", "Apple Inc.", "Apple Inc. is an American multinational corporation...", query_text="Who is the CEO of Apple?")
    assert sim_exact >= 0.90

    sim_mismatch = compute_entity_similarity("Apple", "Chief executive officer", "A chief executive officer (CEO) is the highest-ranking executive...", query_text="Who is the CEO of Apple?")
    assert sim_mismatch < 0.30


def test_entity_type_disambiguation():
    fruit_page_sim = compute_entity_similarity("Apple", "Apple", "An apple is the round, edible fruit of an apple tree (Malus spp.)...", query_text="Who is the CEO of Apple?")
    assert fruit_page_sim < 0.30

    company_page_sim = compute_entity_similarity("Apple", "Apple Inc.", "Apple Inc. is an American multinational technology company...", query_text="Who is the CEO of Apple?")
    assert company_page_sim >= 0.90


def test_semantic_entity_extraction_and_relation_registry():
    retriever = WikipediaRetriever()

    # Query 1: Capital of Germany (capital_of -> Subject: Germany)
    rel1 = RelationRegistry.extract_relation("The capital of Germany is Munich.")
    assert rel1 is not None
    assert rel1.relation == "capital_of"
    assert rel1.subject == "Germany"
    assert rel1.object == "Munich"

    terms1 = retriever._extract_entity_terms("The capital of Germany is Munich.")
    assert terms1[0] == "Germany"

    # Query 2: CEO of Apple (ceo_of -> Subject: Apple)
    rel2 = RelationRegistry.extract_relation("The CEO of Apple is Tim Cook.")
    assert rel2 is not None
    assert rel2.relation == "ceo_of"
    assert rel2.subject == "Apple"
    assert rel2.object == "Tim Cook"

    terms2 = retriever._extract_entity_terms("The CEO of Apple is Tim Cook.")
    assert terms2[0] == "Apple"


@pytest.mark.anyio
async def test_relational_regression_suite():
    retriever = WikipediaRetriever()

    # Case 1: Capital of Germany
    res1 = await retriever.retrieve("The capital of Germany is Munich.")
    assert len(res1) >= 1
    assert res1[0]["title"] in ("Germany", "Berlin")
    assert res1[0]["entity_type"] in ("Country", "City")

    # Case 2: CEO of Apple
    res2 = await retriever.retrieve("Who is the CEO of Apple?")
    assert len(res2) >= 1
    assert "Apple" in res2[0]["title"]
    assert res2[0]["entity_type"] == "Organization"

    # Case 3: Headquarters of Amazon
    res3 = await retriever.retrieve("Amazon is headquartered in Seattle.")
    assert len(res3) >= 1
    assert "Amazon" in res3[0]["title"]
    assert res3[0]["entity_type"] == "Organization"

    # Case 4: Capital of France
    res4 = await retriever.retrieve("Paris is the capital of France.")
    assert len(res4) >= 1
    assert res4[0]["title"] in ("France", "Paris")
    assert res4[0]["entity_type"] in ("Country", "City")

    # Case 5: Founder of Tesla
    res5 = await retriever.retrieve("Elon Musk founded Tesla.")
    assert len(res5) >= 1
    assert "Tesla" in res5[0]["title"] or "Elon Musk" in res5[0]["title"]
