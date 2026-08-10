import time
import pytest
from retrieval.evidence_selector import EvidenceSelector


def test_sentence_cleaning_and_splitting():
    raw_doc = (
        "Apple Inc. is a technology company [1]. "
        "Tim Cook has served as CEO of Apple since 2011 [citation needed]. "
        "It is headquartered in Cupertino, California [2]."
    )

    sentences = EvidenceSelector.clean_text_into_sentences(raw_doc)
    assert len(sentences) == 3
    assert "[1]" not in sentences[0]
    assert "[citation needed]" not in sentences[1]
    assert "Tim Cook" in sentences[1]


def test_s_rank_scoring_and_support_contradiction_scores():
    claim = "The CEO of Apple is Tim Cook."
    sentence_direct = "He was succeeded as CEO by Tim Cook in August 2011."
    sentence_indirect = "The company designs consumer electronics and software."
    sentence_contradict = "The CEO of Apple is not Tim Cook."

    score_direct = EvidenceSelector.compute_sentence_score(sentence_direct, claim, target_entities=["Apple", "Tim Cook"])
    score_indirect = EvidenceSelector.compute_sentence_score(sentence_indirect, claim, target_entities=["Apple", "Tim Cook"])
    score_contra = EvidenceSelector.compute_sentence_score(sentence_contradict, claim, target_entities=["Apple", "Tim Cook"])

    assert score_direct.score > score_indirect.score
    assert score_direct.support_score > 0.40
    assert score_contra.contradiction_score > 0.50


def test_adaptive_threshold_weak_sentence_filtering():
    claim = "The CEO of Apple is Tim Cook."
    doc_text = (
        "Jackson, Apple's vice president of Environment who reports directly to CEO, Tim Cook. "
        "He was succeeded as CEO by Tim Cook in August 2011. "
        "In 1996, Spindler was replaced as CEO by Gil Amelio, who was hired for his reputation."
    )

    selector = EvidenceSelector()
    res = selector.select_best_sentences(claim, doc_text, top_k=3, target_entities=["Apple", "Tim Cook"])

    assert len(res["supporting_sentences"]) >= 1
    assert "Tim Cook" in res["best_excerpt"]
    assert res["evidence_strength"] >= 0.85
