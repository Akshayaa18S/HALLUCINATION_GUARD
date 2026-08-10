import pytest
from pipeline.stages.claim_extraction import RuleBasedClaimExtractor, decompose_sentence_into_claims
from hallucination.verification import _check_negative_claim_support
from config.settings import settings


def test_decompose_sentence_into_claims():
    compound_sentence = "They are a South Korean boy group formed by JYP Entertainment in 2013."
    claims = decompose_sentence_into_claims(compound_sentence)

    assert len(claims) == 3
    assert claims[0] == "They are a South Korean boy group."
    assert claims[1] == "They were formed by JYP Entertainment."
    assert claims[2] == "They were formed in 2013."


def test_negative_claim_support():
    negative_claim = "No, BTS is not from India."
    evidence_texts = [
        "BTS, also known as the Bangtan Boys, is a South Korean boy band formed in 2010."
    ]

    is_supported = _check_negative_claim_support(negative_claim, evidence_texts)
    assert is_supported is True


def test_rule_based_extractor():
    extractor = RuleBasedClaimExtractor()
    text = "No, BTS is not from India. They are a South Korean boy group formed by JYP Entertainment in 2013."
    extracted = extractor.extract(text)

    assert len(extracted) >= 3
    assert any("not from India" in c for c in extracted)
    assert any("South Korean boy group" in c for c in extracted)


def test_confidence_weights_setting():
    weights = getattr(settings, "confidence_weights", None)
    assert weights is not None
    assert "ensemble" in weights
    assert "margin" in weights
    assert "evidence" in weights
    assert sum(weights.values()) == pytest.approx(1.0)


def test_expanded_verification_regression_suite():
    # 1. Alias & Synonym Support
    alias_claim = "BTS is also known as Bangtan Boys."
    evidence = ["BTS, also known as the Bangtan Boys, is a South Korean boy band formed in 2010 by Big Hit Entertainment."]
    assert any("bangtan boys" in ev.lower() for ev in evidence)

    # 2. Company Contradiction
    company_claim = "BTS was formed by SM Entertainment."
    assert "sm entertainment" in company_claim.lower() and "big hit" in evidence[0].lower()

    # 3. Predicate-Aware Temporal Contradiction (formed 2013 vs formed 2010)
    formed_claim = "They were formed in 2013."
    assert "formed in 2013" in formed_claim.lower() and "formed in 2010" in evidence[0].lower()
