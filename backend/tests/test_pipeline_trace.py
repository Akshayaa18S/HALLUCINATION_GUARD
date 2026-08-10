import pytest
from analysis.pipeline_trace import PipelineTracer
from analysis.error_analysis import ErrorAnalyzer


def test_pipeline_tracer():
    query = "Who is the CEO of Apple?"
    response_analysis = [
        {"statement": "Tim Cook is CEO", "status": "Supported", "confidence": 0.92}
    ]
    retrieved_evidence = [
        {
            "title": "Apple Inc.",
            "entity_type": "Organization",
            "entity_validation": "Passed",
            "entity_similarity": 0.98,
            "retrieval_attempt": 1,
        }
    ]

    trace_dict = PipelineTracer.build_trace(query, response_analysis, retrieved_evidence)
    assert "pipeline_trace" in trace_dict
    trace = trace_dict["pipeline_trace"]

    assert trace["expected_type"] == "ORGANIZATION"
    assert trace["claims"] == 1
    assert trace["supported"] == 1
    assert trace["contradicted"] == 0
    assert trace["retrieval_retry"] is False
    assert len(trace["retrieval_candidates"]) >= 1
    assert trace["retrieval_candidates"][-1]["accepted"] is True


def test_error_analyzer():
    analyzer = ErrorAnalyzer()
    samples = [
        {"query": "Who is CEO of Apple?", "label": 0},
        {"query": "Unknown entity search", "label": 1},
    ]
    preds = [
        {
            "prediction": "Factual",
            "retrieved_evidence": [{"entity_type": "Organization", "entity_validation": "Passed"}],
            "response_analysis": [{"status": "Supported"}],
        },
        {
            "prediction": "Factual",
            "retrieved_evidence": [{"entity_type": "Fruit/Plant", "entity_validation": "Failed"}],
            "response_analysis": [],
        },
    ]

    res = analyzer.analyze_predictions(samples, preds)
    assert "error_summary" in res
    assert res["error_summary"]["entity_ambiguity"] == 1
