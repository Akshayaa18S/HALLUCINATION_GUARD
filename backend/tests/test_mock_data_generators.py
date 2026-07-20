import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.utils.mock_data import MockDataGenerator


def test_mock_data_generator_creates_valid_job_id():
    job_id = MockDataGenerator.generate_job_id()
    assert isinstance(job_id, str)
    assert len(job_id) > 0


def test_mock_data_generator_creates_input_text():
    text = MockDataGenerator.generate_input_text()
    assert isinstance(text, str)
    assert len(text) > 0


def test_mock_data_generator_creates_hidden_states():
    states = MockDataGenerator.generate_hidden_states()
    assert "token_embeddings" in states
    assert "attention_maps" in states
    assert isinstance(states["token_embeddings"], list)


def test_mock_data_generator_creates_extracted_features():
    features = MockDataGenerator.generate_extracted_features()
    assert "multi_scale_attention" in features
    assert isinstance(features["multi_scale_attention"], list)


def test_mock_data_generator_creates_hallucination_result():
    result = MockDataGenerator.generate_hallucination_result(is_hallucination=True)
    assert result["prediction"] is True
    assert "confidence" in result
    assert 0 <= result["probability"] <= 1


def test_mock_data_generator_creates_retrieved_evidence():
    evidence = MockDataGenerator.generate_retrieved_evidence()
    assert "sources" in evidence
    assert "supporting_documents" in evidence


def test_mock_data_generator_creates_explanation():
    explanation = MockDataGenerator.generate_explanation()
    assert "shap_values" in explanation
    assert "important_tokens" in explanation
    assert "explanation_text" in explanation


def test_mock_data_generator_creates_full_pipeline_state():
    state = MockDataGenerator.generate_full_pipeline_state(job_id="test-job-123")
    assert state["job_id"] == "test-job-123"
    assert "input_text" in state
    assert "generated_response" in state
    assert "hallucination_result" in state


def test_mock_data_generator_creates_stage_events():
    events = MockDataGenerator.generate_stage_events(job_id="test-job-456", num_stages=8)
    assert len(events) == 8
    assert all(e["job_id"] == "test-job-456" for e in events)
    assert events[0]["stage"] == 1
    assert events[-1]["stage"] == 8


def test_mock_data_generator_creates_batch_results():
    results = MockDataGenerator.generate_batch_analysis_results(num_jobs=5)
    assert len(results) == 5
    assert all("job_id" in r for r in results)
    assert all("hallucination" in r for r in results)
    assert all("confidence" in r for r in results)
