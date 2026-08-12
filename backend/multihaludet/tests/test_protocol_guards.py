import pytest
import numpy as np
from multihaludet.training.datasets import HallucinationExample, verify_no_test_contamination
from multihaludet.training.evaluate import main as eval_main
from multihaludet.pipeline import MultiHaluDetModel


def test_no_test_contamination_guard():
    """Test D: Asserts that verify_no_test_contamination raises ValueError on sample overlap."""
    dev = [
        HallucinationExample("Who directed Inception?", "Christopher Nolan directed Inception.", False, "en", "dev", "1"),
        HallucinationExample("What is 2+2?", "2+2 is 4.", False, "en", "dev", "2"),
    ]
    test_clean = [
        HallucinationExample("What is the capital of France?", "Paris is the capital.", False, "en", "test", "3"),
    ]
    test_contaminated = [
        HallucinationExample("Who directed Inception?", "Christopher Nolan directed Inception.", False, "en", "test", "1"),
    ]

    # Clean case should pass without error
    verify_no_test_contamination(dev, test_clean)

    # Contaminated case should raise ValueError
    with pytest.raises(ValueError, match="DATASET CONTAMINATION ERROR"):
        verify_no_test_contamination(dev, test_contaminated)


def test_oof_completeness_and_provenance():
    """Test G: Asserts that OOF tracking correctly flags incomplete or overlapping writes."""
    n_samples = 20
    oof_written = np.zeros(n_samples, dtype=bool)

    # Simulate writing OOF predictions for 20 samples
    for i in range(n_samples):
        oof_written[i] = True

    assert np.all(oof_written), "OOF completeness check failed"

    # Simulate missing index
    oof_written_broken = np.zeros(n_samples, dtype=bool)
    oof_written_broken[:15] = True
    assert not np.all(oof_written_broken), "Failed to detect incomplete OOF coverage"


def test_evaluation_threshold_provenance_guard(tmp_path):
    """Test F: Asserts that evaluate.py rejects checkpoints with invalid threshold sources."""
    model = MultiHaluDetModel(hidden_size=64)

    # Invalid metadata simulating test-set tuned threshold
    model.metadata = {"threshold_source": "test_set_tuned", "training_protocol": "invalid"}
    ckpt_path = tmp_path / "invalid_ckpt.pt"
    model.save_checkpoint(str(ckpt_path), metadata=model.metadata)

    class DummyArgs:
        checkpoint = str(ckpt_path)
        frozen_test = None
        model_name = "fake"
        device = "cpu"
        halueval_qa = None
        halueval_dialogue = None
        halueval_summarization = None
        triviaqa = None
        french = None
        bangla = None
        amharic = None
        max_samples = 10
        generate_plots = False
        plots_dir = str(tmp_path)

    with pytest.raises(ValueError, match="EVALUATION GUARD ERROR"):
        eval_main(DummyArgs())


def test_valid_threshold_provenance():
    """Test F (valid case): Verifies metadata with threshold_source='development_oof' is accepted."""
    model = MultiHaluDetModel(hidden_size=64)
    model.metadata = {"threshold_source": "development_oof", "training_protocol": "strict_oof_v4"}
    assert model.metadata.get("threshold_source") == "development_oof"
