"""
Unit tests for multihaludet dataset downloading, provenance tracking, and schema parsing.
"""

from __future__ import annotations

import json
from pathlib import Path

from multihaludet.training.datasets import (
    HallucinationExample,
    load_halueval,
    load_triviaqa,
)
from multihaludet.training import train as train_mod
from multihaludet.training import evaluate as evaluate_mod


def test_hallucination_example_provenance_default():
    ex = HallucinationExample("q", "r", False)
    assert ex.provenance == ""


def test_load_halueval_parses_all_task_schemas(tmp_path: Path):
    qa_file = tmp_path / "qa.jsonl"
    qa_file.write_text(
        json.dumps({"question": "Q?", "right_answer": "A", "hallucinated_answer": "H", "provenance": "prov_qa"}) + "\n"
    )

    dialogue_file = tmp_path / "dialogue.jsonl"
    dialogue_file.write_text(
        json.dumps({
            "knowledge": "K",
            "dialogue_history": "D",
            "right_response": "R",
            "hallucinated_response": "HR",
            "derivation_method": "prov_dial",
        }) + "\n"
    )

    summarization_file = tmp_path / "summarization.jsonl"
    summarization_file.write_text(
        json.dumps({
            "document": "Doc",
            "right_summary": "S",
            "hallucinated_summary": "HS",
        }) + "\n"
    )

    qa_exs = list(load_halueval(str(qa_file), task="qa"))
    assert len(qa_exs) == 2
    assert qa_exs[0].provenance == "prov_qa"

    dial_exs = list(load_halueval(str(dialogue_file), task="dialogue"))
    assert len(dial_exs) == 2
    assert dial_exs[0].query == "K"
    assert dial_exs[0].provenance == "prov_dial"

    sum_exs = list(load_halueval(str(summarization_file), task="summarization"))
    assert len(sum_exs) == 2
    assert sum_exs[0].query == "Doc"
    assert sum_exs[0].provenance == "halueval_summarization_benchmark"


def test_load_triviaqa_preserves_provenance(tmp_path: Path):
    triv_file = tmp_path / "triv.jsonl"
    triv_file.write_text(
        json.dumps({
            "question": "What element is Au?",
            "model_response": "Gold",
            "is_hallucination": False,
            "derivation_method": "triviaqa_exact_match_v1",
            "provenance": "Explicit provenance details",
        }) + "\n"
    )

    exs = list(load_triviaqa(str(triv_file)))
    assert len(exs) == 1
    assert exs[0].provenance == "Explicit provenance details"
    assert exs[0].label is False


def test_max_samples_truncation(tmp_path: Path):
    qa_file = tmp_path / "qa.jsonl"
    lines = [
        json.dumps({"question": f"Q{i}", "right_answer": f"A{i}", "hallucinated_answer": f"H{i}"})
        for i in range(10)
    ]
    qa_file.write_text("\n".join(lines) + "\n")

    import argparse

    args = argparse.Namespace(
        halueval_qa=str(qa_file),
        halueval_dialogue=None,
        halueval_summarization=None,
        triviaqa=None,
        french=None,
        bangla=None,
        amharic=None,
        max_samples=5,
    )

    collected = train_mod._collect_examples(args)
    assert len(collected) == 5
