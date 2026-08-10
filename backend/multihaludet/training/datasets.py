"""
Dataset loaders for training/evaluating the MultiHaluDet branch against
the base paper's reported experiments: HaluEval + TriviaQA, plus
multilingual generalization (French, Bangla, Amharic).

Nothing here downloads or trains anything automatically - this sandbox
has no network access and no GPU, so these are working *interfaces* you
run from an environment that has both. Point `*_path` at a local copy of
each dataset (see each loader's docstring for the expected file shape).

Every loader yields `HallucinationExample`, the common shape
train.py/evaluate.py consume regardless of source dataset or language.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

Language = Literal["en", "fr", "bn", "am"]


@dataclass
class HallucinationExample:
    query: str
    response: str  # the (possibly hallucinated) response to classify
    label: bool  # True = hallucination, False = faithful/correct
    language: Language = "en"
    source: str = ""
    provenance: str = ""  # metadata describing how the label/example was derived


def load_halueval(path: str, task: str = "qa") -> Iterator[HallucinationExample]:
    """HaluEval (Li et al., 2023) qa/dialogue/summarization JSONL splits."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"HaluEval file not found at {path}. Download the '{task}' "
            "split from https://github.com/RUCAIBox/HaluEval and point "
            "this loader at the extracted JSONL file."
        )
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = (
                row.get("question")
                or row.get("knowledge")
                or row.get("document")
                or row.get("dialogue_history")
                or ""
            )
            right = (
                row.get("right_answer")
                or row.get("right_response")
                or row.get("right_summary")
            )
            hallucinated = (
                row.get("hallucinated_answer")
                or row.get("hallucinated_response")
                or row.get("hallucinated_summary")
            )
            prov = row.get("provenance") or row.get("derivation_method") or f"halueval_{task}_benchmark"

            if right is not None:
                yield HallucinationExample(question, right, False, "en", f"halueval_{task}", prov)
            if hallucinated is not None:
                yield HallucinationExample(question, hallucinated, True, "en", f"halueval_{task}", prov)


def load_triviaqa(path: str) -> Iterator[HallucinationExample]:
    """TriviaQA (Joshi et al., 2017). Expected JSONL with
    {"question": "...", "answer": {"value": "..."}, "model_response": "...",
    "is_hallucination": true/false} - expected to carry label derivation provenance metadata.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"TriviaQA labeled-response file not found at {path}. See "
            "http://nlp.cs.washington.edu/triviaqa/ for the base dataset; "
            "generate + label model responses before using this loader."
        )
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prov = (
                row.get("provenance")
                or row.get("derivation_method")
                or "triviaqa_derived_dataset"
            )
            yield HallucinationExample(
                row["question"],
                row["model_response"],
                bool(row["is_hallucination"]),
                "en",
                "triviaqa",
                prov,
            )


_MULTILINGUAL_LANGS: dict[Language, str] = {"fr": "French", "bn": "Bangla", "am": "Amharic"}


def load_multilingual(path: str, language: Language) -> Iterator[HallucinationExample]:
    """Cross-lingual generalization split (French / Bangla / Amharic)."""
    if language not in _MULTILINGUAL_LANGS:
        raise ValueError(f"Unsupported language '{language}'; expected one of {list(_MULTILINGUAL_LANGS)}")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{_MULTILINGUAL_LANGS[language]} evaluation file not found at {path}. "
            "Populate it with {query, response, label} JSONL rows in that language."
        )
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prov = row.get("provenance") or f"multilingual_{language}_benchmark"
            yield HallucinationExample(row["query"], row["response"], bool(row["label"]), language, f"multilingual_{language}", prov)


def sample_representative_subset(
    examples: list[HallucinationExample], max_samples: int | None, seed: int = 42
) -> list[HallucinationExample]:
    """Shuffles deterministically with seed and performs stratified sampling across
    (source, label) pairs so max-samples N returns a representative mixture."""
    if max_samples is None or max_samples <= 0 or len(examples) <= max_samples:
        rng = random.Random(seed)
        shuffled = list(examples)
        rng.shuffle(shuffled)
        return shuffled

    strata: dict[tuple[str, bool], list[HallucinationExample]] = defaultdict(list)
    for ex in examples:
        strata[(ex.source, ex.label)].append(ex)

    rng = random.Random(seed)
    for key in strata:
        rng.shuffle(strata[key])

    total_n = len(examples)
    selected: list[HallucinationExample] = []

    for key, items in strata.items():
        count = int(round((len(items) / total_n) * max_samples))
        count = max(0, min(count, len(items)))
        selected.extend(items[:count])

    if len(selected) < max_samples:
        selected_set = set(id(ex) for ex in selected)
        remaining = [ex for ex in examples if id(ex) not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_samples - len(selected)])
    elif len(selected) > max_samples:
        rng.shuffle(selected)
        selected = selected[:max_samples]

    rng.shuffle(selected)
    return selected


def get_dataset_diagnostics(examples: list[HallucinationExample]) -> dict[str, Any]:
    """Returns total, positive/negative count, positive percentage, and breakdown per source."""
    total = len(examples)
    if total == 0:
        return {"total": 0, "positives": 0, "negatives": 0, "positive_pct": 0.0, "sources": {}}

    positives = sum(1 for ex in examples if ex.label)
    negatives = total - positives
    pos_pct = round((positives / total) * 100, 2)

    sources: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "positives": 0, "negatives": 0, "positive_pct": 0.0})
    for ex in examples:
        s = ex.source or "unknown"
        sources[s]["total"] += 1
        if ex.label:
            sources[s]["positives"] += 1
        else:
            sources[s]["negatives"] += 1

    for s, data in sources.items():
        if data["total"] > 0:
            data["positive_pct"] = round((data["positives"] / data["total"]) * 100, 2)

    return {
        "total": total,
        "positives": positives,
        "negatives": negatives,
        "positive_pct": pos_pct,
        "sources": dict(sources),
    }

