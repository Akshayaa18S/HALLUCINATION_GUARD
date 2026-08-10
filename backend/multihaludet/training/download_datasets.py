"""
Dataset Downloader and Labeled Split Generator for MultiHaluDet.

Downloads:
1. HaluEval QA split -> multihaludet/data/halueval_qa.jsonl
2. HaluEval Dialogue split -> multihaludet/data/halueval_dialogue.jsonl
3. HaluEval Summarization split -> multihaludet/data/halueval_summarization.jsonl
4. Reproducibly derived labeled TriviaQA split -> multihaludet/data/triviaqa_labeled.jsonl
   (Includes provenance fields describing derivation method & source dataset)
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger("hallucination_guard.multihaludet.download_datasets")

HALUEVAL_URLS = {
    "qa": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json",
    "dialogue": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/dialogue_data.json",
    "summarization": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/summarization_data.json",
}

TRIVIAQA_HF_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=mandarjoshi/trivia_qa&config=rc.nocontext&split=validation"
)


def download_halueval_splits(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}

    for task, url in HALUEVAL_URLS.items():
        out_file = data_dir / f"halueval_{task}.jsonl"
        logger.info("Downloading HaluEval '%s' split from %s...", task, url)

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode("utf-8")

        # RUCAIBox/HaluEval files are line-delimited JSON. Write directly to jsonl out_file.
        out_file.write_text(content, encoding="utf-8")
        lines = [line for line in content.splitlines() if line.strip()]
        logger.info("Saved %d %s examples to %s", len(lines), task, out_file)
        downloaded[task] = out_file

    return downloaded


def fetch_triviaqa_hf_rows(max_rows: int = 500) -> list[dict[str, Any]]:
    logger.info("Fetching TriviaQA (mandarjoshi/trivia_qa) validation set from HuggingFace API...")
    rows: list[dict[str, Any]] = []
    offset = 0
    length = 100

    while len(rows) < max_rows:
        fetch_len = min(length, max_rows - len(rows))
        url = f"{TRIVIAQA_HF_URL}&offset={offset}&length={fetch_len}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                batch = [item["row"] for item in data.get("rows", [])]
                if not batch:
                    break
                rows.extend(batch)
                offset += len(batch)
        except Exception as e:
            logger.warning("Error fetching TriviaQA batch at offset %d: %s", offset, e)
            break

    logger.info("Fetched %d raw TriviaQA items from HuggingFace.", len(rows))
    return rows


def generate_triviaqa_labeled_split(data_dir: Path, max_items: int = 500) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "triviaqa_labeled.jsonl"

    raw_items = fetch_triviaqa_hf_rows(max_rows=max_items)
    distractors = [
        "Mars", "Albert Einstein", "1999", "Tokyo", "Python",
        "Jupiter", "Leonardo da Vinci", "1776", "Paris", "Gold",
    ]

    labeled_rows: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        question = item.get("question", "")
        answer_obj = item.get("answer", {}) or {}
        gold_value = answer_obj.get("value") or answer_obj.get("Value") or ""
        aliases = answer_obj.get("aliases") or answer_obj.get("Aliases") or []
        if not question or not gold_value:
            continue

        # 1. Faithful model response (includes gold answer value)
        faithful_response = f"The answer to '{question}' is {gold_value}."
        labeled_rows.append({
            "question": question,
            "answer": {"value": gold_value, "aliases": aliases},
            "model_response": faithful_response,
            "is_hallucination": False,
            "derivation_method": "triviaqa_hf_rc_nocontext_alias_matching_v1",
            "source_dataset": "mandarjoshi/trivia_qa",
            "provenance": (
                "Reproducibly derived via TriviaQA gold value insertion. "
                "Label is_hallucination=False (faithful exact answer match)."
            ),
        })

        # 2. Hallucinated model response (uses distractor instead of gold answer)
        distractor = distractors[idx % len(distractors)]
        if distractor.lower() in gold_value.lower():
            distractor = "Unknown Entity 404"
        hallucinated_response = f"The answer to '{question}' is {distractor}."
        labeled_rows.append({
            "question": question,
            "answer": {"value": gold_value, "aliases": aliases},
            "model_response": hallucinated_response,
            "is_hallucination": True,
            "derivation_method": "triviaqa_hf_rc_nocontext_alias_matching_v1",
            "source_dataset": "mandarjoshi/trivia_qa",
            "provenance": (
                "Reproducibly derived via TriviaQA distractor insertion. "
                "Label is_hallucination=True (hallucinated non-matching response)."
            ),
        })

    with out_file.open("w", encoding="utf-8") as f:
        for row in labeled_rows:
            f.write(json.dumps(row) + "\n")

    logger.info("Saved %d reproducibly derived TriviaQA labeled examples to %s", len(labeled_rows), out_file)
    return out_file


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./multihaludet/data", help="Target directory for downloaded datasets")
    parser.add_argument("--max-triviaqa", type=int, default=500, help="Number of TriviaQA questions to fetch and label")
    args = parser.parse_args()

    target_dir = Path(args.data_dir)
    download_halueval_splits(target_dir)
    generate_triviaqa_labeled_split(target_dir, max_items=args.max_triviaqa)
    logger.info("All datasets successfully downloaded and saved to %s", target_dir)


if __name__ == "__main__":
    main()
