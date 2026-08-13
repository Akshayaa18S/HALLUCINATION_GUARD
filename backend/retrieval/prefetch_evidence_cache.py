"""
Pre-fetch and persist local evidence disk cache for benchmark and development queries.
"""

import sys
import logging
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from retrieval.evidence_cache import get_evidence_cache
from multihaludet.training.datasets import load_frozen_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hallucination_guard.prefetch_cache")


def prefetch_benchmark_cache():
    cache = get_evidence_cache()
    test_path = backend_dir / "data" / "halueval_fever_benchmark_500.csv"
    if not test_path.exists():
        test_path = backend_dir / "multihaludet" / "data" / "halueval_benchmark_500.jsonl"

    logger.info("Loading benchmark queries from '%s'...", test_path)
    examples = load_frozen_benchmark(str(test_path))
    logger.info("Pre-fetching evidence for %d benchmark queries...", len(examples))

    hits = 0
    fetched = 0
    for i, ex in enumerate(examples):
        cached = cache.get(ex.query)
        if cached is not None:
            hits += 1
        else:
            snips = cache.get_or_fetch(ex.query, top_k=3)
            fetched += 1
        if (i + 1) % 50 == 0 or (i + 1) == len(examples):
            logger.info("Progress: %d/%d queries processed (Cache Hits: %d, Fetched: %d)", i + 1, len(examples), hits, fetched)

    cache.save()
    logger.info("Successfully saved evidence cache to '%s'", cache.cache_file)

    # Verification: Read disk file directly and confirm 100% coverage
    import json
    with open(cache.cache_file, "r", encoding="utf-8") as f:
        disk_data = json.load(f)

    missing = [ex.query for ex in examples if ex.query.strip().lower() not in disk_data]
    if missing:
        logger.error("CACHE VERIFICATION FAILED: %d/%d benchmark queries missing from disk cache!", len(missing), len(examples))
        raise RuntimeError(f"{len(missing)} queries missing from evidence cache file.")

    logger.info("CACHE VERIFICATION SUCCESSFUL: 100%% of benchmark queries (%d/%d) present in disk cache '%s'!", len(examples), len(examples), cache.cache_file)


if __name__ == "__main__":
    prefetch_benchmark_cache()
