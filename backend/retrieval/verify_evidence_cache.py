"""
Verification script for the evidence disk cache.

Checks that 100% of the frozen-500 benchmark queries are covered in `backend/data/evidence_cache.json`.
"""

import sys
import json
import logging
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from multihaludet.training.datasets import load_frozen_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hallucination_guard.verify_cache")


def verify_evidence_cache():
    cache_path = backend_dir / "data" / "evidence_cache.json"
    if not cache_path.exists():
        logger.error("VERIFICATION FAILED: Cache file does not exist at '%s'", cache_path)
        sys.exit(1)

    with open(cache_path, "r", encoding="utf-8") as f:
        disk_data = json.load(f)

    test_path = backend_dir / "data" / "halueval_fever_benchmark_500.csv"
    if not test_path.exists():
        test_path = backend_dir / "multihaludet" / "data" / "halueval_benchmark_500.jsonl"

    examples = load_frozen_benchmark(str(test_path))

    covered = 0
    missing = []
    empty = []

    for ex in examples:
        key = ex.query.strip().lower()
        if key not in disk_data:
            missing.append(ex.query)
        elif not disk_data[key]:
            empty.append(ex.query)
        else:
            covered += 1

    total = len(examples)
    logger.info("=" * 60)
    logger.info("EVIDENCE DISK CACHE VERIFICATION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Cache File Path      : %s", cache_path)
    logger.info("  Unique Cached Keys   : %d", len(disk_data))
    logger.info("  Total Benchmark Rows : %d", total)
    logger.info("  Covered Benchmark    : %d / %d (%.1f%%)", covered, total, (covered / total) * 100.0)
    logger.info("  Missing Queries      : %d", len(missing))
    logger.info("  Empty Queries        : %d", len(empty))
    logger.info("=" * 60)

    if missing or empty:
        logger.error("VERIFICATION FAILED: %d missing, %d empty queries.", len(missing), len(empty))
        sys.exit(1)
    else:
        logger.info("SUCCESS: 100%% of frozen-500 benchmark queries (%d/%d) are fully covered in disk cache!", total, total)


if __name__ == "__main__":
    verify_evidence_cache()
