"""
Experiment Manifest Generator and Manager for Publication Pipeline.
Tracks dataset split SHA-256 hashes, hyperparameter logs, model versions, and seeds.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger("hallucination_guard.config.experiment_manifest")


def compute_file_sha256(filepath: Path | str) -> str:
    """Computes SHA-256 hash of a dataset split file for zero-tamper verification."""
    path = Path(filepath)
    if not path.exists():
        return "FILE_NOT_FOUND"
    
    sha256_hash = hashlib.sha256()
    with path.open("rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class ExperimentManifestManager:
    """Manages frozen publication experiment manifests."""

    def __init__(self, manifest_path: str | Path = "./reports/experiments/manifest.yaml") -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def create_manifest(
        self,
        splits_dir: str | Path = "./data/splits",
        backbone: str = "Qwen/Qwen2.5-3B-Instruct",
        seeds: list[int] | None = None,
        hyperparameters: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Generates and writes a publication manifest recording hashes and parameters."""
        splits_path = Path(splits_dir)
        seed_list = seeds or [42, 123, 2024, 3407]

        split_hashes = {}
        if splits_path.exists():
            for split_file in splits_path.glob("*.jsonl"):
                split_hashes[split_file.name] = compute_file_sha256(split_file)
            for csv_file in splits_path.glob("*.csv"):
                split_hashes[csv_file.name] = compute_file_sha256(csv_file)

        manifest_data: Dict[str, Any] = {
            "experiment_version": "publication_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "backbone": {
                "model_name": backbone,
                "precision": "float16",
                "status": "UNIFIED_LOCKED",
            },
            "seeds": {
                "primary": 42,
                "robustness_seeds": seed_list,
            },
            "dataset_splits": split_hashes,
            "hyperparameters": hyperparameters or {
                "batch_size": 32,
                "learning_rate": 0.0001,
                "epochs": 10,
                "sampled_layers": 6,
                "attention_scales": [1, 2, 4],
                "mixup_alpha": 0.2,
                "ensemble_members": 5,
            },
            "frozen_status": "LOCKED",
        }

        with self.manifest_path.open("w", encoding="utf-8") as f:
            yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False)

        logger.info("Created frozen experiment manifest at %s", self.manifest_path)
        return manifest_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = ExperimentManifestManager()
    manifest = manager.create_manifest()
    print("Manifest created successfully:")
    print(json.dumps(manifest, indent=2))
