"""Experiment logging module.

Captures timestamped benchmark run configurations, git commit metadata,
classification metrics, performance latencies, and device info inside `backend/reports/experiments/`.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any


@dataclass
class ExperimentLog:
    experiment_id: str
    timestamp: str
    dataset: str
    model_version: str
    checkpoint: str
    git_commit: str
    random_seed: int
    python_version: str
    device: str
    metrics: dict[str, Any]
    configuration: dict[str, Any]
    performance: dict[str, Any]
    calibration: dict[str, Any] = field(default_factory=dict)
    statistical_significance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentLogger:
    """Logs experiment results as timestamped JSON files."""

    def __init__(self, reports_dir: str | Path | None = None):
        if reports_dir is None:
            backend_dir = Path(__file__).resolve().parent.parent
            reports_dir = backend_dir / "reports" / "experiments"
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_git_commit() -> str:
        try:
            cmd = ["git", "rev-parse", "--short", "HEAD"]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=2.0)
            return out.decode("utf-8").strip()
        except Exception:
            return "unknown_commit"

    def log_experiment(
        self,
        dataset: str,
        metrics: dict[str, Any],
        configuration: dict[str, Any],
        performance: dict[str, Any],
        calibration: dict[str, Any] | None = None,
        statistical_significance: dict[str, Any] | None = None,
        model_version: str = "1.0",
        checkpoint: str = "multihaludet.pt",
        random_seed: int = 42,
        device: str = "CUDA",
    ) -> Path:
        """Save a new timestamped experiment log JSON file."""
        now = datetime.now()
        ts_str = now.strftime("%Y%m%d_%H%M%S")
        exp_id = f"exp_{ts_str}"

        log_data = ExperimentLog(
            experiment_id=exp_id,
            timestamp=now.isoformat(),
            dataset=dataset,
            model_version=model_version,
            checkpoint=checkpoint,
            git_commit=self.get_git_commit(),
            random_seed=random_seed,
            python_version=platform.python_version(),
            device=device,
            metrics=metrics,
            configuration=configuration,
            performance=performance,
            calibration=calibration or {},
            statistical_significance=statistical_significance or {},
        )

        out_file = self.reports_dir / f"{exp_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(log_data.to_dict(), f, indent=2)

        return out_file
