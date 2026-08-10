"""Evaluation report generator module.

Aggregates metrics, ablation study Markdown tables, calibration ECE, performance stats,
error taxonomy summaries, and statistical significance into Markdown, JSON, and CSV summaries inside `backend/reports/`.
"""

import csv
import json
from pathlib import Path
from typing import Any


class EvaluationReportGenerator:
    """Generates multi-format reports (JSON, CSV, Markdown) for research evaluation runs."""

    def __init__(self, reports_dir: str | Path | None = None):
        if reports_dir is None:
            backend_dir = Path(__file__).resolve().parent.parent
            reports_dir = backend_dir / "reports"
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_reports(
        self,
        benchmark_name: str,
        metrics: dict[str, Any],
        ablation_summary: dict[str, Any] | None = None,
        performance_summary: dict[str, Any] | None = None,
        calibration_summary: dict[str, Any] | None = None,
        error_summary: dict[str, Any] | None = None,
        statistical_summary: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Generate evaluation_summary.json, evaluation_summary.csv, and evaluation_summary.md."""
        data = {
            "benchmark": benchmark_name,
            "metrics": metrics,
            "ablation": ablation_summary or {},
            "performance": performance_summary or {},
            "calibration": calibration_summary or {},
            "error_analysis": error_summary or {},
            "statistical_significance": statistical_summary or {},
        }

        # 1. JSON Report
        json_path = self.reports_dir / "evaluation_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # 2. CSV Report
        csv_path = self.reports_dir / "evaluation_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for k, v in metrics.items():
                if isinstance(v, (int, float, str, bool)):
                    writer.writerow([k, v])
            if performance_summary:
                for k, v in performance_summary.items():
                    if isinstance(v, (int, float, str, bool)):
                        writer.writerow([f"perf_{k}", v])
            if calibration_summary:
                for k, v in calibration_summary.items():
                    if isinstance(v, (int, float, str, bool)):
                        writer.writerow([f"calib_{k}", v])

        # 3. Markdown Report
        md_path = self.reports_dir / "evaluation_summary.md"
        md_content = self._build_markdown_report(data)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return {
            "json": json_path,
            "csv": csv_path,
            "markdown": md_path,
        }

    def _build_markdown_report(self, data: dict[str, Any]) -> str:
        b_name = data.get("benchmark", "HaluEval")
        m = data.get("metrics", {})
        p = data.get("performance", {})
        c = data.get("calibration", {})
        e = data.get("error_analysis", {})
        s = data.get("statistical_significance", {})
        abl = data.get("ablation", {})

        lines = [
            f"# Research Evaluation Summary: {b_name}",
            "",
            "## 1. Classification Metrics Summary",
            "",
            "| Metric | Value |",
            "| :--- | :---: |",
            f"| **Samples** | {m.get('samples', 0)} |",
            f"| **Accuracy** | {m.get('accuracy', 0.0):.4f} |",
            f"| **Precision** | {m.get('precision', 0.0):.4f} |",
            f"| **Recall** | {m.get('recall', 0.0):.4f} |",
            f"| **F1 Score** | {m.get('f1', 0.0):.4f} |",
            f"| **ROC-AUC** | {m.get('roc_auc', 0.0):.4f} |",
            f"| **False Positive Rate (FPR)** | {m.get('false_positive_rate', 0.0):.4f} |",
            f"| **False Negative Rate (FNR)** | {m.get('false_negative_rate', 0.0):.4f} |",
            "",
        ]

        if s:
            lines.extend([
                "## 2. Statistical Significance Testing (McNemar Test)",
                "",
                f"- **Baseline Accuracy**: {s.get('baseline_accuracy', 0.0):.4f}",
                f"- **Proposed Hybrid Accuracy**: {s.get('comparison_accuracy', 0.0):.4f}",
                f"- **Absolute Improvement**: {s.get('accuracy_improvement', 0.0):+.4f}",
                f"- **McNemar p-value**: `{s.get('mcnemar_p_value', 1.0):.5f}`",
                f"- **Statistically Significant**: `{'YES' if s.get('significant') else 'NO'}`",
                "",
            ])

        if abl and "comparison_table_markdown" in abl:
            lines.extend([
                "## 3. Ablation Study Matrix",
                "",
                abl["comparison_table_markdown"],
                "",
            ])

        if p:
            lines.extend([
                "## 4. Performance & Resource Benchmarks",
                "",
                "| Stage / Resource | Benchmark Value |",
                "| :--- | :---: |",
                f"| **Retrieval Latency** | {p.get('retrieval_ms', 0.0):.2f} ms |",
                f"| **Verification Latency** | {p.get('verification_ms', 0.0):.2f} ms |",
                f"| **Feature Extraction Latency** | {p.get('feature_extraction_ms', 0.0):.2f} ms |",
                f"| **Classification Latency** | {p.get('classification_ms', 0.0):.2f} ms |",
                f"| **Total Pipeline Latency** | {p.get('total_ms', 0.0):.2f} ms |",
                f"| **Memory Usage** | {p.get('memory_mb', 0.0):.2f} MB |",
                f"| **CPU Usage** | {p.get('cpu_percent', 0.0):.2f} % |",
                "",
            ])

        if c:
            lines.extend([
                "## 5. Confidence Calibration",
                "",
                f"- **Expected Calibration Error (ECE)**: `{c.get('expected_calibration_error', 0.0):.4f}`",
                f"- **Brier Score**: `{c.get('brier_score', 0.0):.4f}`",
                "",
            ])

        if e:
            lines.extend([
                "## 6. Error Analysis Taxonomy Breakdown",
                "",
                "| Error Category | Count |",
                "| :--- | :---: |",
            ])
            for err_k, err_v in e.get("error_summary", {}).items():
                lines.append(f"| **{err_k}** | {err_v} |")
            lines.append("")

        return "\n".join(lines)
