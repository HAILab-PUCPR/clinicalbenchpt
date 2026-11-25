"""
Audit logging and reporting for ClinicalBenchPT.
Follows Section 6 of the Curation Specification.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CURATION_REPORT_FIELDS = [
    "dataset",
    "original_id",
    "group_id",
    "issue_type",
    "action",
    "reason",
    "retained_id",
    "input_hash",
    "original_split",
    "status",
]

CURATION_SUMMARY_FIELDS = [
    "dataset",
    "initial_records",
    "excluded_empty",
    "dedup_exact",
    "dedup_normalized",
    "excluded_conflict",
    "excluded_multianswer",
    "final_records",
    "pct_removed",
    "label_distribution_before",
    "label_distribution_after",
]

SPLIT_INTEGRITY_FIELDS = [
    "dataset",
    "train_count",
    "dev_count",
    "test_count",
    "id_overlap",
    "input_overlap",
    "exact_example_overlap",
    "internal_duplicates",
    "status",
]


class AuditLogger:
    """Manages audit files in data/audit directory."""

    def __init__(self, audit_dir: Path):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.curation_report_path = self.audit_dir / "curation_report.csv"
        self.curation_summary_path = self.audit_dir / "curation_summary.csv"
        self.split_integrity_path = self.audit_dir / "split_integrity_report.csv"
        self.audit_metadata_path = self.audit_dir / "audit_metadata.json"

    def log_curation_events(self, events: List[Dict[str, Any]]) -> None:
        """Append curation event rows to curation_report.csv."""
        if not events:
            return

        file_exists = self.curation_report_path.exists()
        with open(self.curation_report_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CURATION_REPORT_FIELDS)
            if not file_exists:
                writer.writeheader()
            for event in events:
                row = {k: event.get(k, "") for k in CURATION_REPORT_FIELDS}
                writer.writerow(row)

    def update_summary(self, summary_data: Dict[str, Any]) -> None:
        """Add or update summary row for a dataset in curation_summary.csv."""
        rows = {}
        if self.curation_summary_path.exists():
            with open(self.curation_summary_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows[r["dataset"]] = r

        rows[summary_data["dataset"]] = {k: summary_data.get(k, "") for k in CURATION_SUMMARY_FIELDS}

        with open(self.curation_summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CURATION_SUMMARY_FIELDS)
            writer.writeheader()
            for r in rows.values():
                writer.writerow(r)

    def update_split_integrity(self, integrity_data: Dict[str, Any]) -> None:
        """Add or update split integrity row for a dataset."""
        rows = {}
        if self.split_integrity_path.exists():
            with open(self.split_integrity_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows[r["dataset"]] = r

        rows[integrity_data["dataset"]] = {k: integrity_data.get(k, "") for k in SPLIT_INTEGRITY_FIELDS}

        with open(self.split_integrity_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SPLIT_INTEGRITY_FIELDS)
            writer.writeheader()
            for r in rows.values():
                writer.writerow(r)

    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        """Write execution metadata to audit_metadata.json."""
        existing = {}
        if self.audit_metadata_path.exists():
            try:
                with open(self.audit_metadata_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

        existing.update(metadata)
        existing["last_updated"] = datetime.now().isoformat()

        with open(self.audit_metadata_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
