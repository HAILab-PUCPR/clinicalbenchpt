"""
ClinicalBenchPT Data Quality, Curation and Audit Package.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .audit_logging import AuditLogger
from .deduplication import curate_dataset, get_input_text
from .normalization import compute_sha256, normalize_text
from .split_validation import audit_splits
from .validation import validate_record

logger = logging.getLogger(__name__)


def curate_dataset_records(
    records: List[Dict[str, Any]],
    dataset_name: str,
    task: str,
    audit_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Curate records before splitting:
    - Removes empty fields (EXCLUDE_EMPTY)
    - Deduplicates identical input/output pairs (DEDUP_*)
    - Excludes contradictory input/output pairs (EXCLUDE_CONFLICT)
    - Logs events and summary to audit_dir (default: data/audit)
    """
    if audit_dir is None:
        # Default to repo_root / data / audit
        audit_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "audit"

    audit_logger = AuditLogger(audit_dir)

    curated, events, summary = curate_dataset(records, dataset_name, task)

    audit_logger.log_curation_events(events)
    audit_logger.update_summary(summary)

    logger.info(
        f"[{dataset_name}] Curation complete: {summary['initial_records']} -> {summary['final_records']} "
        f"({summary['pct_removed']} removed | {summary['excluded_conflict']} conflicts, "
        f"{summary['dedup_exact'] + summary['dedup_normalized']} duplicates, {summary['excluded_empty']} empty)"
    )

    return curated


def audit_dataset_splits(
    train: List[Dict[str, Any]],
    dev: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    dataset_name: str,
    task: str,
    audit_dir: Optional[Path] = None,
) -> bool:
    """
    Verify split integrity:
    - Checks ID overlap
    - Checks input overlap
    - Checks internal duplicates
    - Logs to split_integrity_report.csv
    """
    if audit_dir is None:
        audit_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "audit"

    audit_logger = AuditLogger(audit_dir)
    is_clean, report_data = audit_splits(train, dev, test, dataset_name, task)
    audit_logger.update_split_integrity(report_data)

    if not is_clean:
        logger.warning(
            f"[{dataset_name}] Split integrity FAILED: {report_data['input_overlap']} input overlaps, "
            f"{report_data['id_overlap']} ID overlaps"
        )
    else:
        logger.info(f"[{dataset_name}] Split integrity PASSED: 0 overlap between train/dev/test.")

    return is_clean


__all__ = [
    "curate_dataset_records",
    "audit_dataset_splits",
    "curate_dataset",
    "validate_record",
    "normalize_text",
    "compute_sha256",
    "audit_splits",
    "AuditLogger",
]
