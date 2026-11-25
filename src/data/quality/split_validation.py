"""
Split validation and integrity verification.
Follows Section 6.3 and Section 8 of Curation Specification.
"""

from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Tuple

from .deduplication import get_input_text
from .normalization import compute_sha256, normalize_text


def audit_splits(
    train: List[Dict[str, Any]],
    dev: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    dataset_name: str,
    task: str,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Audit splits for ID overlap, input overlap, exact example overlap, and internal duplicates.

    Returns:
        is_clean: True if 0 ID overlap, 0 input overlap, 0 internal duplicate
        report_data: Dict matching SPLIT_INTEGRITY_FIELDS
    """
    splits = {
        "train": train,
        "dev": dev,
        "test": test,
    }

    # 1. ID Overlap
    id_sets = {name: set(str(r.get("id")) for r in recs) for name, recs in splits.items()}
    id_overlaps = 0
    for s1, s2 in combinations(["train", "dev", "test"], 2):
        common_ids = id_sets[s1] & id_sets[s2]
        id_overlaps += len(common_ids)

    # 2. Input Overlap (normalized SHA-256)
    input_hashes = {
        name: [compute_sha256(normalize_text(get_input_text(r, task))) for r in recs]
        for name, recs in splits.items()
    }
    input_hash_sets = {name: set(hashes) for name, hashes in input_hashes.items()}

    input_overlaps = 0
    for s1, s2 in combinations(["train", "dev", "test"], 2):
        common_inputs = input_hash_sets[s1] & input_hash_sets[s2]
        input_overlaps += len(common_inputs)

    # 3. Exact example overlap (full serialized content)
    import json
    def record_signature(r):
        return json.dumps(r, sort_keys=True, ensure_ascii=False)

    sig_sets = {name: set(record_signature(r) for r in recs) for name, recs in splits.items()}
    exact_overlaps = 0
    for s1, s2 in combinations(["train", "dev", "test"], 2):
        common_sigs = sig_sets[s1] & sig_sets[s2]
        exact_overlaps += len(common_sigs)

    # 4. Internal duplicates within each split
    internal_duplicates = 0
    for name, hashes in input_hashes.items():
        counts = Counter(hashes)
        for h, count in counts.items():
            if count > 1:
                internal_duplicates += (count - 1)

    is_clean = (id_overlaps == 0) and (input_overlaps == 0) and (internal_duplicates == 0)
    status = "PASSED" if is_clean else "FAILED"

    report_data = {
        "dataset": dataset_name,
        "train_count": len(train),
        "dev_count": len(dev),
        "test_count": len(test),
        "id_overlap": id_overlaps,
        "input_overlap": input_overlaps,
        "exact_example_overlap": exact_overlaps,
        "internal_duplicates": internal_duplicates,
        "status": status,
    }

    return is_clean, report_data
