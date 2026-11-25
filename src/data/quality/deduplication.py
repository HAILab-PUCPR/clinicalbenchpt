"""
Deduplication and conflict resolution for ClinicalBenchPT.
Follows Section 3.3, 3.4, 3.5, 3.6 of Curation Specification.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .normalization import compute_sha256, normalize_text
from .validation import validate_record


def get_input_text(record: Dict[str, Any], task: str) -> str:
    """Extract primary input text for a given task."""
    if task == "qa":
        return record.get("question") or record.get("Question") or ""
    elif task == "classification":
        return record.get("Evolucao", "")
    elif task == "multiple_choice":
        return record.get("enunciado", "")
    elif task == "summarization" or task == "ner":
        return record.get("text", "")
    return ""


def get_canonical_output(record: Dict[str, Any], task: str) -> Any:
    """Extract canonical normalized output representation for comparison."""
    if task == "qa":
        ans = record.get("answer") if record.get("answer") is not None else (record.get("Response") or record.get("response") or "")
        return normalize_text(ans)

    elif task == "classification":
        target = record.get("Target")
        return str(target).strip()

    elif task == "multiple_choice":
        enunciado = record.get("enunciado", "")
        alternativas = record.get("alternativas", {})
        resposta = record.get("resposta", "")
        correct_text = alternativas.get(resposta, "") if isinstance(alternativas, dict) else ""
        alt_values = tuple(sorted(normalize_text(v) for v in alternativas.values())) if isinstance(alternativas, dict) else ()
        return (normalize_text(correct_text), alt_values)

    elif task == "summarization":
        return normalize_text(record.get("summary", ""))

    elif task == "ner":
        tags = record.get("tags", {})
        if not isinstance(tags, dict):
            return ()
        canonical_tags = []
        for group, ents in tags.items():
            if ents and isinstance(ents, list):
                norm_ents = tuple(sorted(normalize_text(e) for e in ents if e is not None))
                canonical_tags.append((normalize_text(group), norm_ents))
        return tuple(sorted(canonical_tags))

    return normalize_text(str(record.get("label", "")))


def get_raw_output_repr(record: Dict[str, Any], task: str) -> str:
    """Get raw string representation of output for exact equality check."""
    if task == "qa":
        ans = record.get("answer") if record.get("answer") is not None else (record.get("Response") or record.get("response") or "")
        return str(ans)
    elif task == "classification":
        return str(record.get("Target", ""))
    elif task == "multiple_choice":
        return f"{record.get('resposta', '')}|{record.get('alternativas', {})}"
    elif task == "summarization":
        return str(record.get("summary", ""))
    elif task == "ner":
        return str(record.get("tags", {}))
    return str(record.get("label", ""))


def curate_dataset(
    records: List[Dict[str, Any]],
    dataset_name: str,
    task: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply full curation pipeline on raw records before splitting:
    1. Validation of empty/invalid fields (EXCLUDE_EMPTY)
    2. Deterministic normalization
    3. Grouping by normalized input text
    4. Deduplication of equivalent records (DEDUP_EXACT / DEDUP_NORMALIZED)
    5. Exclusion of conflicting records (EXCLUDE_CONFLICT)

    Returns:
        curated_records: List of retained valid records
        events: List of audit event dicts for curation_report.csv
        summary: Summary dict for curation_summary.csv
    """
    initial_count = len(records)
    events: List[Dict[str, Any]] = []

    # Track label distribution before curation
    label_dist_before: Counter = Counter()
    label_dist_after: Counter = Counter()

    for r in records:
        lbl = r.get("Target") if task == "classification" else (r.get("resposta") if task == "multiple_choice" else None)
        if lbl is not None:
            label_dist_before[str(lbl)] += 1

    # ----------------------------------------------------
    # Step 1: Validation of empty / invalid records
    # ----------------------------------------------------
    valid_records: List[Dict[str, Any]] = []
    excluded_empty_count = 0

    for idx, record in enumerate(records):
        rec_id = str(record.get("id", idx))
        is_valid, field, reason = validate_record(record, task)

        if not is_valid:
            excluded_empty_count += 1
            events.append({
                "dataset": dataset_name,
                "original_id": rec_id,
                "group_id": "",
                "issue_type": "EMPTY_FIELD",
                "action": "EXCLUDE_EMPTY",
                "reason": f"Field '{field}': {reason}",
                "retained_id": "",
                "input_hash": "",
                "original_split": "",
                "status": "EXCLUDED",
            })
        else:
            valid_records.append(record)

    # ----------------------------------------------------
    # Step 2: Grouping by normalized input
    # ----------------------------------------------------
    groups = defaultdict(list)
    for r in valid_records:
        inp = get_input_text(r, task)
        norm_inp = normalize_text(inp)
        inp_hash = compute_sha256(norm_inp)
        groups[inp_hash].append(r)

    # ----------------------------------------------------
    # Step 3: Deduplication & Conflict Resolution
    # ----------------------------------------------------
    curated_records: List[Dict[str, Any]] = []
    dedup_exact_count = 0
    dedup_norm_count = 0
    excluded_conflict_count = 0
    excluded_multianswer_count = 0

    group_counter = 1

    for inp_hash, group_records in groups.items():
        if len(group_records) == 1:
            # Unique record, preserve directly
            curated_records.append(group_records[0])
            continue

        # Group with 2+ records sharing the exact same normalized input
        group_id = f"GRP_{group_counter:05d}"
        group_counter += 1

        outputs = [get_canonical_output(r, task) for r in group_records]
        unique_canonical_outputs = set(outputs)

        if len(unique_canonical_outputs) > 1:
            # CONFLICT: Identical input but different answers/labels!
            # Section 3.4: Exclude all records in this group
            issue_type = "MULTIANSWER_CONFLICT" if task == "qa" else "LABEL_CONFLICT"
            action_code = "EXCLUDE_CONFLICT"
            excluded_conflict_count += len(group_records)

            for r in group_records:
                rec_id = str(r.get("id", ""))
                events.append({
                    "dataset": dataset_name,
                    "original_id": rec_id,
                    "group_id": group_id,
                    "issue_type": issue_type,
                    "action": action_code,
                    "reason": f"Contradictory or multi-answer output across {len(group_records)} records sharing identical input",
                    "retained_id": "",
                    "input_hash": inp_hash,
                    "original_split": "",
                    "status": "EXCLUDED",
                })
        else:
            # EQUIVALENT: All outputs in this group are canonical equivalents!
            # Section 3.3: Retain exactly one instance deterministically (lowest ID)
            sorted_group = sorted(group_records, key=lambda r: str(r.get("id", "")))
            retained_record = sorted_group[0]
            retained_id = str(retained_record.get("id", ""))
            curated_records.append(retained_record)

            raw_inp_rep = get_input_text(retained_record, task)
            raw_out_rep = get_raw_output_repr(retained_record, task)

            for r in sorted_group[1:]:
                rec_id = str(r.get("id", ""))
                other_raw_inp = get_input_text(r, task)
                other_raw_out = get_raw_output_repr(r, task)

                is_exact = (raw_inp_rep == other_raw_inp) and (raw_out_rep == other_raw_out)
                action_code = "DEDUP_EXACT" if is_exact else "DEDUP_NORMALIZED"

                if is_exact:
                    dedup_exact_count += 1
                else:
                    dedup_norm_count += 1

                events.append({
                    "dataset": dataset_name,
                    "original_id": rec_id,
                    "group_id": group_id,
                    "issue_type": "DUPLICATE_ENTRY",
                    "action": action_code,
                    "reason": f"Duplicate of retained record ID {retained_id}",
                    "retained_id": retained_id,
                    "input_hash": inp_hash,
                    "original_split": "",
                    "status": "EXCLUDED",
                })

    # Track label distribution after curation
    for r in curated_records:
        lbl = r.get("Target") if task == "classification" else (r.get("resposta") if task == "multiple_choice" else None)
        if lbl is not None:
            label_dist_after[str(lbl)] += 1

    final_count = len(curated_records)
    removed_count = initial_count - final_count
    pct_removed = (removed_count / initial_count * 100.0) if initial_count > 0 else 0.0

    summary = {
        "dataset": dataset_name,
        "initial_records": initial_count,
        "excluded_empty": excluded_empty_count,
        "dedup_exact": dedup_exact_count,
        "dedup_normalized": dedup_norm_count,
        "excluded_conflict": excluded_conflict_count,
        "excluded_multianswer": excluded_multianswer_count,
        "final_records": final_count,
        "pct_removed": f"{pct_removed:.2f}%",
        "label_distribution_before": dict(label_dist_before) if label_dist_before else "",
        "label_distribution_after": dict(label_dist_after) if label_dist_after else "",
    }

    return curated_records, events, summary
