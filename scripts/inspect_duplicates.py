
import json
import hashlib
import re
import csv
import unicodedata

from pathlib import Path
from collections import defaultdict
from itertools import combinations


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_DIR / "data" / "benchmark"
OUTPUT_FILE = PROJECT_DIR / "split_duplicates_report.csv"

DATASETS = [
    "abcfarma_qa",
    "amazon_diseases",
    "clinical_ner",
    "drbodebench",
    "fall_detection",
    "multiclinsum_pt",
    "semclinbr",
    "wikidoc_pt",
]
SPLITS = ["train", "dev", "test"]

TEXT_FIELDS = [
    "text",
    "Evolucao",
    "question",
    "enunciado",
    "Question",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def hash_text(text):
    return hashlib.sha256(
        normalize(text).encode("utf-8")
    ).hexdigest()


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def get_content(record):
    for field in TEXT_FIELDS:
        if field in record and record[field] is not None:
            return record[field]

    return None


def get_label(record):
    if "answer" in record:
        return record["answer"]

    if "Target" in record:
        return record["Target"]

    if "resposta" in record:
        return record["resposta"]

    return None


# ============================================================
# DUPLICATE INSPECTION
# ============================================================

def inspect_duplicates(folder):
    print(f"\n{'=' * 65}")
    print(f"DATASET: {folder.name}")
    print(f"{'=' * 65}")

    splits = {}

    for split in SPLITS:
        path = folder / f"{split}.jsonl"

        if not path.exists():
            print(f"WARNING: Missing {path}")
            return []

        splits[split] = load_jsonl(path)

    results = []

    for split_a, split_b in combinations(SPLITS, 2):

        index_a = defaultdict(list)
        index_b = defaultdict(list)

        for record in splits[split_a]:
            content = get_content(record)

            if content is not None:
                index_a[hash_text(content)].append(record)

        for record in splits[split_b]:
            content = get_content(record)

            if content is not None:
                index_b[hash_text(content)].append(record)

        common_hashes = index_a.keys() & index_b.keys()

        print(f"\n{split_a} vs {split_b}")
        print(f"Identical normalized texts: {len(common_hashes)}")

        for content_hash in sorted(common_hashes):

            for record_a in index_a[content_hash]:
                for record_b in index_b[content_hash]:

                    content = get_content(record_a)
                    normalized = normalize(content)

                    label_a = get_label(record_a)
                    label_b = get_label(record_b)

                    same_label = (
                        label_a == label_b
                        if label_a is not None and label_b is not None
                        else None
                    )

                    result = {
                        "dataset": folder.name,
                        "split_a": split_a,
                        "id_a": record_a.get("id"),
                        "split_b": split_b,
                        "id_b": record_b.get("id"),
                        "text_length": len(normalized),
                        "content_sha256": content_hash,
                        "same_label_or_answer": same_label,
                    }

                    results.append(result)

                    print(
                        f"  ID {result['id_a']} "
                        f"({split_a}) <-> "
                        f"ID {result['id_b']} "
                        f"({split_b}) | "
                        f"length={result['text_length']} | "
                        f"same_label={same_label}"
                    )

    return results


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(results):
    if not results:
        print("\nNo duplicate texts found.")
        return

    fieldnames = [
        "dataset",
        "split_a",
        "id_a",
        "split_b",
        "id_b",
        "text_length",
        "content_sha256",
        "same_label_or_answer",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nReport saved to: {OUTPUT_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("ClinicalBenchPT - Split Duplicate Inspection")
    print(f"Benchmark directory: {BENCHMARK_DIR}")

    all_results = []

    for dataset in DATASETS:
        folder = BENCHMARK_DIR / dataset

        if not folder.exists():
            print(f"\nWARNING: Dataset not found: {folder}")
            continue

        results = inspect_duplicates(folder)
        all_results.extend(results)

    save_report(all_results)

    print(f"\n{'=' * 65}")
    print("AUDIT SUMMARY")
    print(f"{'=' * 65}")

    for dataset in DATASETS:
        dataset_results = [
            r for r in all_results
            if r["dataset"] == dataset
        ]

        print(
            f"{dataset}: "
            f"{len(dataset_results)} duplicate record pairs"
        )

    print("\nInspection completed.")


if __name__ == "__main__":
    main()
