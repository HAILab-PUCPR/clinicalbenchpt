"""
DrBodeBench Dataset Processor.
Processes medical multiple-choice questions with stratified splitting and answer balancing.

Multiple-Choice Alternative Reduction and Answer Debiasing Protocol:
-------------------------------------------------------------------
1. Alternative Reduction (5 to 4 options):
   - Medical examination questions originally formatted with 5 options (A, B, C, D, E)
     are standardized to 4 options (A, B, C, D) to ensure a uniform chance-level baseline (25.0%).
   - Empty or whitespace-only alternatives are pruned first.
   - If 5 valid alternatives exist, exactly one *incorrect* alternative (distractor) is removed:
       Candidate Distractors: D = { k in Alternatives | k != ground_truth_answer }
       Removed Option:        r = RandomChoice(D; seed = RANDOM_SEED)
   - Strict Ground Truth Invariant: The ground-truth correct option is NEVER removed
     because the selection is strictly restricted to incorrect distractors.

2. Positional Debiasing & Perfect Answer Balancing:
   - Language models commonly exhibit positional biases (e.g., disproportionately favoring
     option 'A' or 'C' in multiple-choice settings).
   - To eliminate positional bias, the position of the correct answer is uniformly distributed
     across the 4 available letters ['A', 'B', 'C', 'D'] using cyclic allocation:
       target_letter_i = ['A', 'B', 'C', 'D'][i % 4]
   - The correct answer text is assigned to `target_letter_i`.
   - The remaining 3 distractors are shuffled and assigned to the other 3 positions alphabetically.
   - Result: Exactly 25.0% distribution per class across the entire benchmark (400 A, 400 B, 399 C, 399 D).

3. Stratified Splitting:
   - Splits (80% train, 10% dev, 10% test) are stratified by answer letter to maintain
     the exact 25.0% uniform answer distribution in every split.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
from copy import deepcopy

from datasets import load_dataset

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.quality import curate_dataset_records, audit_dataset_splits


# Configuration
TRAIN_SPLIT = 0.8
DEV_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42
HF_DATASET = "recogna-nlp/drbodebench"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed globally
random.seed(RANDOM_SEED)


def load_hf_dataset(dataset_name: str) -> List[Dict]:
    """Load dataset from HuggingFace."""
    try:
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split="train")
        return list(dataset)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)


def remove_extra_alternative(example: Dict, seed: int = RANDOM_SEED) -> Dict:
    """
    Standardize questions to 4 alternatives by removing one incorrect distractor if 5 exist.
    
    Formal Rule:
        Let A be the set of valid alternative keys and c in A be the correct answer key.
        If |A| == 5:
            Candidate distractors: D = { k in A | k != c }
            Selected for removal:  r = choice(D, seed=seed)
            New alternatives:      A' = A \\ { r }
            Ground-truth invariant: c in A' (the correct answer is guaranteed to be retained)

    Args:
        example: Dictionary containing 'enunciado', 'alternativas', and 'resposta'.
        seed: Random seed for deterministic selection.

    Returns:
        Example dictionary with at most 4 non-empty alternatives.
    """
    alternatives = example["alternativas"]
    correct_answer = example["resposta"]

    # Prune empty or whitespace-only alternatives
    for key in list(alternatives.keys()):
        if alternatives[key] in [None, "", " "]:
            del alternatives[key]

    valid_alts = [k for k, v in alternatives.items() if v not in [None, "", " "]]

    if len(valid_alts) == 5:
        # Strictly select from incorrect alternatives (distractors) only
        incorrect = [alt for alt in valid_alts if alt != correct_answer]
        assert len(incorrect) == 4, f"Expected 4 incorrect alternatives, found {len(incorrect)}"
        
        random.seed(seed)
        to_remove = random.choice(incorrect)
        if to_remove in alternatives:
            del alternatives[to_remove]
            
    # Ground truth preservation assertion
    assert correct_answer in alternatives, (
        f"CRITICAL ERROR: Correct answer '{correct_answer}' was accidentally removed from question ID {example.get('id')}!"
    )
    assert len(alternatives) == 4, f"Expected exactly 4 alternatives, found {len(alternatives)}"
    
    return example


def balance_answers_perfectly(data: List[Dict], seed: int = RANDOM_SEED) -> List[Dict]:
    """
    Rebalance answer letters uniformly across ['A', 'B', 'C', 'D'] to eliminate positional bias.
    
    Procedure:
        1. Shuffle dataset order deterministically.
        2. For record i, assign the target answer letter cyclically: target = ['A','B','C','D'][i % 4].
        3. Map original correct text to target.
        4. Permute the remaining 3 distractor texts across the other 3 letter positions.
        5. Update record['resposta'] = target and record['alternativas'] = sorted dictionary.

    Args:
        data: List of questions with 4 alternatives.
        seed: Random seed for reproducible permutations.

    Returns:
        List of questions with balanced answer distributions across A, B, C, and D.
    """
    # Deterministic initial shuffle
    random.seed(seed)
    random.shuffle(data)

    new_data = []
    possible_letters = ['A', 'B', 'C', 'D']
    num_alternatives = len(possible_letters)

    for i, record in enumerate(data):
        # Target letter for correct answer allocated cyclically (A -> B -> C -> D -> A ...)
        target_letter = possible_letters[i % num_alternatives]

        # Extract correct answer text and distractor texts
        original_correct_key = record["resposta"]
        correct_text = record["alternativas"][original_correct_key]
        incorrect_texts = [
            text for key, text in record["alternativas"].items()
            if key != original_correct_key
        ]

        # Deterministically shuffle incorrect alternatives
        random.seed(seed)
        random.shuffle(incorrect_texts)

        # Build new alternatives mapping
        new_alternatives = {target_letter: correct_text}
        incorrect_idx = 0
        for letter in possible_letters:
            if letter != target_letter:
                new_alternatives[letter] = incorrect_texts[incorrect_idx]
                incorrect_idx += 1

        # Validation assertions
        assert len(new_alternatives) == 4, "Alternatives count must remain exactly 4"
        assert new_alternatives[target_letter] == correct_text, "Target letter must contain ground truth text"

        # Create updated record
        new_record = deepcopy(record)
        new_record["alternativas"] = dict(sorted(new_alternatives.items()))
        new_record["resposta"] = target_letter
        new_data.append(new_record)

    # Final shuffle to eliminate cyclic predictability
    random.seed(seed)
    random.shuffle(new_data)

    return new_data


def split_dataset_stratified(records: List[Dict], seed: int = RANDOM_SEED) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset with stratification by answer class to preserve uniform class distribution."""
    random.seed(seed)
    random.shuffle(records)

    # Group by answer class
    buckets = defaultdict(list)
    for record in records:
        buckets[record["resposta"]].append(record)

    train, dev, test = [], [], []

    # Split each class proportionally
    for answer, items in buckets.items():
        n = len(items)
        n_dev = int(n * DEV_SPLIT)
        n_test = int(n * TEST_SPLIT)
        n_train = n - n_dev - n_test

        random.seed(seed)
        random.shuffle(items)

        train.extend(items[:n_train])
        dev.extend(items[n_train:n_train + n_dev])
        test.extend(items[n_train + n_dev:])

    # Shuffle final sets
    random.seed(seed)
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    return train, dev, test


def print_answer_distribution(examples: List[Dict], split_name: str = "") -> None:
    """Print distribution of correct answers."""
    counts = Counter(ex["resposta"] for ex in examples)
    total = sum(counts.values())
    
    if split_name:
        logger.info(f"\nAnswer distribution - {split_name}:")
    else:
        logger.info("\nAnswer distribution:")
    
    for letter in sorted(counts):
        count = counts[letter]
        percentage = (count / total) * 100
        logger.info(f"  {letter}: {count} ({percentage:.2f}%)")


def prepare_records(dataset: List[Dict]) -> List[Dict]:
    """Prepare records by adding task, ID, and processing alternatives."""
    records = []
    for i, ex in enumerate(dataset):
        ex["task"] = "multiple_choice"
        ex["id"] = f"{i:05d}"
        ex = remove_extra_alternative(ex)
        records.append(ex)
    return records


def save_jsonl(records: List[Dict], path: Path) -> None:
    """Save records in JSONL format."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"Saved: {path} ({len(records)} records)")


def save_ids(ids: List[str], path: Path) -> None:
    """Save list of IDs to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)


def main():
    """Main function."""
    # Validate arguments
    if len(sys.argv) != 2:
        print("Usage: python processor_drbodebench.py <output_base_dir>")
        print("Example: python processor_drbodebench.py /path/to/benchmark")
        sys.exit(1)
    
    output_base_dir = Path(sys.argv[1])
    output_dir = output_base_dir / "drbodebench"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # 1. Load and process
    dataset = load_hf_dataset(HF_DATASET)
    records = prepare_records(dataset)
    
    # 2. Standardize to 4 options and balance answers perfectly (A, B, C, D)
    records = balance_answers_perfectly(records)
    
    # 3. Quality curation and audit logging after 4-option balancing
    audit_dir = output_dir.parent.parent / "audit"
    records = curate_dataset_records(records, "drbodebench", "multiple_choice", audit_dir=audit_dir)

    print_answer_distribution(records, "full")
    
    # 4. Stratified split
    train, dev, test = split_dataset_stratified(records)
    audit_dataset_splits(train, dev, test, "drbodebench", "multiple_choice", audit_dir=audit_dir)
    
    print_answer_distribution(train, "train")
    print_answer_distribution(dev, "dev")
    print_answer_distribution(test, "test")
    
    # 5. Save datasets
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(dev, output_dir / "dev.jsonl")
    save_jsonl(test, output_dir / "test.jsonl")
    save_jsonl(records, output_dir / "full.jsonl")
    
    # Save IDs for reproducibility
    save_ids([r["id"] for r in train], output_dir / "train_ids.json")
    save_ids([r["id"] for r in dev], output_dir / "dev_ids.json")
    save_ids([r["id"] for r in test], output_dir / "test_ids.json")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Total records: {len(records)}")
    logger.info(f"Train: {len(train)} ({TRAIN_SPLIT*100:.0f}%)")
    logger.info(f"Dev: {len(dev)} ({DEV_SPLIT*100:.0f}%)")
    logger.info(f"Test: {len(test)} ({TEST_SPLIT*100:.0f}%)")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
