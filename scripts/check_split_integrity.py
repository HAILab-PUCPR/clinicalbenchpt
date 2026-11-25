
import json
import hashlib
import re
import unicodedata
from pathlib import Path
from itertools import combinations

BENCHMARK_DIR = Path("data/benchmark")
SPLITS = ["train", "dev", "test"]

# Campos que representam o conteúdo principal de cada tarefa
TEXT_FIELDS = [
    "text", "Evolucao", "question", "enunciado"
]

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
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def get_content(record):
    # Para sumarização, usamos o texto-fonte,
    # não o resumo de referência
    if "text" in record:
        return record["text"]

    for field in TEXT_FIELDS:
        if field in record:
            return record[field]

    return None

def check_dataset(folder):
    data = {}

    for split in SPLITS:
        file = folder / f"{split}.jsonl"
        if not file.exists():
            return

        data[split] = load_jsonl(file)

    print(f"\nDataset: {folder.name}")

    for a, b in combinations(SPLITS, 2):
        ids_a = {str(r["id"]) for r in data[a] if "id" in r}
        ids_b = {str(r["id"]) for r in data[b] if "id" in r}

        content_a = {
            hash_text(get_content(r))
            for r in data[a]
            if get_content(r) is not None
        }
        content_b = {
            hash_text(get_content(r))
            for r in data[b]
            if get_content(r) is not None
        }

        common_ids = ids_a & ids_b
        common_texts = content_a & content_b

        print(
            f"{a} vs {b}: "
            f"{len(common_ids)} shared IDs; "
            f"{len(common_texts)} identical texts"
        )

for folder in sorted(BENCHMARK_DIR.iterdir()):
    if folder.is_dir():
        check_dataset(folder)
