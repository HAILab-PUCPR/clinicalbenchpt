"""
Script to display and consolidate all BERT metrics into a summary table,
matching Table 5 from the ClinicalBenchPT paper.

Usage:
    python scripts/show_bert_results.py
"""

import json
import csv
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

METRICS_DIR = Path("results/v2/metrics_bert")
OUT_CSV = Path("results/v2/bert_metrics_summary.csv")

MODELS = [
    ("pucpr_biobertpt-all", "BioBERTpt"),
    ("neuralmind_bert-base-portuguese-cased", "BERTimbau"),
    ("bert-base-multilingual-cased", "BERT-multilingual"),
]

DATASETS = [
    ("clinical_ner", "CLI (NER)", "ner"),
    ("semclinbr", "SEM (SemClinBr)", "ner"),
    ("fall_detection", "FAL (Fall)", "classification"),
    ("drbodebench", "DRB (DrBode)", "multiple_choice"),
]

def load_metric(dataset: str, model_tag: str, task: str) -> float:
    # Pattern: {dataset}_{model_tag}_bert.jsonl_metrics.json
    metric_file = METRICS_DIR / f"{dataset}_{model_tag}_bert.jsonl_metrics.json"
    if not metric_file.exists():
        metric_file = METRICS_DIR / f"{dataset}_{model_tag}_bert_metrics.json"

    if not metric_file.exists():
        return None

    try:
        with open(metric_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if task == "classification":
            c = data.get("classification", data.get("metrics", data))
            return c.get("f1", c.get("macro_f1", None))
        elif task == "multiple_choice":
            mc = data.get("multiple_choice", data.get("metrics", data))
            return mc.get("f1_macro", mc.get("macro_f1", mc.get("f1", None)))
        elif task == "ner":
            nm = data.get("ner_macro", data.get("macro", data))
            return nm.get("f1", None)
    except Exception as e:
        return None

    return None

def main():
    print("=" * 80)
    print(" 📊 RESULTADOS OFICIAIS DOS MODELOS BASEADOS EM BERT (Tabela 5 da TACL)")
    print("=" * 80)

    rows = []
    headers = ["Model", "CLI (NER)", "SEM (SemClinBr)", "FAL (Fall)", "DRB (DrBode)"]
    for model_tag, model_display in MODELS:
        row = {"Model": model_display}
        for dataset, col_name, task in DATASETS:
            score = load_metric(dataset, model_tag, task)
            row[col_name] = f"{score:.4f}" if score is not None else "Em espera / N/A"
        rows.append(row)

    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_CSV, index=False)
    else:
        # Fallback pure python print & csv
        col_w = [20, 15, 18, 15, 15]
        header_str = "".join(f"{h:<{w}}" for h, w in zip(headers, col_w))
        print(header_str)
        print("-" * 65)
        for r in rows:
            line_str = "".join(f"{r[h]:<{w}}" for h, w in zip(headers, col_w))
            print(line_str)

        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    print("=" * 80)
    print(f"\n[OK] Tabela consolidada salva em: {OUT_CSV}")

if __name__ == "__main__":
    main()
