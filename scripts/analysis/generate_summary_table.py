"""
Generates consolidated statistics table across all 8 ClinicalBenchPT datasets.
Exports to CSV and LaTeX (ready for the LRE paper).
"""

import csv
import json
import sys
from pathlib import Path
import numpy as np

DATASET_CONFIG = {
    "abcfarma_qa": {"task": "Question Answering", "domain": "Farmacologia / Principios Ativos"},
    "amazon_diseases": {"task": "Question Answering", "domain": "Doencas Tropicais / Amazonia"},
    "wikidoc_pt": {"task": "Question Answering", "domain": "Medicina Geral / WikiDoc"},
    "drbodebench": {"task": "Multiple Choice", "domain": "Exames Medicos (Revalida/Residencia)"},
    "fall_detection": {"task": "Classification", "domain": "Risco de Quedas / Prontuarios"},
    "multiclinsum_pt": {"task": "Summarization", "domain": "Casos Clinicos Hospitalares"},
    "semclinbr": {"task": "Named Entity Rec.", "domain": "Notas Clinicas com Grupos Semanticos"},
    "clinical_ner": {"task": "Named Entity Rec.", "domain": "Textos Biomedicos e Clinicos"},
}


def compute_dataset_stats(dataset_dir: Path, dataset_name: str, config: dict) -> dict:
    splits_data = {}
    for split in ["train", "dev", "test", "full"]:
        p = dataset_dir / f"{split}.jsonl"
        recs = []
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        recs.append(json.loads(line))
        splits_data[split] = recs

    full_recs = splits_data["full"] if splits_data["full"] else (splits_data["train"] + splits_data["dev"] + splits_data["test"])
    task = config["task"]

    input_lens = []
    output_lens = []
    total_entities = 0

    for r in full_recs:
        # Input text length
        if "question" in r or "Question" in r:
            inp = r.get("question") or r.get("Question") or ""
        elif "Evolucao" in r:
            inp = r.get("Evolucao") or ""
        elif "enunciado" in r:
            inp = r.get("enunciado") or ""
        else:
            inp = r.get("text") or ""

        inp_tokens = len(str(inp).strip().split())
        input_lens.append(inp_tokens)

        # Output text length / entities
        if task == "Question Answering":
            out = r.get("answer") or r.get("Response") or ""
            output_lens.append(len(str(out).strip().split()))
        elif task == "Summarization":
            out = r.get("summary") or ""
            output_lens.append(len(str(out).strip().split()))
        elif task == "Named Entity Rec.":
            tags = r.get("tags", {})
            if isinstance(tags, dict):
                total_entities += sum(len(ents) for ents in tags.values() if isinstance(ents, list))

    return {
        "dataset": dataset_name,
        "task": task,
        "domain": config["domain"],
        "train": len(splits_data["train"]),
        "dev": len(splits_data["dev"]),
        "test": len(splits_data["test"]),
        "total": len(full_recs),
        "input_len_mean": round(float(np.mean(input_lens)), 1) if input_lens else 0,
        "input_len_median": int(np.median(input_lens)) if input_lens else 0,
        "input_len_max": int(np.max(input_lens)) if input_lens else 0,
        "output_len_mean": round(float(np.mean(output_lens)), 1) if output_lens else 0,
        "total_entities": total_entities,
    }


def main():
    repo_root = Path(__file__).resolve().parents[2]

    if len(sys.argv) > 1:
        benchmark_dir = Path(sys.argv[1])
    else:
        p1 = repo_root / "data" / "benchmark"
        p2 = repo_root / "nova_data" / "benchmark"
        benchmark_dir = p1 if p1.exists() else p2

    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = repo_root / "results" / "corpus_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analisando benchmark em: {benchmark_dir.resolve()}")
    print(f"Diretorio de saida:       {output_dir.resolve()}")

    rows = []
    for dname, cfg in DATASET_CONFIG.items():
        ds_dir = benchmark_dir / dname
        if ds_dir.exists():
            st = compute_dataset_stats(ds_dir, dname, cfg)
            rows.append(st)
        else:
            print(f"[AVISO] Dataset nao encontrado: {ds_dir}")

    # 1. Export CSV
    csv_path = output_dir / "benchmark_statistics_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset",
            "task",
            "domain",
            "train",
            "dev",
            "test",
            "total",
            "input_len_mean",
            "input_len_median",
            "input_len_max",
            "output_len_mean",
            "total_entities",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[OK] Tabela CSV salva em: {csv_path}")

    # 2. Export LaTeX table for paper (Language Resources & Evaluation)
    tex_path = output_dir / "benchmark_statistics_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Tabela gerada automaticamente para o artigo (LRE Journal)\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\caption{Descriptive statistics of the eight curated clinical NLP datasets in ClinicalBenchPT.}\n")
        f.write("\\label{tab:dataset_stats}\n")
        f.write("\\begin{tabular}{llrrrrrrr}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Dataset} & \\textbf{Task} & \\textbf{Train} & \\textbf{Dev} & \\textbf{Test} & \\textbf{Total} & \\textbf{Avg Tokens} & \\textbf{Med Tokens} & \\textbf{Max Tokens} \\\\\n")
        f.write("\\hline\n")
        for r in rows:
            f.write(f"{r['dataset']} & {r['task']} & {r['train']:,} & {r['dev']:,} & {r['test']:,} & {r['total']:,} & {r['input_len_mean']} & {r['input_len_median']} & {r['input_len_max']} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    print(f"[OK] Tabela LaTeX salva em: {tex_path}")

    # 3. Print terminal preview
    print("\n" + "=" * 80)
    print(f"{'Dataset':<18} | {'Task':<18} | {'Train':>6} | {'Dev':>5} | {'Test':>5} | {'Total':>6} | {'Avg Tok':>7}")
    print("-" * 80)
    for r in rows:
        print(f"{r['dataset']:<18} | {r['task']:<18} | {r['train']:>6} | {r['dev']:>5} | {r['test']:>5} | {r['total']:>6} | {r['input_len_mean']:>7}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
