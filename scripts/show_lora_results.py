import json
import glob
import csv
from pathlib import Path

files = sorted(glob.glob("results/v2/metrics_lora/*.json"))

if not files:
    print("Nenhum arquivo de métricas de LoRA encontrado em results/v2/metrics_lora/.")
    exit(0)

print(f"\nEncontrados {len(files)} arquivos de métricas de LoRA.\n")

csv_rows = []
sep_major = "=" * 115
print(sep_major)
print(f"{'MODELO LoRA / DATASET':<52} | DETALHES DE TODAS AS MÉTRICAS LoRA")
print(sep_major)

for f in files:
    name = Path(f).stem.replace("_metrics", "").replace(".jsonl", "").replace("_lora", "")
    with open(f, "r", encoding="utf-8") as fp:
        try:
            d = json.load(fp)
        except Exception:
            continue

    row = {"model_dataset": name}
    print(f"\n[+] {name}")

    if "rouge" in d:
        r1 = d["rouge"].get("rouge1", {})
        r2 = d["rouge"].get("rouge2", {})
        rl = d["rouge"].get("rougeL", {})
        
        r1_f = r1.get("fmeasure", 0)
        r2_f = r2.get("fmeasure", 0)
        rl_f = rl.get("fmeasure", 0)

        row["Task"] = "QA/Summ"
        row["ROUGE-1"] = f"{r1_f:.4f}"
        row["ROUGE-2"] = f"{r2_f:.4f}"
        row["ROUGE-L"] = f"{rl_f:.4f}"

        print(f"    ROUGE-1: {r1_f:.4f} | ROUGE-2: {r2_f:.4f} | ROUGE-L: {rl_f:.4f}")

        bs_bio = d.get("bertscore_biobertpt", {})
        if bs_bio:
            bf = bs_bio.get("f1", 0)
            row["BioBERT_F1"] = f"{bf:.4f}"
            print(f"    BERTScore (BioBERTpt) -> F1: {bf:.4f}")

        bs_imb = d.get("bertscore_bertimbau", {})
        if bs_imb:
            if1 = bs_imb.get("f1", 0)
            row["BERTimbau_F1"] = f"{if1:.4f}"
            print(f"    BERTScore (BERTimbau) -> F1: {if1:.4f}")

    elif "classification" in d:
        c = d["classification"]
        p = c.get("precision", 0)
        r = c.get("recall", 0)
        f1 = c.get("f1", 0)

        row["Task"] = "Classification"
        row["Class_F1"] = f"{f1:.4f}"
        row["Class_Prec"] = f"{p:.4f}"
        row["Class_Rec"] = f"{r:.4f}"

        print(f"    Classificação -> F1: {f1:.4f} | Prec: {p:.4f} | Rec: {r:.4f}")

    elif "multiple_choice" in d:
        mc = d["multiple_choice"]
        f1_m = mc.get("f1_macro", 0)
        p_m = mc.get("precision_macro", 0)
        r_m = mc.get("recall_macro", 0)

        row["Task"] = "Multiple_Choice"
        row["MC_Macro_F1"] = f"{f1_m:.4f}"

        print(f"    Múltipla Escolha -> Macro-F1: {f1_m:.4f} | Prec: {p_m:.4f} | Rec: {r_m:.4f}")

    elif "ner_macro" in d or "ner_micro" in d:
        mac = d.get("ner_macro", {})
        mic = d.get("ner_micro", {})

        mac_f = mac.get("f1", 0)
        mic_f = mic.get("f1", 0)

        row["Task"] = "NER"
        row["NER_Macro_F1"] = f"{mac_f:.4f}"
        row["NER_Micro_F1"] = f"{mic_f:.4f}"

        print(f"    NER Macro -> F1: {mac_f:.4f} | NER Micro -> F1: {mic_f:.4f}")

    csv_rows.append(row)

print("\n" + sep_major)

out_csv = Path("results/v2/lora_metrics_summary.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)

if csv_rows:
    all_keys = ["model_dataset", "Task"]
    for r in csv_rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)

    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n[OK] Tabela LoRA salva em: {out_csv.resolve()}\n")
