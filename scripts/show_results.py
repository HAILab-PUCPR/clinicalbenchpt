import json
import glob
import csv
from pathlib import Path

files = sorted(glob.glob("results/v2/metrics/*.json"))

if not files:
    print("Nenhum arquivo de métricas encontrado em results/v2/metrics/.")
    exit(0)

print(f"\nEncontrados {len(files)} arquivos de métricas.\n")

csv_rows = []

sep_major = "=" * 115
sep_minor = "-" * 115

print(sep_major)
print(f"{'MODELO / DATASET':<52} | DETALHES DE TODAS AS MÉTRICAS")
print(sep_major)

for f in files:
    name = Path(f).stem.replace("_metrics", "").replace(".jsonl", "")
    with open(f, "r", encoding="utf-8") as fp:
        try:
            d = json.load(fp)
        except Exception:
            continue

    row = {"model_dataset": name}
    print(f"\n[+] {name}")

    # 1. QA e Sumarizacao (ROUGE + BERTScore)
    if "rouge" in d:
        r1 = d["rouge"].get("rouge1", {})
        r2 = d["rouge"].get("rouge2", {})
        rl = d["rouge"].get("rougeL", {})
        
        r1_f = r1.get("fmeasure", 0)
        r2_f = r2.get("fmeasure", 0)
        rl_p = rl.get("precision", 0)
        rl_r = rl.get("recall", 0)
        rl_f = rl.get("fmeasure", 0)

        row["Task"] = "QA/Summ"
        row["ROUGE-1"] = f"{r1_f:.4f}"
        row["ROUGE-2"] = f"{r2_f:.4f}"
        row["ROUGE-L_P"] = f"{rl_p:.4f}"
        row["ROUGE-L_R"] = f"{rl_r:.4f}"
        row["ROUGE-L_F1"] = f"{rl_f:.4f}"

        print(f"    ROUGE-1 F1: {r1_f:.4f} | ROUGE-2 F1: {r2_f:.4f} | ROUGE-L F1: {rl_f:.4f} (P: {rl_p:.4f}, R: {rl_r:.4f})")

        # BERTScore BioBERTpt
        bs_bio = d.get("bertscore_biobertpt", {})
        if bs_bio:
            bp, br, bf = bs_bio.get("precision", 0), bs_bio.get("recall", 0), bs_bio.get("f1", 0)
            row["BioBERT_P"] = f"{bp:.4f}"
            row["BioBERT_R"] = f"{br:.4f}"
            row["BioBERT_F1"] = f"{bf:.4f}"
            print(f"    BERTScore (BioBERTpt) -> F1: {bf:.4f} | Prec: {bp:.4f} | Rec: {br:.4f}")

        # BERTScore BERTimbau
        bs_imb = d.get("bertscore_bertimbau", {})
        if bs_imb:
            ip, ir, if1 = bs_imb.get("precision", 0), bs_imb.get("recall", 0), bs_imb.get("f1", 0)
            row["BERTimbau_P"] = f"{ip:.4f}"
            row["BERTimbau_R"] = f"{ir:.4f}"
            row["BERTimbau_F1"] = f"{if1:.4f}"
            print(f"    BERTScore (BERTimbau) -> F1: {if1:.4f} | Prec: {ip:.4f} | Rec: {ir:.4f}")

    # 2. Classificacao (Fall Detection)
    elif "classification" in d:
        c = d["classification"]
        p = c.get("precision", 0)
        r = c.get("recall", 0)
        f1 = c.get("f1", 0)
        acc = c.get("accuracy", 0)

        row["Task"] = "Classification"
        row["Class_F1"] = f"{f1:.4f}"
        row["Class_Prec"] = f"{p:.4f}"
        row["Class_Rec"] = f"{r:.4f}"
        row["Class_Acc"] = f"{acc:.4f}"

        print(f"    Classificação -> F1: {f1:.4f} | Prec: {p:.4f} | Rec: {r:.4f} | Acc: {acc:.4f}")

    # 3. Multipla Escolha (DrBodeBench)
    elif "multiple_choice" in d:
        mc = d["multiple_choice"]
        f1_m = mc.get("f1_macro", 0)
        p_m = mc.get("precision_macro", 0)
        r_m = mc.get("recall_macro", 0)
        acc = mc.get("accuracy", 0)

        row["Task"] = "Multiple_Choice"
        row["MC_Macro_F1"] = f"{f1_m:.4f}"
        row["MC_Macro_P"] = f"{p_m:.4f}"
        row["MC_Macro_R"] = f"{r_m:.4f}"
        row["MC_Acc"] = f"{acc:.4f}"

        print(f"    Múltipla Escolha -> Macro-F1: {f1_m:.4f} | Macro-Prec: {p_m:.4f} | Macro-Rec: {r_m:.4f} | Acc: {acc:.4f}")

    # 4. NER (SemClinBr e Clinical NER)
    elif "ner_macro" in d or "ner_micro" in d:
        mac = d.get("ner_macro", {})
        mic = d.get("ner_micro", {})

        mac_f, mac_p, mac_r = mac.get("f1", 0), mac.get("precision", 0), mac.get("recall", 0)
        mic_f, mic_p, mic_r = mic.get("f1", 0), mic.get("precision", 0), mic.get("recall", 0)

        row["Task"] = "NER"
        row["NER_Macro_F1"] = f"{mac_f:.4f}"
        row["NER_Macro_P"] = f"{mac_p:.4f}"
        row["NER_Macro_R"] = f"{mac_r:.4f}"
        row["NER_Micro_F1"] = f"{mic_f:.4f}"
        row["NER_Micro_P"] = f"{mic_p:.4f}"
        row["NER_Micro_R"] = f"{mic_r:.4f}"

        print(f"    NER Macro -> F1: {mac_f:.4f} | Prec: {mac_p:.4f} | Rec: {mac_r:.4f}")
        print(f"    NER Micro -> F1: {mic_f:.4f} | Prec: {mic_p:.4f} | Rec: {mic_r:.4f}")

    csv_rows.append(row)

print("\n" + sep_major)

# Exportar para CSV consolidado
out_csv = Path("results/v2/metrics_summary.csv")
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

    print(f"\n[OK] Tabela consolidada completa salva em: {out_csv.resolve()}\n")
