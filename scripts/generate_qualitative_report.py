"""
Script to generate a comprehensive qualitative analysis report (Appendix C)
for ClinicalBenchPT.

For each dataset, it extracts N representative test instances and displays:
1. Input Prompt / Clinical Text
2. Official Ground Truth
3. Responses across all 5 LLMs (Zero-Shot and LoRA side-by-side)
4. BERT prediction (for discriminative tasks: fall_detection, drbodebench, semclinbr, clinical_ner)

Usage:
    python scripts/generate_qualitative_report.py [--samples 3] [--output results/v2/exemplos_qualitativos_inferencia.md]
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
DATASETS = [
    ("abcfarma_qa", "ABCFarma QA", "Pergunta e Resposta Farmacológica (Bulas)"),
    ("amazon_diseases", "Amazon Diseases", "Doenças Tropicais e Negligenciadas"),
    ("wikidoc_pt", "WikiDoc PT", "Medicina Geral e Enciclopédia Clínica"),
    ("multiclinsum_pt", "MultiClinSum PT", "Sumarização de Prontuários e Casos Clínicos"),
    ("fall_detection", "Fall Detection", "Triagem Binária de Risco de Queda"),
    ("drbodebench", "DrBodeBench", "Exame Revalida (Múltipla Escolha A-D)"),
    ("semclinbr", "SemClinBr", "Reconhecimento de Entidades (Prontuários - 4 Classes)"),
    ("clinical_ner", "Clinical NER (ClinPT)", "Reconhecimento de Entidades (Casos - 13 Classes)"),
]

LLM_MODELS = [
    ("google_gemma-3-4b-it", "Gemma-3-4B-IT", "Base Geral"),
    ("CEIA-UFG_Gemma-3-Gaia-PT-BR-4b-it", "Gaia-Gemma-PT-4B", "Especialista em Português"),
    ("google_medgemma-4b-it", "MedGemma-4B-IT", "Especialista Clínico"),
    ("Qwen_Qwen3-4B", "Qwen3-4B", "Raciocínio Multilíngue"),
    ("meta-llama_Llama-3.2-3B-Instruct", "LLaMA-3.2-3B-Instruct", "Compacto Multilíngue"),
]

BERT_MODELS = [
    ("pucpr_biobertpt-all", "BioBERTpt"),
    ("neuralmind_bert-base-portuguese-cased", "BERTimbau"),
]

def load_jsonl_by_id(filepath: Path) -> Dict[str, Dict]:
    """Load JSONL file into dictionary keyed by example ID."""
    records = {}
    if not filepath.exists():
        return records
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                rec_id = str(data.get("id", len(records)))
                records[rec_id] = data
            except Exception:
                continue
    return records

def format_ground_truth(dataset_name: str, test_item: Dict) -> str:
    """Format ground truth reference based on task type."""
    if dataset_name in ["abcfarma_qa", "amazon_diseases", "wikidoc_pt"]:
        ans = test_item.get("answer") or test_item.get("Response") or "N/A"
        return f"**Resposta Esperada:**\n{ans}"
    elif dataset_name == "multiclinsum_pt":
        summ = test_item.get("summary") or test_item.get("summary_pt") or "N/A"
        return f"**Sumário Clínico Esperado:**\n{summ}"
    elif dataset_name == "fall_detection":
        target = test_item.get("Target", "N/A")
        label_str = "1 (Risco / Houve Queda)" if str(target) == "1" else "0 (Sem Queda / Baixo Risco)"
        return f"**Classe Alvo:** `Queda: {target}` ({label_str})"
    elif dataset_name == "drbodebench":
        resp = test_item.get("resposta", "N/A")
        alt = test_item.get("alternativas", {}).get(resp, "")
        return f"**Gabarito Oficial:** Alternativa **({resp})** — {alt}"
    elif dataset_name in ["semclinbr", "clinical_ner"]:
        tags = test_item.get("tags", {})
        lines = []
        for cat, ents in tags.items():
            if ents:
                lines.append(f"- **{cat}:** {'; '.join(ents)}")
        return "**Entidades de Referência:**\n" + ("\n".join(lines) if lines else "*(Nenhuma entidade anotada)*")
    return str(test_item)

def format_input_context(dataset_name: str, test_item: Dict) -> str:
    """Format input context / prompt text for the instance."""
    if dataset_name in ["abcfarma_qa", "amazon_diseases", "wikidoc_pt"]:
        q = test_item.get("question") or test_item.get("Question") or ""
        return f"**Pergunta Clínica:**\n> {q}"
    elif dataset_name == "multiclinsum_pt":
        text = test_item.get("text") or test_item.get("dialogue") or ""
        return f"**Caso Clínico (Entrada):**\n```text\n{text.strip()}\n```"
    elif dataset_name == "fall_detection":
        text = test_item.get("Evolucao") or test_item.get("text") or test_item.get("Sentence") or ""
        return f"**Nota de Enfermagem / Evolução:**\n> *\"{text.strip()}\"*"
    elif dataset_name == "drbodebench":
        enunciado = test_item.get("enunciado", "")
        img = test_item.get("img_description", "")
        alts = test_item.get("alternativas", {})
        alts_str = "\n".join([f"  - **({k})** {v}" for k, v in sorted(alts.items())])
        img_str = f"\n*[Descrição de Imagem: {img}]*\n" if img else ""
        return f"**Enunciado da Questão:**\n{enunciado}{img_str}\n\n**Alternativas:**\n{alts_str}"
    elif dataset_name in ["semclinbr", "clinical_ner"]:
        text = test_item.get("text", "")
        return f"**Texto Clínico:**\n```text\n{text.strip()}\n```"
    return str(test_item)

def clean_output(pred_str: Any) -> str:
    """Clean model prediction output string."""
    if isinstance(pred_str, dict):
        # NER parsed dict
        lines = []
        for k, v in pred_str.items():
            if v:
                lines.append(f"{k}: {'; '.join(v) if isinstance(v, list) else str(v)}")
        return "\n".join(lines) if lines else "*(Nenhuma entidade detectada)*"
    if not isinstance(pred_str, str):
        return str(pred_str)
    return pred_str.strip()

def main():
    parser = argparse.ArgumentParser(description="Generate qualitative analysis report")
    parser.add_argument("--samples", type=int, default=3, help="Number of instances per dataset (default: 3)")
    parser.add_argument("--output", type=str, default="results/v2/exemplos_qualitativos_inferencia.md", help="Output MD path")
    args = parser.parse_args()

    num_samples = args.samples
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"📊 GERANDO RELATÓRIO QUALITATIVO ({num_samples} INSTÂNCIAS POR DATASET)")
    print("=" * 80)

    md_lines = []
    md_lines.append("# 🔬 Apêndice C: Análise Qualitativa e Estudos de Caso Comparativos\n")
    md_lines.append(
        "Este documento apresenta amostras qualitativas de inferência para todos os **8 datasets** do benchmark "
        "**ClinicalBenchPT**. Para cada dataset, são inspecionadas instâncias reais do conjunto de teste "
        "e comparadas lado a lado com as predições geradas pelos **5 LLMs (Zero-Shot e Fine-Tuning LoRA)** "
        "e pelos **Encoders BERT**, confrontadas contra o **Gabarito Oficial**.\n"
    )
    md_lines.append("---\n")

    for ds_id, ds_title, ds_desc in DATASETS:
        print(f"\n[+] Processando dataset: {ds_id} ({ds_title})...")
        md_lines.append(f"## 📁 {ds_title}\n")
        md_lines.append(f"**Descrição da Tarefa:** {ds_desc}\n")

        # Carregar ground-truth test data
        test_file = Path(f"data/benchmark/{ds_id}/test.jsonl")
        test_items = []
        if test_file.exists():
            with open(test_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        test_items.append(json.loads(line))
        else:
            print(f"  [!] Aviso: {test_file} não encontrado localmente.")

        # Carregar predições Zero-Shot e LoRA de todos os modelos
        preds_zs = {}
        preds_lora = {}
        for m_tag, m_name, _ in LLM_MODELS:
            zs_path = Path(f"results/v2/predictions/{ds_id}_{m_tag}.jsonl")
            preds_zs[m_tag] = load_jsonl_by_id(zs_path)

            lora_path = Path(f"results/v2/predictions_lora/{ds_id}_{m_tag}_lora.jsonl")
            preds_lora[m_tag] = load_jsonl_by_id(lora_path)

        # Carregar predições BERT se tarefa discriminativa
        preds_bert = {}
        if ds_id in ["fall_detection", "drbodebench", "semclinbr", "clinical_ner"]:
            for b_tag, b_name in BERT_MODELS:
                bert_path = Path(f"results/v2/predictions_bert/{ds_id}_{b_tag}_bert.jsonl")
                preds_bert[b_tag] = load_jsonl_by_id(bert_path)

        # Identificar IDs de instâncias disponíveis
        if test_items:
            available_ids = [str(item["id"]) for item in test_items[:num_samples]]
        else:
            # Fallback a partir do primeiro arquivo de predições que existir
            available_ids = []
            for p_dict in list(preds_zs.values()) + list(preds_lora.values()):
                if p_dict:
                    available_ids = list(p_dict.keys())[:num_samples]
                    break

        if not available_ids:
            md_lines.append("*Aguardando geração de predições para este dataset.*\n\n---\n")
            continue

        for idx, inst_id in enumerate(available_ids, 1):
            md_lines.append(f"### 🔍 Instância {idx} (ID: `{inst_id}`)\n")

            # Obter o item de teste correspondente
            current_test_item = next((it for it in test_items if str(it.get("id")) == inst_id), None)
            if not current_test_item:
                # Recuperar de alguma predição que guarde prompt/referência
                for p_dict in list(preds_lora.values()) + list(preds_zs.values()):
                    if inst_id in p_dict:
                        current_test_item = p_dict[inst_id]
                        break

            if current_test_item:
                md_lines.append(format_input_context(ds_id, current_test_item))
                md_lines.append("\n" + format_ground_truth(ds_id, current_test_item) + "\n")
            else:
                md_lines.append(f"*Informações de entrada para ID {inst_id} indisponíveis.*\n")

            md_lines.append("#### 🤖 Comparativo das Respostas dos Modelos:\n")

            # Tabela comparativa dos 5 LLMs (Zero-Shot vs LoRA)
            md_lines.append("| Modelo | Resposta Zero-Shot | Resposta Fine-Tuning LoRA |")
            md_lines.append("| :--- | :--- | :--- |")

            for m_tag, m_name, m_family in LLM_MODELS:
                zs_rec = preds_zs[m_tag].get(inst_id, {})
                lora_rec = preds_lora[m_tag].get(inst_id, {})

                zs_out = clean_output(zs_rec.get("prediction", "N/A")).replace("\n", "<br>").replace("|", "\\|")
                lora_out = clean_output(lora_rec.get("prediction", "N/A")).replace("\n", "<br>").replace("|", "\\|")

                if len(zs_out) > 300:
                    zs_out = zs_out[:300] + "... *(truncado)*"
                if len(lora_out) > 300:
                    lora_out = lora_out[:300] + "... *(truncado)*"

                md_lines.append(f"| **{m_name}**<br>*{m_family}* | {zs_out} | {lora_out} |")

            # Adicionar predição do BioBERT se aplicável
            if ds_id in ["fall_detection", "drbodebench", "semclinbr", "clinical_ner"]:
                b_rec = preds_bert.get("pucpr_biobertpt-all", {}).get(inst_id, {})
                b_out = clean_output(b_rec.get("labels_pred", b_rec.get("prediction", "N/A")))
                b_out_fmt = b_out.replace("\n", "<br>").replace("|", "\\|")
                if len(b_out_fmt) > 300:
                    b_out_fmt = b_out_fmt[:300] + "... *(truncado)*"
                md_lines.append(f"| **BioBERTpt**<br>*(Encoder Discriminativo)* | — | **{b_out_fmt}** |")

            md_lines.append("\n---\n")

    full_md = "\n".join(md_lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(full_md)

    print("\n" + "=" * 80)
    print(f"✅ Relatório qualitativo gerado com sucesso em: {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
