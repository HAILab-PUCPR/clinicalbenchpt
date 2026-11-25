"""
Corpus analysis script for Named Entity Recognition (NER) datasets.
Compatible with ClinicalBenchPT format (text + tags) and legacy format (tokens + entities).
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict


def analisar_ner_arquivo(caminho_jsonl, output_dir):
    tipo_entidade_counter = Counter()
    total_tokens = 0
    tamanho_entidades = defaultdict(list)
    exemplos = 0
    exemplos_com_entidades = 0
    tokens_em_entidades = 0

    with open(caminho_jsonl, "r", encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            doc = json.loads(linha)
            exemplos += 1

            # 1. Formato ClinicalBenchPT: {"text": str, "tags": {group: [entity, ...]}}
            if "text" in doc and "tags" in doc:
                texto = doc.get("text", "")
                tokens = texto.strip().split()
                total_tokens += len(tokens)

                tags = doc.get("tags", {})
                doc_has_entities = False

                for label, entity_list in tags.items():
                    if isinstance(entity_list, list):
                        for ent in entity_list:
                            if not ent:
                                continue
                            doc_has_entities = True
                            tipo_entidade_counter[label] += 1
                            ent_tokens = len(str(ent).strip().split())
                            tokens_em_entidades += ent_tokens
                            tamanho_entidades[label].append(ent_tokens)

                if doc_has_entities:
                    exemplos_com_entidades += 1

            # 2. Formato legado: {"tokens": [...], "entities": [{"label": ..., "tokens": [...]}]}
            elif "tokens" in doc and "entities" in doc:
                tokens = doc.get("tokens", [])
                entities = doc.get("entities", [])
                total_tokens += len(tokens)

                if entities:
                    exemplos_com_entidades += 1

                for ent in entities:
                    label = ent.get("label", "ENTITY")
                    ent_toks = ent.get("tokens", [])
                    token_count = len(ent_toks) if isinstance(ent_toks, list) else 1
                    tipo_entidade_counter[label] += 1
                    tokens_em_entidades += token_count
                    tamanho_entidades[label].append(token_count)

    nome_base = caminho_jsonl.stem
    output_path = output_dir / f"analise_ner_{nome_base}.txt"

    total_entidades = sum(tipo_entidade_counter.values())

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"Arquivo: {caminho_jsonl.name}\n")
        out.write(f"Total de exemplos: {exemplos}\n")
        out.write(f"Exemplos com pelo menos uma entidade: {exemplos_com_entidades}\n")
        out.write(f"Tokens totais: {total_tokens}\n")
        
        pct_tokens_ent = (tokens_em_entidades / total_tokens * 100) if total_tokens > 0 else 0
        out.write(f"Tokens em entidades: {tokens_em_entidades} ({pct_tokens_ent:.2f}%)\n")
        out.write(f"Total de entidades: {total_entidades}\n")
        
        media_ent_exemplo = (total_entidades / exemplos) if exemplos > 0 else 0
        out.write(f"Media de entidades por exemplo: {media_ent_exemplo:.2f}\n\n")

        out.write("Entidades por tipo / grupo semantico:\n")
        for label, count in tipo_entidade_counter.most_common():
            lens = tamanho_entidades[label]
            avg_len = sum(lens) / len(lens) if lens else 0
            pct_ent = (count / total_entidades * 100) if total_entidades > 0 else 0
            out.write(f" - {label}: {count} entidades ({pct_ent:.2f}% | media de {avg_len:.2f} tokens)\n")

    print(f"[OK] Estatisticas salvas em: {output_path}")


def main():
    if len(sys.argv) != 3:
        print("Uso: python analyze_ner.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    arquivos = ["train.jsonl", "dev.jsonl", "test.jsonl", "full.jsonl"]
    for nome in arquivos:
        caminho = input_dir / nome
        if caminho.exists():
            analisar_ner_arquivo(caminho, output_dir)
        else:
            print(f"[AVISO] Arquivo nao encontrado: {caminho}")


if __name__ == "__main__":
    main()
