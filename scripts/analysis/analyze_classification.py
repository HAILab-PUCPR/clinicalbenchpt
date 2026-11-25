"""
Corpus analysis script for Clinical Text Classification datasets (e.g. Fall Detection).
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict


def analisar_arquivo_classificacao(caminho_jsonl, output_dir):
    textos = []
    labels = []
    comprimentos = []
    comprimentos_por_classe = defaultdict(list)
    label_counter = Counter()

    with open(caminho_jsonl, "r", encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            doc = json.loads(linha)
            texto = doc.get("Evolucao") or doc.get("text") or ""
            texto = str(texto).strip()
            label = doc.get("Target") if doc.get("Target") is not None else doc.get("label")

            if texto == "" or label is None:
                continue

            tokens = texto.split()
            tamanho = len(tokens)

            textos.append(texto)
            labels.append(label)
            comprimentos.append(tamanho)
            comprimentos_por_classe[label].append(tamanho)
            label_counter[label] += 1

    nome_base = caminho_jsonl.stem
    output_path = output_dir / f"analise_cls_{nome_base}.txt"

    total = len(labels)
    if total == 0:
        print(f"[AVISO] Nenhum exemplo valido encontrado em {caminho_jsonl}")
        return

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"Arquivo: {caminho_jsonl.name}\n")
        out.write(f"Total de exemplos: {total}\n\n")

        out.write("Distribuicao de classes:\n")
        for classe, count in sorted(label_counter.items()):
            porcentagem = (count / total) * 100
            out.write(f" - Classe {classe}: {count} exemplos ({porcentagem:.2f}%)\n")

        balanceado = all(abs((count / total) - 0.5) < 0.2 for count in label_counter.values())
        out.write(f"\nCorpus {'balanceado' if balanceado else 'desbalanceado'}\n")

        out.write("\nEstatisticas gerais de comprimento dos textos (tokens):\n")
        out.write(f" - Media:   {np.mean(comprimentos):.2f}\n")
        out.write(f" - Mediana: {np.median(comprimentos):.2f}\n")
        out.write(f" - Minimo:  {np.min(comprimentos)}\n")
        out.write(f" - Maximo:  {np.max(comprimentos)}\n")

        out.write("\nComprimento medio por classe:\n")
        for classe, tamanhos in sorted(comprimentos_por_classe.items()):
            out.write(f" - Classe {classe}: media de {np.mean(tamanhos):.2f} tokens\n")

    print(f"[OK] Estatisticas salvas em: {output_path}")


def main():
    if len(sys.argv) != 3:
        print("Uso: python analyze_classification.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    arquivos = ["train.jsonl", "dev.jsonl", "test.jsonl", "full.jsonl"]
    for nome in arquivos:
        caminho = input_dir / nome
        if caminho.exists():
            analisar_arquivo_classificacao(caminho, output_dir)
        else:
            print(f"[AVISO] Arquivo nao encontrado: {caminho}")


if __name__ == "__main__":
    main()
