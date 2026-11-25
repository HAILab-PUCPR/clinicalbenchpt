"""
Corpus analysis script for Question Answering (QA) datasets.
Analyzes token lengths for both questions and answers across splits.
"""

import json
import sys
import numpy as np
from pathlib import Path


def analisar_qa(caminho_jsonl):
    tamanhos_perguntas = []
    tamanhos_respostas = []

    with open(caminho_jsonl, "r", encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            exemplo = json.loads(linha)

            pergunta = exemplo.get("question") or exemplo.get("Question") or ""
            if pergunta.strip():
                tamanhos_perguntas.append(len(pergunta.strip().split()))

            resposta = exemplo.get("answer") or exemplo.get("Response") or ""
            if resposta.strip():
                tamanhos_respostas.append(len(resposta.strip().split()))

    return tamanhos_perguntas, tamanhos_respostas


def salvar_estatisticas(nome_arquivo, tamanhos_p, tamanhos_r, output_dir):
    if not tamanhos_p and not tamanhos_r:
        print(f"[AVISO] Nenhum exemplo encontrado em {nome_arquivo}.")
        return

    output_path = output_dir / f"analise_qa_{nome_arquivo}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Arquivo: {nome_arquivo}.jsonl\n")
        f.write(f"Total de exemplos: {max(len(tamanhos_p), len(tamanhos_r))}\n\n")

        if tamanhos_p:
            f.write("Comprimento das Perguntas (tokens/palavras):\n")
            f.write(f" - Media:         {np.mean(tamanhos_p):.2f}\n")
            f.write(f" - Mediana:       {np.median(tamanhos_p):.2f}\n")
            f.write(f" - Desvio Padrao: {np.std(tamanhos_p):.2f}\n")
            f.write(f" - Minimo:        {int(np.min(tamanhos_p))}\n")
            f.write(f" - Maximo:        {int(np.max(tamanhos_p))}\n\n")

        if tamanhos_r:
            f.write("Comprimento das Respostas de Referencia (tokens/palavras):\n")
            f.write(f" - Media:         {np.mean(tamanhos_r):.2f}\n")
            f.write(f" - Mediana:       {np.median(tamanhos_r):.2f}\n")
            f.write(f" - Desvio Padrao: {np.std(tamanhos_r):.2f}\n")
            f.write(f" - Minimo:        {int(np.min(tamanhos_r))}\n")
            f.write(f" - Maximo:        {int(np.max(tamanhos_r))}\n")

    print(f"[OK] Estatisticas salvas em: {output_path}")


def main():
    if len(sys.argv) != 3:
        print("Uso: python analyze_qa.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    arquivos = ["train.jsonl", "dev.jsonl", "test.jsonl", "full.jsonl"]
    for nome_arquivo in arquivos:
        caminho_jsonl = input_dir / nome_arquivo
        if not caminho_jsonl.exists():
            print(f"[AVISO] Arquivo nao encontrado: {caminho_jsonl}")
            continue

        tam_p, tam_r = analisar_qa(caminho_jsonl)
        salvar_estatisticas(nome_arquivo.replace(".jsonl", ""), tam_p, tam_r, output_dir)


if __name__ == "__main__":
    main()
