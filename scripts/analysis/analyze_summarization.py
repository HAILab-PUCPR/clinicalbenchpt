"""
Summarization Corpus Analysis for ClinicalBenchPT (e.g. MultiClinSum PT).
Analyzes source text length, summary length, compression ratio, bucket distribution, etc.
"""

import json
import sys
import numpy as np
from pathlib import Path


def analisar_summarization_arquivo(caminho_jsonl: Path, output_dir: Path):
    text_lens = []
    summary_lens = []
    compression_ratios = []
    resumos_vazios = 0
    resumos_muito_curto = 0
    buckets_text = [0] * 6  # Faixas: 0-100, 100-200, ..., 500+
    buckets_sum = [0] * 6

    with open(caminho_jsonl, "r", encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            ex = json.loads(linha)
            texto = ex.get("text", "").strip()
            resumo = ex.get("summary", "").strip()

            len_text = len(texto.split())
            len_summary = len(resumo.split())

            text_lens.append(len_text)
            summary_lens.append(len_summary)

            if len_summary > 0:
                compression_ratios.append(len_text / len_summary)
            else:
                compression_ratios.append(0)

            if len_summary == 0:
                resumos_vazios += 1
            elif len_summary < 5:
                resumos_muito_curto += 1

            for i, limit in enumerate([100, 200, 300, 400, 500]):
                if len_text <= limit:
                    buckets_text[i] += 1
                    break
            else:
                buckets_text[-1] += 1

            for i, limit in enumerate([20, 40, 60, 80, 100]):
                if len_summary <= limit:
                    buckets_sum[i] += 1
                    break
            else:
                buckets_sum[-1] += 1

    nome_base = caminho_jsonl.stem
    output_path = output_dir / f"analise_sumarizacao_{nome_base}.txt"

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"Arquivo: {caminho_jsonl.name}\n")
        out.write(f"Total de exemplos: {len(text_lens)}\n\n")

        out.write("Comprimento dos textos (tokens):\n")
        if text_lens:
            out.write(f" - Media: {np.mean(text_lens):.2f}\n")
            out.write(f" - Mediana: {np.median(text_lens):.2f}\n")
            out.write(f" - Minimo: {np.min(text_lens)}\n")
            out.write(f" - Maximo: {np.max(text_lens)}\n\n")

        out.write("Comprimento dos resumos (tokens):\n")
        if summary_lens:
            out.write(f" - Media: {np.mean(summary_lens):.2f}\n")
            out.write(f" - Mediana: {np.median(summary_lens):.2f}\n")
            out.write(f" - Minimo: {np.min(summary_lens)}\n")
            out.write(f" - Maximo: {np.max(summary_lens)}\n\n")

        out.write("Taxa de compressao (texto/resumo):\n")
        if compression_ratios:
            out.write(f" - Media: {np.mean(compression_ratios):.2f}\n")
            out.write(f" - Mediana: {np.median(compression_ratios):.2f}\n\n")

        out.write("[AVISO] Qualidade dos resumos:\n")
        out.write(f" - Resumos vazios: {resumos_vazios}\n")
        out.write(f" - Resumos com menos de 5 tokens: {resumos_muito_curto}\n\n")

        out.write("Distribuicao dos tamanhos dos textos:\n")
        for i, count in enumerate(buckets_text):
            if i < 5:
                out.write(f" - {i*100}-{(i+1)*100} tokens: {count}\n")
            else:
                out.write(f" - >500 tokens: {count}\n")

        out.write("\nDistribuicao dos tamanhos dos resumos:\n")
        for i, count in enumerate(buckets_sum):
            if i < 5:
                out.write(f" - {i*20}-{(i+1)*20} tokens: {count}\n")
            else:
                out.write(f" - >100 tokens: {count}\n")

    print(f"[OK] Estatisticas salvas em: {output_path}")


def main():
    if len(sys.argv) != 3:
        print("Uso: python analyze_summarization.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    arquivos = ["train.jsonl", "dev.jsonl", "test.jsonl", "full.jsonl"]
    for nome in arquivos:
        caminho = input_dir / nome
        if caminho.exists():
            print(f"Analisando sumarizacao: {nome}")
            analisar_summarization_arquivo(caminho, output_dir)
        else:
            print(f"[AVISO] Arquivo nao encontrado: {caminho}")


if __name__ == "__main__":
    main()
