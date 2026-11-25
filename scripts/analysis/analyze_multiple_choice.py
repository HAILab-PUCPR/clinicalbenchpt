"""
Multiple Choice Corpus Analysis for ClinicalBenchPT (e.g. DrBodeBench).
Analyzes question length, option length, option counts, answer distribution, etc.
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

MIN_ALT = 2  # Minimo aceitavel de alternativas por questao


def analisar_mc_arquivo(caminho_jsonl: Path, output_dir: Path):
    total_questoes = 0
    respostas_corretas = Counter()
    enunciado_tokens = []
    enunciado_tokens_por_resposta = defaultdict(list)
    tem_imagem = 0
    alt_faltando = 0
    alt_repetidas = 0
    alt_vazias = 0
    alt_vazias_por_letra = Counter()

    estrutura_invalida = 0
    tokens_por_alternativa = []
    quantidade_alternativas = Counter()

    with open(caminho_jsonl, "r", encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            ex = json.loads(linha)
            total_questoes += 1

            if not all(chave in ex for chave in ("enunciado", "alternativas", "resposta")):
                estrutura_invalida += 1
                continue

            enunciado = ex["enunciado"]
            alternativas = ex.get("alternativas")
            resposta = ex["resposta"]

            if not isinstance(alternativas, dict):
                estrutura_invalida += 1
                continue

            alt_keys = set(alternativas.keys())
            quantidade_alternativas[len(alt_keys)] += 1

            if len(alt_keys) < MIN_ALT:
                alt_faltando += 1

            alt_values = [v if isinstance(v, str) else "" for v in alternativas.values()]
            if len(set(alt_values)) < len(alt_values):
                alt_repetidas += 1

            for letra, alt_text in alternativas.items():
                if not isinstance(alt_text, str):
                    alt_text = ""
                else:
                    alt_text = alt_text.strip()
                if alt_text == "":
                    alt_vazias += 1
                    alt_vazias_por_letra[letra] += 1
                tokens_por_alternativa.append(len(alt_text.split()))

            en_tokens_len = len(enunciado.strip().split())
            enunciado_tokens.append(en_tokens_len)
            enunciado_tokens_por_resposta[resposta].append(en_tokens_len)
            respostas_corretas[resposta] += 1

            if ex.get("contains_img", False):
                tem_imagem += 1

    nome_base = caminho_jsonl.stem
    output_path = output_dir / f"analise_mc_{nome_base}.txt"

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"Arquivo: {caminho_jsonl.name}\n")
        out.write(f"Total de questoes: {total_questoes}\n\n")

        out.write("[OK] Contagem e validacao das alternativas:\n")
        out.write(f" - Questoes com menos de {MIN_ALT} alternativas: {alt_faltando}\n")
        out.write(f" - Questoes com alternativas repetidas: {alt_repetidas}\n")
        out.write(f" - Alternativas vazias: {alt_vazias}\n\n")

        out.write("Alternativas vazias por letra:\n")
        for letra in sorted(alt_vazias_por_letra.keys()):
            out.write(f" - {letra}: {alt_vazias_por_letra[letra]} vazias\n")

        out.write("\nDistribuicao da quantidade de alternativas por questao:\n")
        for qtd, contagem in sorted(quantidade_alternativas.items()):
            out.write(f" - {qtd} alternativas: {contagem} questoes\n")

        out.write("\nDistribuicao das respostas corretas:\n")
        for letra in sorted(respostas_corretas):
            pct = (respostas_corretas[letra] / total_questoes) * 100 if total_questoes > 0 else 0
            out.write(f" - {letra}: {respostas_corretas[letra]} ({pct:.2f}%)\n")

        out.write("\nComprimento medio dos enunciados (tokens):\n")
        if enunciado_tokens:
            out.write(f" - Media geral:   {np.mean(enunciado_tokens):.2f}\n")
            out.write(f" - Mediana:       {np.median(enunciado_tokens):.2f}\n")
            out.write(f" - Minimo:        {np.min(enunciado_tokens)}\n")
            out.write(f" - Maximo:        {np.max(enunciado_tokens)}\n")

        out.write("\nComprimento dos enunciados por resposta correta:\n")
        for letra in sorted(enunciado_tokens_por_resposta):
            media = np.mean(enunciado_tokens_por_resposta[letra])
            out.write(f" - {letra}: media de {media:.2f} tokens\n")

        out.write(f"\nPresenca de imagens:\n")
        out.write(f" - Questoes com imagem: {tem_imagem}\n")

        out.write("\nComprimento das alternativas (tokens):\n")
        if tokens_por_alternativa:
            out.write(f" - Media:  {np.mean(tokens_por_alternativa):.2f}\n")
            out.write(f" - Desvio: {np.std(tokens_por_alternativa):.2f}\n")
            out.write(f" - Min:    {np.min(tokens_por_alternativa)}\n")
            out.write(f" - Max:    {np.max(tokens_por_alternativa)}\n")

        out.write(f"\n[AVISO] Questoes com estrutura invalida: {estrutura_invalida}\n")

    print(f"[OK] Estatisticas salvas em: {output_path}")


def main():
    if len(sys.argv) != 3:
        print("Uso: python analyze_multiple_choice.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    arquivos = ["train.jsonl", "dev.jsonl", "test.jsonl", "full.jsonl"]
    for nome in arquivos:
        caminho = input_dir / nome
        if caminho.exists():
            print(f"Analisando o corpus {nome}...")
            analisar_mc_arquivo(caminho, output_dir)
        else:
            print(f"[AVISO] Arquivo nao encontrado: {caminho}")


if __name__ == "__main__":
    main()
