#!/bin/bash

# ==============================================================================
# Script to run comprehensive corpus analysis across all 8 ClinicalBenchPT datasets
# Outputs statistics reports, CSV summary, and LaTeX table for publication.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Define benchmark directory (can be overridden by $1)
DEFAULT_BENCHMARK="$REPO_ROOT/data/benchmark"
if [ ! -d "$DEFAULT_BENCHMARK" ] && [ -d "$REPO_ROOT/nova_data/benchmark" ]; then
  DEFAULT_BENCHMARK="$REPO_ROOT/nova_data/benchmark"
fi

BENCHMARK_DIR="${1:-$DEFAULT_BENCHMARK}"
OUTPUT_BASE_DIR="${2:-$REPO_ROOT/results/corpus_analysis}"
LOGS_DIR="$OUTPUT_BASE_DIR/logs"

mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p "$LOGS_DIR"

LOGFILE="$LOGS_DIR/analyze_corpus_$(date +'%Y-%m-%d_%H-%M').log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=================================================================="
echo "Iniciando analise estatistica do ClinicalBenchPT"
echo "=================================================================="

PYTHON_BIN="${PYTHON:-python}"
echo "Python utilizado: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"

if [ ! -d "$BENCHMARK_DIR" ]; then
  echo "[ERRO] Diretorio do benchmark nao encontrado: $BENCHMARK_DIR"
  echo "Uso: ./analyze_corpus.sh [caminho_do_benchmark] [caminho_de_saida]"
  exit 1
fi

echo "Benchmark analisado: $BENCHMARK_DIR"
echo "Saida das analises:   $OUTPUT_BASE_DIR"
echo "Logs salvos em:       $LOGFILE"
echo "------------------------------------------------------------------"

time_step() {
  local step_name=$1
  local start=$2
  local end=$3
  local duration=$((end - start))
  echo "Tempo gasto em $step_name: ${duration}s"
  echo "------------------------------------------------------------------"
}

# 1) ABCFarma QA
echo "1) Analisando abcfarma_qa..."
START=$SECONDS
$PYTHON_BIN "$SCRIPT_DIR/analyze_qa.py" "$BENCHMARK_DIR/abcfarma_qa" "$OUTPUT_BASE_DIR/abcfarma_qa"
time_step "abcfarma_qa" $START $SECONDS

# 2) Fall Detection
echo "2) Analisando fall_detection..."
START=$SECONDS
$PYTHON_BIN "$SCRIPT_DIR/analyze_classification.py" "$BENCHMARK_DIR/fall_detection" "$OUTPUT_BASE_DIR/fall_detection"
time_step "fall_detection" $START $SECONDS

# 3) Clinical NER
echo "3) Analisando clinical_ner..."
START=$SECONDS
$PYTHON_BIN "$SCRIPT_DIR/analyze_ner.py" "$BENCHMARK_DIR/clinical_ner" "$OUTPUT_BASE_DIR/clinical_ner"
time_step "clinical_ner" $START $SECONDS

# 4) DrBodeBench
echo "4) Analisando drbodebench..."
START=$SECONDS
$PYTHON_BIN "$SCRIPT_DIR/analyze_multiple_choice.py" "$BENCHMARK_DIR/drbodebench" "$OUTPUT_BASE_DIR/drbodebench"
time_step "drbodebench" $START $SECONDS

# 5) Amazon Diseases
echo "5) Analisando amazon_diseases..."
START=$SECONDS
$PYTHON_BIN "$SCRIPT_DIR/analyze_qa.py" "$BENCHMARK_DIR/amazon_diseases" "$OUTPUT_BASE_DIR/amazon_diseases"
time_step "amazon_diseases" $START $SECONDS

# 6) WikiDoc PT
echo "6) Analisando wikidoc_pt..."
START=$SECONDS
$PYTHON_BIN "$SCRIPT_DIR/analyze_qa.py" "$BENCHMARK_DIR/wikidoc_pt" "$OUTPUT_BASE_DIR/wikidoc_pt"
time_step "wikidoc_pt" $START $SECONDS

# 7) MultiClinSum PT
echo "7) Analisando multiclinsum_pt..."
START=$SECONDS
$PYTHON_BIN "$SCRIPT_DIR/analyze_summarization.py" "$BENCHMARK_DIR/multiclinsum_pt" "$OUTPUT_BASE_DIR/multiclinsum_pt"
time_step "multiclinsum_pt" $START $SECONDS

# 8) SemClinBr
echo "8) Analisando semclinbr..."
START=$SECONDS
$PYTHON_BIN "$SCRIPT_DIR/analyze_ner.py" "$BENCHMARK_DIR/semclinbr" "$OUTPUT_BASE_DIR/semclinbr"
time_step "semclinbr" $START $SECONDS

# 9) Tabela Resumo Consolidada (CSV e LaTeX)
echo "9) Gerando tabelas consolidadas para o artigo..."
$PYTHON_BIN "$SCRIPT_DIR/generate_summary_table.py" "$BENCHMARK_DIR" "$OUTPUT_BASE_DIR"

echo "=================================================================="
echo "[OK] Analise concluida com sucesso!"
echo "Relatorios detalhados em: $OUTPUT_BASE_DIR"
echo "Tabela LaTeX para o paper: $OUTPUT_BASE_DIR/benchmark_statistics_table.tex"
echo "Tabela CSV consolidada:   $OUTPUT_BASE_DIR/benchmark_statistics_summary.csv"
echo "=================================================================="
