# Croissant Metadata Files

This directory contains [Croissant metadata](http://mlcommons.org/croissant/) files (`JSON-LD`) for all 8 benchmark datasets in the ClinicalBenchPT benchmark suite. Croissant is an open-source metadata format standard developed by MLCommons for describing machine learning datasets.

## 📋 What is Croissant?

Croissant provides a standardized way to describe datasets, enabling:
- **Discoverability**: Better indexing in dataset search engines (e.g., Google Dataset Search)
- **Interoperability**: Easy loading with standard tools and libraries (PyTorch, Hugging Face, TensorFlow)
- **Reproducibility**: Complete documentation of data structure, exact SHA-256 hashes, and schema definitions
- **Transparency**: Clear information about licensing, task types, splits, and ethical considerations

Learn more at: [mlcommons.org/croissant](http://mlcommons.org/croissant/)

---

## 📁 Available Metadata Files

| Benchmark Dataset | Croissant Metadata File | Task Type | Splits Available |
|---|---|---|---|
| **ABCFarma-QA** | [`croissant_abcfarma.jsonld`](croissant_abcfarma.jsonld) | Question Answering (QA) | `full`, `train`, `dev`, `test` |
| **AmazonDiseases-QA** | [`croissant_amazon_diseases.jsonld`](croissant_amazon_diseases.jsonld) | Question Answering (QA) | `full`, `train`, `dev`, `test` |
| **PortugueseClinicalNER (ClinPT)** | [`croissant_clinpt.jsonld`](croissant_clinpt.jsonld) | Named Entity Recognition (NER) | `full`, `train`, `dev`, `test` |
| **DrBodeBench-DRB** | [`croissant_drbodebench_drb.jsonld`](croissant_drbodebench_drb.jsonld) | Multiple Choice QA | `full`, `train`, `dev`, `test` |
| **FallDetection-PT** | [`croissant_fall_detection_pt.jsonld`](croissant_fall_detection_pt.jsonld) | Binary Classification | `full`, `train`, `dev`, `test` |
| **MultiClinSum-PT** | [`croissant_multiclinsum_pt.jsonld`](croissant_multiclinsum_pt.jsonld) | Summarization | `full`, `train`, `dev`, `test` |
| **SemClinBr-SEM** | [`croissant_semclinbr_sem.jsonld`](croissant_semclinbr_sem.jsonld) | Named Entity Recognition (NER) | `full`, `train`, `dev`, `test` |
| **Wikidoc-PT-QA** | [`croissant_wikidoc_pt.jsonld`](croissant_wikidoc_pt.jsonld) | Medical QA | `full`, `train`, `dev`, `test` |

---

## 🚀 Loading Datasets with `mlcroissant`

### 1. Installation
```bash
pip install mlcroissant
```

### 2. Loading Records by Split or Complete Benchmark
Each Croissant file provides both task-level default record sets and split-level record sets (`train`, `dev`, `test`):

```python
import mlcroissant as mlc

# Load from local Croissant metadata file
dataset = mlc.Dataset("data/croissant/croissant_abcfarma.jsonld")

# Available recordSets: ['questions', 'train', 'dev', 'test']
print("RecordSets:", [rs.id for rs in dataset.metadata.record_sets])

# Load training split
for record in dataset.records("train"):
    print("ID:", record["train/id"])
    print("Question:", record["train/question"])
    print("Answer:", record["train/answer"])
    break

# Load development/validation split
for record in dataset.records("dev"):
    print("Dev record:", record)
    break

# Load complete benchmark (full)
for record in dataset.records("questions"):
    print("Full record:", record)
    break
```

---

## 🛠️ Regenerating / Updating Metadata

To re-compute hashes and update all Croissant files whenever the benchmark files in `data/benchmark/` are modified, run:

```bash
python scripts/generate_croissant_metadata.py
```
