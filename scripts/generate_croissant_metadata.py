import os
import json
import hashlib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BENCHMARK_DIR = DATA_DIR / "benchmark"
CROISSANT_DIR = DATA_DIR / "croissant"

BENCHMARK_CONFIGS = [
    {
        "benchmark_id": "abcfarma_qa",
        "croissant_file": "croissant_abcfarma.jsonld",
        "primary_record_set": "questions",
        "primary_description": "Perguntas e respostas sobre medicamentos (completo).",
        "id_field_type": "sc:Integer",
        "fields": [
            {"id": "id", "name": "id", "dataType": "sc:Integer", "description": "Identificador único"},
            {"id": "question", "name": "question", "dataType": "sc:Text", "description": "Pergunta textual sobre medicamentos."},
            {"id": "answer", "name": "answer", "dataType": "sc:Text", "description": "Princípio ativo esperado como resposta."},
            {"id": "task", "name": "task", "dataType": "sc:Text", "description": "Tipo de tarefa (qa)."}
        ]
    },
    {
        "benchmark_id": "amazon_diseases",
        "croissant_file": "croissant_amazon_diseases.jsonld",
        "primary_record_set": "questions",
        "primary_description": "Perguntas e respostas sobre doenças negligenciadas da Amazônia (completo).",
        "id_field_type": "sc:Text",
        "fields": [
            {"id": "id", "name": "id", "dataType": "sc:Text", "description": "Identificador único"},
            {"id": "question", "name": "question", "dataType": "sc:Text", "description": "Pergunta do usuário sobre doenças negligenciadas."},
            {"id": "answer", "name": "answer", "dataType": "sc:Text", "description": "Resposta médica detalhada."},
            {"id": "task", "name": "task", "dataType": "sc:Text", "description": "Tipo de tarefa (qa)."}
        ]
    },
    {
        "benchmark_id": "clinical_ner",
        "croissant_file": "croissant_clinpt.jsonld",
        "primary_record_set": "ner-examples",
        "primary_description": "Anotações de entidades clínicas nomeadas (completo).",
        "id_field_type": "sc:Text",
        "fields": [
            {"id": "id", "name": "id", "dataType": "sc:Text", "description": "Identificador único"},
            {"id": "text", "name": "text", "dataType": "sc:Text", "description": "Texto do relatório clínico."},
            {"id": "tags", "name": "tags", "dataType": "sc:Text", "description": "Anotações de entidades nomeadas por categoria.", "special": "tags_dict"},
            {"id": "task", "name": "task", "dataType": "sc:Text", "description": "Tipo de tarefa (ner)."}
        ]
    },
    {
        "benchmark_id": "drbodebench",
        "croissant_file": "croissant_drbodebench_drb.jsonld",
        "primary_record_set": "multiple-choice-examples",
        "primary_description": "Questões de múltipla escolha para exames médicos (completo).",
        "id_field_type": "sc:Text",
        "fields": [
            {"id": "id", "name": "id", "dataType": "sc:Text", "description": "Identificador único da questão"},
            {"id": "origem", "name": "origem", "dataType": "sc:Text", "description": "Instituição de origem da prova"},
            {"id": "prova", "name": "prova", "dataType": "sc:Integer", "description": "Ano da prova"},
            {"id": "numero", "name": "numero", "dataType": "sc:Integer", "description": "Número da questão na prova"},
            {"id": "enunciado", "name": "enunciado", "dataType": "sc:Text", "description": "Enunciado da questão"},
            {"id": "alternativas", "name": "alternativas", "dataType": "sc:Text", "description": "Dicionário de alternativas de múltipla escolha.", "special": "alternativas_dict"},
            {"id": "resposta", "name": "resposta", "dataType": "sc:Text", "description": "Alternativa correta (gabarito)"},
            {"id": "contains_img", "name": "contains_img", "dataType": "sc:Boolean", "description": "Se a questão contém imagem associada"},
            {"id": "img_path", "name": "img_path", "dataType": "sc:Text", "description": "Caminho da imagem se houver"},
            {"id": "img_description", "name": "img_description", "dataType": "sc:Text", "description": "Descrição da imagem se houver"},
            {"id": "task", "name": "task", "dataType": "sc:Text", "description": "Tipo de tarefa (multiple_choice)."}
        ]
    },
    {
        "benchmark_id": "fall_detection",
        "croissant_file": "croissant_fall_detection_pt.jsonld",
        "primary_record_set": "classification-examples",
        "primary_description": "Classificação clínica de risco/ocorrência de quedas (completo).",
        "id_field_type": "sc:Text",
        "fields": [
            {"id": "id", "name": "id", "dataType": "sc:Text", "description": "Identificador do registro"},
            {"id": "Evolucao", "name": "Evolucao", "dataType": "sc:Text", "description": "Texto da evolução médica/enfermagem"},
            {"id": "Target", "name": "Target", "dataType": "sc:Integer", "description": "Rótulo de classe de queda (0 ou 1)"},
            {"id": "task", "name": "task", "dataType": "sc:Text", "description": "Tipo de tarefa (classification)."}
        ]
    },
    {
        "benchmark_id": "multiclinsum_pt",
        "croissant_file": "croissant_multiclinsum_pt.jsonld",
        "primary_record_set": "summarization-examples",
        "primary_description": "Sumarização de prontuários clínicos em português (completo).",
        "id_field_type": "sc:Text",
        "fields": [
            {"id": "id", "name": "id", "dataType": "sc:Text", "description": "Identificador único"},
            {"id": "text", "name": "text", "dataType": "sc:Text", "description": "Texto completo do caso clínico"},
            {"id": "summary", "name": "summary", "dataType": "sc:Text", "description": "Sumário clínico de referência"},
            {"id": "task", "name": "task", "dataType": "sc:Text", "description": "Tipo de tarefa (summarization)."}
        ]
    },
    {
        "benchmark_id": "semclinbr",
        "croissant_file": "croissant_semclinbr_sem.jsonld",
        "primary_record_set": "ner-examples",
        "primary_description": "Reconhecimento de entidades clínicas em prontuários SemClinBr (completo).",
        "id_field_type": "sc:Text",
        "fields": [
            {"id": "id", "name": "id", "dataType": "sc:Text", "description": "Identificador único"},
            {"id": "text", "name": "text", "dataType": "sc:Text", "description": "Texto da anotação clínica"},
            {"id": "tags", "name": "tags", "dataType": "sc:Text", "description": "Entidades clínicas anotadas.", "special": "tags_dict"},
            {"id": "task", "name": "task", "dataType": "sc:Text", "description": "Tipo de tarefa (ner)."}
        ]
    },
    {
        "benchmark_id": "wikidoc_pt",
        "croissant_file": "croissant_wikidoc_pt.jsonld",
        "primary_record_set": "questions",
        "primary_description": "Perguntas e respostas médicas da WikiDoc em português (completo).",
        "id_field_type": "sc:Text",
        "fields": [
            {"id": "id", "name": "id", "dataType": "sc:Text", "description": "Identificador único"},
            {"id": "question", "name": "question", "dataType": "sc:Text", "description": "Pergunta sobre condição médica"},
            {"id": "answer", "name": "answer", "dataType": "sc:Text", "description": "Explicação e fisiopatologia de referência"},
            {"id": "task", "name": "task", "dataType": "sc:Text", "description": "Tipo de tarefa (qa)."}
        ]
    }
]

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def count_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def generate_fields_for_split(field_defs, file_obj_id, prefix=""):
    fields = []
    for f in field_defs:
        fid = f"{prefix}/{f['id']}" if prefix else f['id']
        field_dict = {
            "@type": "cr:Field",
            "@id": fid,
            "name": f["name"],
            "description": f["description"],
            "dataType": f["dataType"]
        }
        if f.get("special") in ["tags_dict", "alternativas_dict"]:
            field_dict["source"] = {
                "fileObject": {"@id": file_obj_id},
                "extract": {"column": "task"},
                "transform": {"replace": f"*/Please refer to the '{f['id']}' field in the original dataset."}
            }
        else:
            field_dict["source"] = {
                "fileObject": {"@id": file_obj_id},
                "extract": {"column": f["id"]}
            }
        fields.append(field_dict)
    return fields

def generate_croissant_for_benchmark(config):
    b_id = config["benchmark_id"]
    croissant_fname = config["croissant_file"]
    b_dir = BENCHMARK_DIR / b_id
    
    existing_file = CROISSANT_DIR / croissant_fname
    old_data = {}
    if existing_file.exists():
        with open(existing_file, "r", encoding="utf-8") as f:
            old_data = json.load(f)
            
    # File statistics
    splits = ["train", "dev", "test", "full"]
    stats = {}
    for sp in splits:
        p = b_dir / f"{sp}.jsonl"
        if p.exists():
            stats[sp] = {
                "sha256": get_sha256(p),
                "lines": count_lines(p),
                "rel_path": f"../benchmark/{b_id}/{sp}.jsonl"
            }
            
    # Base metadata
    name = old_data.get("name", b_id)
    description = old_data.get("description", f"Clinical benchmark dataset: {b_id}")
    license_str = old_data.get("license", "Unknown")
    citation = old_data.get("citation", "Please cite the original authors.")
    url = "https://github.com/HAILab-PUCPR/clinicalbenchpt"
    keywords = old_data.get("keywords", ["clinical NLP", "Brazilian Portuguese"])
    
    # Context
    context = {
        "@language": "pt",
        "@vocab": "https://schema.org/",
        "citeAs": "cr:citeAs",
        "column": "cr:column",
        "conformsTo": "dct:conformsTo",
        "cr": "http://mlcommons.org/croissant/",
        "data": {"@id": "cr:data", "@type": "@json"},
        "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
        "dct": "http://purl.org/dc/terms/",
        "examples": {"@id": "cr:examples", "@type": "@json"},
        "extract": "cr:extract",
        "field": "cr:field",
        "fileProperty": "cr:fileProperty",
        "fileObject": "cr:fileObject",
        "fileSet": "cr:fileSet",
        "format": "cr:format",
        "includes": "cr:includes",
        "isLiveDataset": "cr:isLiveDataset",
        "jsonPath": "cr:jsonPath",
        "key": "cr:key",
        "md5": "cr:md5",
        "parentField": "cr:parentField",
        "path": "cr:path",
        "rai": "http://mlcommons.org/croissant/RAI/",
        "recordSet": "cr:recordSet",
        "references": "cr:references",
        "regex": "cr:regex",
        "repeated": "cr:repeated",
        "replace": "cr:replace",
        "sc": "https://schema.org/",
        "separator": "cr:separator",
        "source": "cr:source",
        "subField": "cr:subField",
        "transform": "cr:transform",
        "wd": "https://www.wikidata.org/wiki/"
    }
    
    # Distribution
    distributions = []
    
    # Preserve original source repo if existed
    for old_dist in old_data.get("distribution", []):
        if old_dist.get("@id") == "source-repository":
            distributions.append(old_dist)
            break
            
    for sp in ["full", "train", "dev", "test"]:
        if sp in stats:
            distributions.append({
                "@type": "cr:FileObject",
                "@id": f"{sp}.jsonl",
                "name": f"{sp}.jsonl",
                "contentUrl": stats[sp]["rel_path"],
                "encodingFormat": "application/jsonlines",
                "description": f"Split '{sp}' com {stats[sp]['lines']} registros.",
                "sha256": stats[sp]["sha256"]
            })
            
    # RecordSets
    record_sets = []
    
    # 1. Primary record set (e.g. questions, ner-examples, pointing to full.jsonl or train.jsonl)
    primary_id = config["primary_record_set"]
    target_full_obj = "full.jsonl" if "full" in stats else "train.jsonl"
    record_sets.append({
        "@type": "cr:RecordSet",
        "@id": primary_id,
        "name": primary_id,
        "description": config["primary_description"],
        "key": {"@id": "id"},
        "field": generate_fields_for_split(config["fields"], target_full_obj, prefix=primary_id)
    })
    
    # 2. Split record sets (train, dev, test)
    for sp in ["train", "dev", "test"]:
        if sp in stats:
            record_sets.append({
                "@type": "cr:RecordSet",
                "@id": sp,
                "name": sp,
                "description": f"Conjunto de dados para o split '{sp}'.",
                "key": {"@id": "id"},
                "field": generate_fields_for_split(config["fields"], f"{sp}.jsonl", prefix=sp)
            })
            
    doc = {
        "@context": context,
        "@type": "sc:Dataset",
        "name": name,
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "version": "1.0.0",
        "datePublished": "2025-11-24",
        "license": license_str,
        "description": description,
        "url": url,
        "citeAs": citation,
        "citation": citation,
        "keywords": keywords,
        "distribution": distributions,
        "recordSet": record_sets
    }
    
    out_path = CROISSANT_DIR / croissant_fname
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
        
    print(f"Generated {out_path.name}: {len(record_sets)} recordSets, {len(distributions)} distributions")

def main():
    CROISSANT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating Croissant metadata files into {CROISSANT_DIR}...")
    for cfg in BENCHMARK_CONFIGS:
        generate_croissant_for_benchmark(cfg)
    print("Done!")

if __name__ == "__main__":
    main()
