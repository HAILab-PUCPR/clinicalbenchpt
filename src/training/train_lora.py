"""
LoRA Fine-Tuning Script: Trains LoRA adapters on clinical benchmark datasets.

Supports two precision modes:
1. Native bfloat16 (Default, recommended for NVIDIA H100, A100 or GPUs with >= 24GB VRAM):
   - Faster execution (no dequantization overhead).
   - Numerically stable and native to Tensor Cores.
2. 4-bit QLoRA (Optional via --use-4bit flag, recommended for smaller GPUs with 12GB - 16GB VRAM):
   - Uses bitsandbytes NF4 quantization to minimize memory footprint.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# -------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# PROMPT FORMATTERS
# -------------------------------------------------------------

def generate_prompt_drbodebench() -> str:
    return (
        "Você é um especialista na área de Medicina, resolvendo uma questão de uma prova de alto nível. "
        "Seu objetivo é demonstrar domínio do assunto, informando na resposta a letra da alternativa correta (dentre A, B, C e D). "
        "Sua resposta deve seguir estritamente o formato:\n\n"
        "Resposta: <LETRA>.\n"
        "Breve justificativa: <após informar a letra, dê uma breve justificativa da escolha baseada no conteúdo da questão, sem adicionar suposições>"
    )

def generate_prompt_clinical_ner() -> str:
    entities = {
        "Condição": 'Condição clínica ou diagnóstico. Engloba doenças, sintomas ou condições clínicas. Exemplos: "dor", "febre", "infecção aguda", "otite", "comichão grave".',
        "Anatomia": 'Local anatômico ou partes do corpo humano mencionado no contexto clínico. Exemplos: "orelha esquerda", "região", "cabeça", "abdómen", "região do joelho".',
        "Terapia": 'Terapia, tratamentos, medicamento ou intervenção clínica aplicada. Exemplos: "paracetamol", "antibiótico", "uso de ibuprofeno", "tratamento tópico", "analgésico".',
        "Procedimento": 'Exame médico, teste diagnóstico ou procedimento de imagem. Exemplos: "TC cerebral", "ressonância", "hemograma", "eletrocardiograma", "endoscopia".',
        "Evolução": 'Mudança, egresso ou evolução do quadro clínico ao longo do tempo. Exemplos: "agravamento", "melhora", "persistência", "regressão", "piora".',
        "Genética": 'Fator ou marcador genético associado à condição. Exemplos: "BRCA1", "mutação genética", "gene CFTR", "deleção 22q11", "fator hereditário".',
        "Negação": 'Negação de condição, evento ou sintoma. Exemplos: "sem", "não apresenta dor", "ausência de sinais", "afebril", "sem alterações".',
        "Observação": 'Observações adicionais que não se enquadram em outras categorias. Exemplos: "virar a calça", "realizar ações complexas", "apenas com a mão direita".',
        "Resultado": 'Resultado de exames ou observações objetivas. Exemplos: "normal", "alterado", "positivo", "dentro dos parâmetros".',
        "Tempo": 'Expressões temporais indicando duração ou data de sintomas/eventos. OBS: Idade do paciente não entra aqui. Exemplos: "19 meses", "há três dias", "hoje", "desde ontem".',
        "Valor": 'Valor numérico relevante clinicamente. Medidas numéricas associadas a sinais vitais ou exames (ex.: dosagem, temperatura, pressão). OBS: Idade do paciente não entra aqui. Exemplos: "38ºC", "120/80", "98%", "70 bpm", "5 mg".',
        "ViaAdmin": 'Via de administração de medicamento ou terapia. Exemplos: "via oral", "intravenosa", "tópica", "intramuscular", "subcutânea".',
        "Característica": 'Característica ou descrição que qualifica uma condição clínica. Exemplos: "associado", "diminuição", "semelhante", "recorrente", "tátil".',
    }
    entities_texto = "\n".join([f"{k}: {v}" for k, v in entities.items()])
    return f"""Você é um reconhecedor de entidades nomeadas (NER) especializado em textos clínicos. Sua tarefa é identificar e rotular entidades no texto fornecido.

        As únicas categorias de entidade permitidas são as seguintes: 
        {entities_texto}

        Regras para a resposta:
        - Você deve identificar as entidades no texto e agrupá-las por categoria.
        - A resposta deve ter o seguinte formato:
            "Anatomia: entidade1; entidade2; entidade3
            Condição: entidade1; entidade2
            Resultado: entidade1"
        - Se nenhuma entidade de uma classe estiver presente no texto, não inclua essa classe na resposta.
        - Use os termos exatamente como aparecem no texto, sem corrigir, reescrever, normalizar ou expandir abreviação.
        - Sua resposta deve conter apenas a lista estruturada das entidades. Não inclua explicações ou comentários.
        - OBS: Cada entidade pode ser formada por um ou mais tokens."""

def generate_prompt_semclinbr() -> str:
    entities = {
        "Disorder": "Doenças, condições clínicas, sinais e sintomas que indicam alterações no estado de saúde. Exemplos: 'sepse', 'dor torácica', 'febre', 'dispneia', 'anemia', 'edema'.",
        "Procedures": "Intervenções e cuidados realizados para diagnóstico, tratamento, prevenção ou acompanhamento. Exemplos: 'angioplastia', 'traqueostomia', 'curativo', 'alta hospitalar', 'fisioterapia respiratória'.",
        "Chemicals and Drugs": "Substâncias químicas, medicamentos e fármacos usados no tratamento, prevenção ou diagnóstico. Exemplos: 'paracetamol', 'adrenalina', 'sinvastatina', 'AAS', 'furosemida'.",
        "Abbreviation": "Siglas e abreviações comuns em registros clínicos. Exemplos: 'PA', 'FC', 'MMII', 'UTI', 'VM', 'HAS'."
    }
    entities_texto = "\n".join([f"{k}: {v}" for k, v in entities.items()])
    return f"""Você é um reconhecedor de entidades nomeadas (NER) especializado em textos clínicos. Sua tarefa é identificar e rotular entidades no texto fornecido.

        As únicas categorias de entidade permitidas são as seguintes: 
        {entities_texto}

        Regras para a resposta:
        - Você deve identificar as entidades no texto e agrupá-las por categoria.
        - A resposta deve ter o seguinte formato:
            "Disorder: entidade1; entidade2; entidade3
            Procedures: entidade1; entidade2
            Chemicals and Drugs: entidade1; 
            Abbreviation: entidade1; entidade2; etc;
        - Se nenhuma entidade de uma classe estiver presente no texto, não inclua essa classe na resposta.
        - Use os termos exatamente como aparecem no texto, sem corrigir, reescrever, normalizar ou expandir abreviação.
        - Sua resposta deve conter apenas a lista estruturada das entidades. Não inclua explicações ou comentários.
        - OBS: Cada entidade pode ser formada por um ou mais tokens."""

def format_prompt(example: Dict, dataset_name: str) -> Tuple[List[Dict], str, Any, str]:
    original_id = example["id"]

    if dataset_name == "amazon_diseases":
        system_prompt = (
            "Você é um assistente de saúde virtual. Responda à pergunta a seguir de forma clara, "
            "objetiva e em um único parágrafo de texto corrido. Sua resposta deve ter aproximadamente 65 palavras. "
            "Não utilize listas, tópicos ou bullet points."
        )
        question = example.get("question") or example.get("Question")
        answer = example.get("answer") or example.get("Response")
        user_prompt = f"Pergunta: {question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer}
        ]
        return messages, original_id, answer, question
        
    elif dataset_name == "wikidoc_pt":
        system_prompt = (
           "Você é um especialista em medicina. Sua tarefa é responder à pergunta a seguir de forma detalhada, "
           "informativa e abrangente, mantendo um tom técnico e preciso. "
           "A resposta deve ser apresentada em um parágrafo único e coeso, com aproximadamente 120 palavras. "
           "Não use listas ou marcadores."
        )
        question = example["question"]
        user_prompt = f"Pergunta: {question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": example["answer"]},
        ]
        return messages, original_id, example["answer"], question

    elif dataset_name == "abcfarma_qa":
        system_prompt = (
            "Você é um assistente especializado em medicamentos. Dada uma pergunta sobre um medicamento, "
            "identifique seu(s) princípio(s) ativo(s). "
            "Instruções de Formatação:"
            "1. Responda exclusivamente com o nome da substância."
            "2. Não inclua nenhuma palavra ou frase adicional. "
            "3. Se houver mais de um princípio ativo, separe-os com um sinal de '+' sem espaços."
        )
        question = example["question"]
        user_prompt = f"Pergunta: {question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": example["answer"]},
        ]
        return messages, original_id, example["answer"], question

    elif dataset_name == "multiclinsum_pt":
        system_prompt = (
            "Você é um especialista em redação médica. Gere um resumo de um caso clínico destacando diagnóstico"
            "principal, achados clínicos relevantes, contexto do paciente (idade, sexo,antecedentes importantes),"
            "conduta médica e evolução do caso. "
            "O resumo deve ser claro, conciso, técnico e ter 20-25% do comprimento do texto original. "
            "Evite listas, marcadores ou interpretações subjetivas."
        )
        text = example["text"]
        user_prompt = f"Caso clínico: {text}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": example["summary"]}
        ]
        return messages, original_id, example["summary"], text

    elif dataset_name == "fall_detection":
        system_prompt = (
            "Você é um especialista em saúde clínica. Dada a descrição da evolução médica de um paciente, "
            " classifique se ele sofreu queda (classe 1) ou não (classe 0)."
            " Forneça a resposta final seguindo estritamente este formato:\n Queda: <0 ou 1>.\n"
        )
        text = example["Evolucao"]
        user_prompt = f"Evolução clínica: {text}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Queda: " + str(example["Target"]) + "."}
        ]
        return messages, original_id, str(example["Target"]), text
        
    elif dataset_name == "drbodebench":
        system_prompt = generate_prompt_drbodebench()
        enunciado = "Questão: " + example['enunciado']
        alternativas = example["alternativas"]
        descricao_imagem = example.get("img_description", "Nenhuma imagem fornecida.")
        texto_alternativas = "\n".join([f"{letra}) {texto}" for letra, texto in alternativas.items()])
        enunciado = enunciado + f"\nDescrição da Imagem:\n{descricao_imagem}\nAlternativas Apresentadas:\n{texto_alternativas}."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enunciado},
            {"role": "assistant", "content": "Resposta: " + str(example["resposta"]) + '.'}
        ]
        return messages, original_id, str(example["resposta"]), enunciado
        
    elif dataset_name == "clinical_ner":
        system_prompt = generate_prompt_clinical_ner()
        text = example["text"]
        tags_dict = example['tags']
        tags_lines = []
        for category, entities in tags_dict.items():
            if entities:  
                entities_str = "; ".join(entities)
                tags_lines.append(f"{category}: {entities_str}")
        tags_str = "\n".join(tags_lines)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
            {"role": "assistant", "content": tags_str},
        ]
        return messages, original_id, tags_dict, text
        
    elif dataset_name == "semclinbr":
        system_prompt = generate_prompt_semclinbr()
        text = example["text"]
        tags_dict = example['tags']
        tags_lines = []
        for category, entities in tags_dict.items():
            if entities:  
                entities_str = "; ".join(entities)
                tags_lines.append(f"{category}: {entities_str}")
        tags_str = "\n".join(tags_lines)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
            {"role": "assistant", "content": tags_str},
        ]
        return messages, original_id, tags_dict, text
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def format_prompt_for_lora(example, dataset_name):
    messages, _id, reference, text = format_prompt(example, dataset_name)
    return {"messages": messages}

# -------------------------------------------------------------
# TRAINING FUNCTION
# -------------------------------------------------------------

def train_lora(
    model_name: str,
    dataset_name: str,
    data_dir: Path,
    output_dir: Path,
    epochs: int = 3,
    use_4bit: bool = False
):
    precision_mode = "4-bit QLoRA (bitsandbytes)" if use_4bit else "Native bfloat16 (high performance)"
    logger.info(f"🚀 Starting LoRA training for {dataset_name} with {model_name} | Mode: {precision_mode}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # Ensure thinking mode is disabled by default for reasoning tokenizers (e.g. Qwen)
    if getattr(tokenizer, "chat_template", None) and "enable_thinking" in tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "enable_thinking | default(true)", "enable_thinking | default(false)"
        ).replace(
            "enable_thinking|default(true)", "enable_thinking|default(false)"
        )

    # Seleção de modo de precisão
    if use_4bit:
        # Modo QLoRA 4-bit (útil para quem reproduzir em GPUs com menor VRAM <= 16GB)
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        
        logger.info("Configuring 4-bit NF4 BitsAndBytes quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # Modo bfloat16 nativo (padrão, recomendado para NVIDIA H100 e GPUs com >= 24GB VRAM)
        logger.info("Loading base model in native bfloat16 (no bitsandbytes quantization overhead)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.train()

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "dev.jsonl"),
        },
    )

    train_ds = dataset["train"].map(
        lambda ex: format_prompt_for_lora(ex, dataset_name)
    )

    val_ds = dataset["validation"].map(
        lambda ex: format_prompt_for_lora(ex, dataset_name)
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        logging_steps=10,
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        save_strategy="no",
        eval_strategy="epoch",
        report_to="none",
        dataset_text_field="messages",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()

    logger.info("💾 Saving LoRA adapter and tokenizer...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    logger.info("🎉 LoRA training completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter on ClinicalBenchPT datasets")
    parser.add_argument("--model-name", type=str, default="google/medgemma-4b-it", help="Base model HuggingFace ID")
    parser.add_argument("--dataset-name", required=True, help="ClinicalBenchPT dataset name")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to dataset directory containing train.jsonl and dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("../../models/lora"), help="Output directory for adapters")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        default=False,
        help="Enable 4-bit QLoRA via bitsandbytes (default: False, runs in native bfloat16 for maximum performance on H100/A100)"
    )

    args = parser.parse_args()
    model_tag = args.model_name.replace("/", "_")

    outdir = args.output_dir / args.dataset_name / model_tag
    outdir.mkdir(parents=True, exist_ok=True)

    train_lora(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        output_dir=outdir,
        epochs=args.epochs,
        use_4bit=args.use_4bit,
    )

if __name__ == "__main__":
    main()
