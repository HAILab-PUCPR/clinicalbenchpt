"""
Lora Inference Script: Evaluates LLMs on benchmark datasets using lora fine-tuning.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default hyperparameter values
BATCH_SIZE = 8
MAX_TOKENS = 512
MAX_TOKENS_EXTENDED = 1024  # For summarization

# Dataset-specific reference keys mapping
DATASET_REFERENCE_KEYS = {
    "amazon_diseases": "answer",
    "wikidoc_pt": "answer",
    "abcfarma_qa": "answer",
    "multiclinsum_pt": "summary",
    "fall_detection": "Target",
    "drbodebench": "resposta",
    "semclinbr": "tags",
    "clinical_ner": "tags",
}

def generate_prompt_drbodebench() -> str:
    """Generate system prompt for DrBodeBench multiple choice questions."""
    return (
        "Você é um especialista na área de Medicina, resolvendo uma questão de uma prova de alto nível. "
        "Seu objetivo é demonstrar domínio do assunto, informando na resposta a letra da alternativa correta (dentre A, B, C e D). "
        "Sua resposta deve seguir estritamente o formato:\n\n"
        "Resposta: <LETRA>.\n"
        "Breve justificativa: <após informar a letra, dê uma breve justificativa da escolha baseada no conteúdo da questão, sem adicionar suposições>"
    )

def extract_thinking_content(text: str) -> Tuple[str, str]:
    """
    Extract thinking content from response if present and return clean response.
    Handles both complete <think>...</think> blocks and unclosed <think>... tags.
    """
    thinking_content = None
    if not isinstance(text, str):
        return text, None

    # Try to extract complete <think>...</think> tags
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        thinking_content = match.group(1).strip()
        clean_response = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return clean_response, thinking_content

    # Handle unclosed <think> tag (e.g. truncated generation)
    if "<think>" in text:
        parts = text.split("<think>", 1)
        clean_response = parts[0].strip()
        thinking_content = parts[1].strip()
        return clean_response, thinking_content

    return text.strip(), None

def generate_prompt_clinical_ner() -> str:
    """Generate system prompt for Clinical NER Custom dataset."""
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
    """Generate system prompt for SemClinBr NER dataset."""
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
    """
    Generate chat messages, ID, and reference based on dataset type.
    Returns: (messages, original_id, reference, text)
    """
    original_id = example["id"]
    
    reference_key = DATASET_REFERENCE_KEYS.get(dataset_name)
    if not reference_key:
        raise ValueError(f"Reference key not defined for dataset: {dataset_name}")
    reference = example[reference_key]

    if dataset_name == "amazon_diseases":
        system_prompt = (
            "Você é um assistente de saúde virtual. Responda à pergunta a seguir de forma clara, "
            "objetiva e em um único parágrafo de texto corrido. Sua resposta deve ter aproximadamente 65 palavras. "
            "Não utilize listas, tópicos ou bullet points."
        )
        
        question = example.get("question") or example.get("Question")
        user_prompt = f"Pergunta: {question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages, original_id, reference, question
        
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
            {"role": "user", "content": user_prompt}
        ]
        return messages, original_id, reference, question

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
            {"role": "user", "content": user_prompt}
        ]
        return messages, original_id, reference, question

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
            {"role": "user", "content": user_prompt}
        ]
        return messages, original_id, reference, text

    elif dataset_name == "fall_detection":
        system_prompt = (
            "Você é um especialista em saúde clínica. Dada a descrição da evolução médica de um paciente, "
            " classifique se ele sofreu queda (classe 1) ou não (classe 0)."
            " Forneça a resposta final seguindo estritamente este formato:\n Queda: <0 ou 1>.\n"
            " Justificativa: <uma breve justificativa, objetiva e concisa baseada nas informações clínicas fornecidas, sem fazer suposições ou usar linguagem especulativa>."
        )
        
        text = example["Evolucao"]
        user_prompt = f"Evolução clínica: {text}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages, original_id, str(reference), text
        
    elif dataset_name == "drbodebench":
        system_prompt = generate_prompt_drbodebench()
        enunciado = "Questão: " + example['enunciado']
        alternativas = example["alternativas"]
        descricao_imagem = example.get("img_description", "Nenhuma imagem fornecida.")
        texto_alternativas = "\n".join([f"{letra}) {texto}" for letra, texto in alternativas.items()])
        enunciado = enunciado + f"\nDescrição da Imagem:\n{descricao_imagem}\nAlternativas Apresentadas:\n{texto_alternativas}."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enunciado}
        ]
        return messages, original_id, str(reference), enunciado
        
    elif dataset_name == "clinical_ner":
        system_prompt = generate_prompt_clinical_ner()
        text = example["text"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        return messages, original_id, reference, text
        
    elif dataset_name == "semclinbr":
        system_prompt = generate_prompt_semclinbr()
        text = example["text"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        return messages, original_id, reference, text
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records

def setup_model_and_tokenizer(model_path_or_name: str):
    """Load model and tokenizer (supports base models and PEFT/LoRA adapter directories)."""
    logger.info(f"Loading model/adapter from: {model_path_or_name}")
    
    adapter_cfg_file = Path(model_path_or_name) / "adapter_config.json"
    
    if adapter_cfg_file.exists():
        with open(adapter_cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        base_model_name = cfg.get("base_model_name_or_path")
        logger.info(f"LoRA adapter detected. Base model: {base_model_name}")
        
        # Load tokenizer from adapter dir if exists, else from base model
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path_or_name,
                trust_remote_code=True,
                padding_side='left'
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                padding_side='left'
            )
            
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, model_path_or_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path_or_name,
            trust_remote_code=True,
            padding_side='left'
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer

def run_batch_inference(
    model,
    tokenizer,
    dataset: List[Dict],
    dataset_name: str,
    output_path: Path,
    batch_size: int = BATCH_SIZE,
    max_tokens: int = MAX_TOKENS,
    use_thinking: bool = False,
):
    """Run batch inference on dataset."""
    logger.info(f"Starting inference on {len(dataset)} samples")
    logger.info(f"Batch size: {batch_size}, Max tokens: {max_tokens}, Thinking mode: {use_thinking}")
    
    results = []
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for start_idx in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
            batch_data = dataset[start_idx:start_idx + batch_size]
            
            prompts_text_batch = []
            batch_metadata = []
            
            for item in batch_data:
                try:
                    messages, item_id, reference, text = format_prompt(item, dataset_name)
                    
                    try:
                        chat_formatted_prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=use_thinking,
                        )
                    except TypeError:
                        # Fallback for models/tokenizers that do not support enable_thinking
                        try:
                            chat_formatted_prompt = tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                        except Exception:
                            full_prompt = "\n".join(
                                f"{m.get('role', '')}: {m.get('content', '')}"
                                for m in messages
                            )
                            chat_formatted_prompt = full_prompt
                    except Exception:
                        full_prompt = "\n".join(
                            f"{m.get('role', '')}: {m.get('content', '')}"
                            for m in messages
                        )
                        chat_formatted_prompt = full_prompt
                    
                    prompts_text_batch.append(chat_formatted_prompt)
                    batch_metadata.append({
                        'id': item_id,
                        'reference': reference,
                        'text': text,
                    })
                    
                except Exception as e:
                    logger.error(f"Error formatting prompt for item {item.get('id')}: {e}")
                    continue
            
            if not prompts_text_batch:
                continue
            
            try:
                inputs = tokenizer(
                    prompts_text_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(model.device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False
                )
                
                input_ids_length = inputs.input_ids.shape[1]
                generated_tokens_batch = outputs[:, input_ids_length:]
                generated_texts = tokenizer.batch_decode(
                    generated_tokens_batch,
                    skip_special_tokens=True
                )
                
                for i, generated_text in enumerate(generated_texts):
                    clean_pred, thinking_content = extract_thinking_content(generated_text)
                    result = {
                        'id': batch_metadata[i]['id'],
                        'prediction': clean_pred,
                        'ground_truth': batch_metadata[i]['reference'],
                        'text': batch_metadata[i]['text'],
                        'task': dataset_name
                    }
                    if thinking_content:
                        result['thinking_content'] = thinking_content
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    results.append(result)
                        
            except Exception as e:
                logger.error(f"Error processing batch starting at {start_idx}: {e}")
                continue
    
    logger.info(f"Inference complete. Results saved to: {output_path}")
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='LoRA evaluation of LLMs on clinical benchmark'
    )
    
    parser.add_argument(
        '--dataset-path',
        type=Path,
        required=True,
        help='Path to dataset JSONL file (eg test.jsonl)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        required=False,
        help='Path to save predictions (JSONL)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name or path to LoRA adapter directory'
    )
    
    parser.add_argument(
        '--dataset-name',
        type=str,
        required=True,
        choices=list(DATASET_REFERENCE_KEYS.keys()),
        help='Dataset name for prompt selection'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size (default: {BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=MAX_TOKENS,
        help=f'Max new tokens (default: {MAX_TOKENS})'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing)'
    )
    
    parser.add_argument(
        '--thinking',
        action='store_true',
        default=False,
        help='Enable thinking mode for extended reasoning models (default: False)'
    )
    
    args = parser.parse_args()
    
    if not args.dataset_path.exists():
        logger.error(f"Dataset path not found: {args.dataset_path}")
        sys.exit(1)

    dataset = load_jsonl(args.dataset_path)
    
    if args.max_samples:
        dataset = dataset[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples for testing")
    
    model, tokenizer = setup_model_and_tokenizer(args.model)
    max_tokens = args.max_tokens

    safe_model_name = args.model.replace("/", "_")
    default_output = (
        Path("../../results/v2/predictions_lora")
        / f"{args.dataset_name}_{safe_model_name}_lora_predictions.jsonl"
    )
    output_path = Path(args.output) if args.output else default_output
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        run_batch_inference(
            model,
            tokenizer,
            dataset,
            args.dataset_name,
            output_path,
            args.batch_size,
            max_tokens,
            use_thinking=args.thinking,
        )
    finally:
        logger.info("Cleaning up GPU memory...")
        del model
        torch.cuda.empty_cache()
    
    logger.info("Inference complete!")

if __name__ == "__main__":
    main()
