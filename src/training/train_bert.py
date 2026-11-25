"""
BERT Fine-Tuning Script for Discriminative Clinical Benchmark Tasks:
1. fall_detection: Binary Sequence Classification (Target: 0 or 1)
2. clinical_ner: Token Classification (BIO tagging for clinical entities)
3. drbodebench: Multiple Choice (Revalida medical exam, choices A-D)

Supports BERT-based encoder models:
- pucpr/biobertpt-all
- neuralmind/bert-base-portuguese-cased
- bert-base-multilingual-cased
"""

import os
import re
import json
import logging
import argparse
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForMultipleChoice,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
)

# -------------------------------------------------------------
# LOGGING SETUP
# -------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# UTILS & NORMALIZATION
# -------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Normalize text by stripping accents, lowercasing, and trimming."""
    text = unicodedata.normalize('NFKD', str(text))
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return text.lower().strip()


def tokenize_words(text: str) -> List[str]:
    """Tokenize text into words and punctuation."""
    return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load records from a jsonl file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict], filepath: Path):
    """Save records to a jsonl file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------------------------------------------
# 1. TASK: FALL DETECTION (Sequence Classification)
# -------------------------------------------------------------
def train_fall_detection(
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    pred_file: Path,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
):
    logger.info("=== Starting BERT Training: Fall Detection ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_data = load_jsonl(data_dir / "train.jsonl")
    dev_data = load_jsonl(data_dir / "dev.jsonl")
    test_data = load_jsonl(data_dir / "test.jsonl")

    def preprocess_examples(examples: List[Dict]):
        texts = [ex["Evolucao"] for ex in examples]
        labels = [int(ex["Target"]) for ex in examples]
        encodings = tokenizer(texts, truncation=True, max_length=512, padding=False)
        encodings["labels"] = labels
        return Dataset.from_dict(encodings)

    train_ds = preprocess_examples(train_data)
    dev_ds = preprocess_examples(dev_data)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        fp16=torch.cuda.is_available(),
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    logger.info("Training model...")
    trainer.train()

    # Save best model
    logger.info(f"Saving fine-tuned BERT model to: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Inference on Test Set
    logger.info("Running inference on test set...")
    test_ds = preprocess_examples(test_data)
    preds_raw = trainer.predict(test_ds)
    preds = np.argmax(preds_raw.predictions, axis=1)

    predictions = []
    for ex, p in zip(test_data, preds):
        predictions.append({
            "id": ex["id"],
            "ground_truth": int(ex["Target"]),
            "prediction": f"Queda: {int(p)}"
        })

    save_jsonl(predictions, pred_file)
    logger.info(f"Saved {len(predictions)} predictions to: {pred_file}")


# -------------------------------------------------------------
# 2. TASK: DRBODEBENCH (Multiple Choice)
# -------------------------------------------------------------
class MultipleChoiceCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        first = features[0]
        batch_size = len(features)
        num_choices = len(first["input_ids"])

        flattened_features = [
            [{k: v[i] for k, v in feature.items() if k != "label"} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        if "label" in first:
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)
        return batch


def train_drbodebench(
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    pred_file: Path,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
):
    logger.info("=== Starting BERT Training: DrBodeBench (Multiple Choice) ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMultipleChoice.from_pretrained(model_name)

    train_data = load_jsonl(data_dir / "train.jsonl")
    dev_data = load_jsonl(data_dir / "dev.jsonl")
    test_data = load_jsonl(data_dir / "test.jsonl")

    choice_letters = ["A", "B", "C", "D"]
    letter_to_id = {c: i for i, c in enumerate(choice_letters)}

    def preprocess_examples(examples: List[Dict]):
        processed = []
        for ex in examples:
            enunciado = ex["enunciado"]
            if ex.get("img_description"):
                enunciado = f"{enunciado} [Imagem: {ex['img_description']}]"

            first_sentences = [enunciado] * 4
            second_sentences = [ex["alternativas"].get(c, "") for c in choice_letters]

            encodings = tokenizer(
                first_sentences,
                second_sentences,
                truncation=True,
                max_length=512,
                padding=False,
            )

            label_idx = letter_to_id.get(ex.get("resposta", "A"), 0)
            encodings["label"] = label_idx
            processed.append(encodings)

        return processed

    train_ds = preprocess_examples(train_data)
    dev_ds = preprocess_examples(dev_data)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        fp16=torch.cuda.is_available(),
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=MultipleChoiceCollator(tokenizer=tokenizer),
    )

    logger.info("Training multiple choice model...")
    trainer.train()

    logger.info(f"Saving fine-tuned BERT model to: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Inference on Test Set
    logger.info("Running inference on test set...")
    test_ds = preprocess_examples(test_data)
    preds_raw = trainer.predict(test_ds)
    preds = np.argmax(preds_raw.predictions, axis=1)

    predictions = []
    for ex, p in zip(test_data, preds):
        pred_letter = choice_letters[int(p)]
        predictions.append({
            "id": ex["id"],
            "ground_truth": ex.get("resposta", "A"),
            "prediction": f"Resposta: {pred_letter}."
        })

    save_jsonl(predictions, pred_file)
    logger.info(f"Saved {len(predictions)} predictions to: {pred_file}")


# -------------------------------------------------------------
# 3. TASK: CLINICAL NER (Token Classification BIO)
# -------------------------------------------------------------
def create_bio_labels_from_tags(text: str, entities: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """Convert text and entities dict into word tokens and BIO labels."""
    tokens = tokenize_words(text)
    labels = ["O"] * len(tokens)
    tokens_normalized = [normalize_text(t) for t in tokens]

    for entity_class, entity_list in entities.items():
        if not entity_list:
            continue
        for ent in entity_list:
            if not ent or not isinstance(ent, str):
                continue
            ent = re.sub(r'[.,;:!?]+$', '', ent.strip())
            if not ent:
                continue

            ent_tokens = tokenize_words(ent)
            ent_tokens_normalized = [normalize_text(t) for t in ent_tokens]
            n = len(ent_tokens_normalized)

            for i in range(len(tokens_normalized) - n + 1):
                if tokens_normalized[i:i+n] == ent_tokens_normalized:
                    labels[i] = f"B-{entity_class.upper()}"
                    for j in range(1, n):
                        labels[i+j] = f"I-{entity_class.upper()}"

    return tokens, labels


def bio_tags_to_entities(words: List[str], bio_tags: List[str], class_map: Dict[str, str]) -> Dict[str, List[str]]:
    """Reconstruct entity dictionary from word tokens and BIO tags."""
    entities = {}
    current_class = None
    current_tokens = []

    for word, tag in zip(words, bio_tags):
        if tag.startswith("B-"):
            if current_class and current_tokens:
                cat = class_map.get(current_class, current_class)
                entities.setdefault(cat, []).append(" ".join(current_tokens))
            current_class = tag[2:]
            current_tokens = [word]
        elif tag.startswith("I-") and current_class == tag[2:]:
            current_tokens.append(word)
        else:
            if current_class and current_tokens:
                cat = class_map.get(current_class, current_class)
                entities.setdefault(cat, []).append(" ".join(current_tokens))
            current_class = None
            current_tokens = []

    if current_class and current_tokens:
        cat = class_map.get(current_class, current_class)
        entities.setdefault(cat, []).append(" ".join(current_tokens))

    return entities


def train_clinical_ner(
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    pred_file: Path,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
):
    logger.info("=== Starting BERT Training: Clinical NER ===")
    train_data = load_jsonl(data_dir / "train.jsonl")
    dev_data = load_jsonl(data_dir / "dev.jsonl")
    test_data = load_jsonl(data_dir / "test.jsonl")

    # Build unique tag set and category mapping
    all_tags = {"O"}
    class_map = {}
    for d in (train_data + dev_data + test_data):
        for k in d.get("tags", {}).keys():
            class_map[k.upper()] = k
            all_tags.add(f"B-{k.upper()}")
            all_tags.add(f"I-{k.upper()}")

    tag_list = sorted(list(all_tags))
    tag2id = {t: i for i, t in enumerate(tag_list)}
    id2tag = {i: t for i, t in enumerate(tag_list)}
    logger.info(f"Total unique BIO tags: {len(tag_list)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(tag_list),
        id2label=id2tag,
        label2id=tag2id,
    )

    def prepare_ner_dataset(examples: List[Dict]):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for ex in examples:
            words, word_labels = create_bio_labels_from_tags(ex["text"], ex.get("tags", {}))
            
            tokenized = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                padding=False,
            )

            word_ids = tokenized.word_ids()
            subword_labels = []
            prev_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    subword_labels.append(-100)
                elif word_idx != prev_word_idx:
                    tag_str = word_labels[word_idx] if word_idx < len(word_labels) else "O"
                    subword_labels.append(tag2id.get(tag_str, tag2id["O"]))
                else:
                    # Inside same word: use -100 to only train on first subword
                    subword_labels.append(-100)
                prev_word_idx = word_idx

            all_input_ids.append(tokenized["input_ids"])
            all_attention_mask.append(tokenized["attention_mask"])
            all_labels.append(subword_labels)

        return Dataset.from_dict({
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        })

    train_ds = prepare_ner_dataset(train_data)
    dev_ds = prepare_ner_dataset(dev_data)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        fp16=torch.cuda.is_available(),
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    )

    logger.info("Training token classification model...")
    trainer.train()

    logger.info(f"Saving fine-tuned BERT model to: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Inference on Test Set
    logger.info("Running inference on test set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    predictions = []
    for ex in test_data:
        text = ex["text"]
        words = tokenize_words(text)

        tokenized = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        word_ids = tokenized.word_ids()

        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0].cpu().numpy()
            pred_tag_ids = np.argmax(logits, axis=1)

        # Map back from first subwords to word-level tags
        pred_word_bio = ["O"] * len(words)
        seen_words = set()
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx not in seen_words and word_idx < len(words):
                seen_words.add(word_idx)
                tag_name = id2tag.get(pred_tag_ids[idx], "O")
                pred_word_bio[word_idx] = tag_name

        pred_entities = bio_tags_to_entities(words, pred_word_bio, class_map)

        predictions.append({
            "id": ex["id"],
            "text": text,
            "labels_ref": ex.get("tags", {}),
            "labels_pred": pred_entities,
        })

    save_jsonl(predictions, pred_file)
    logger.info(f"Saved {len(predictions)} predictions to: {pred_file}")


# -------------------------------------------------------------
# CLI MAIN DISPATCHER
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on discriminative clinical tasks.")
    parser.add_argument("--model-name", type=str, required=True, help="HF model name or path (e.g. pucpr/biobertpt-all)")
    parser.add_argument("--dataset-name", type=str, required=True, choices=["fall_detection", "clinical_ner", "drbodebench", "semclinbr"])
    parser.add_argument("--data-dir", type=str, required=True, help="Path to benchmark dataset folder")
    parser.add_argument("--output-dir", type=str, default="models/bert", help="Output directory for fine-tuned weights")
    parser.add_argument("--pred-file", type=str, default=None, help="Path to save predictions jsonl")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_tag = args.model_name.replace("/", "_")
    output_dir = Path(args.output_dir) / args.dataset_name / model_tag

    if args.pred_file:
        pred_file = Path(args.pred_file)
    else:
        pred_file = Path(f"results/v2/predictions_bert/{args.dataset_name}_{model_tag}_bert.jsonl")

    pred_file.parent.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset_name == "fall_detection":
        train_fall_detection(
            model_name=args.model_name,
            data_dir=data_dir,
            output_dir=output_dir,
            pred_file=pred_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    elif args.dataset_name == "drbodebench":
        train_drbodebench(
            model_name=args.model_name,
            data_dir=data_dir,
            output_dir=output_dir,
            pred_file=pred_file,
            epochs=args.epochs,
            batch_size=min(args.batch_size, 8),
            learning_rate=args.lr,
        )
    elif args.dataset_name in ["clinical_ner", "semclinbr"]:
        train_clinical_ner(
            model_name=args.model_name,
            data_dir=data_dir,
            output_dir=output_dir,
            pred_file=pred_file,
            epochs=args.epochs,
            batch_size=min(args.batch_size, 8),
            learning_rate=args.lr if args.lr != 2e-5 else 3e-5,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")


if __name__ == "__main__":
    main()
