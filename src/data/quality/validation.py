"""
Field validation for records across all benchmark tasks.
Follows Section 3.1 of the ClinicalBenchPT Curation Specification.
"""

from typing import Any, Dict, Optional, Tuple
from .normalization import normalize_text


def validate_record(record: Dict[str, Any], task: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate mandatory fields for a record according to its task.
    
    Returns:
        (is_valid, responsible_field, issue_reason)
    """
    if not isinstance(record, dict):
        return False, "record", "Record is not a dictionary"

    # Task: Question Answering (abcfarma_qa, amazon_diseases, wikidoc_pt)
    if task == "qa":
        question = record.get("question") if record.get("question") is not None else record.get("Question")
        if question is None or normalize_text(question) == "":
            return False, "question", "Missing or empty question"
        
        answer = record.get("answer") if record.get("answer") is not None else (
            record.get("Response") if record.get("Response") is not None else record.get("response")
        )
        if answer is None or normalize_text(answer) == "":
            return False, "answer", "Missing or empty answer"
        
        return True, None, None

    # Task: Classification (fall_detection)
    elif task == "classification":
        evolucao = record.get("Evolucao")
        if evolucao is None or normalize_text(evolucao) == "":
            return False, "Evolucao", "Missing or empty Evolucao"
        
        target = record.get("Target")
        # Target=0 is completely valid! Only reject None or empty string
        if target is None:
            return False, "Target", "Missing Target"
        if isinstance(target, str) and target.strip() == "":
            return False, "Target", "Empty Target string"
        
        return True, None, None

    # Task: Multiple Choice (drbodebench)
    elif task == "multiple_choice":
        enunciado = record.get("enunciado")
        if enunciado is None or normalize_text(enunciado) == "":
            return False, "enunciado", "Missing or empty enunciado"
        
        alternativas = record.get("alternativas")
        if not isinstance(alternativas, dict) or len(alternativas) == 0:
            return False, "alternativas", "Missing, empty, or invalid alternativas dict"
        
        # Check that each alternative has non-empty content
        for opt_key, opt_val in alternativas.items():
            if opt_val is None or normalize_text(opt_val) == "":
                return False, f"alternativas[{opt_key}]", f"Empty option text for alternative {opt_key}"
        
        resposta = record.get("resposta")
        if resposta is None or normalize_text(resposta) == "":
            return False, "resposta", "Missing or empty resposta"
        if resposta not in alternativas:
            return False, "resposta", f"Answer '{resposta}' not found in available alternativas"
        
        return True, None, None

    # Task: Summarization (multiclinsum_pt)
    elif task == "summarization":
        text = record.get("text")
        if text is None or normalize_text(text) == "":
            return False, "text", "Missing or empty text"
        
        summary = record.get("summary")
        if summary is None or normalize_text(summary) == "":
            return False, "summary", "Missing or empty summary"
        
        return True, None, None

    # Task: Named Entity Recognition (semclinbr, clinical_ner)
    elif task == "ner":
        text = record.get("text")
        if text is None or normalize_text(text) == "":
            return False, "text", "Missing or empty text"
        
        tags = record.get("tags")
        # Note: A valid document with no entities can have tags={} (Section 3.1)
        if tags is None or not isinstance(tags, dict):
            return False, "tags", "Missing or invalid tags structure"
        
        return True, None, None

    else:
        # Generic fallback
        return True, None, None
