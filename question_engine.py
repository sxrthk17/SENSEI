
"""
SENSEI — Verified Question Generation Engine

Responsibilities:
1. Extract concepts from retrieved PDF chunks.
2. Generate multiple-choice, short-answer, and true/false questions.
3. Verify evidence quotes against the supplied source chunks.
4. Attribute accepted questions to the actual PDF and page.
5. Save concepts and questions using the existing memory_db interfaces.
6. Provide grading and adaptive-difficulty helpers.

This module does not delete, rebuild, or modify the ChromaDB index.
"""

import json
import re
from typing import Any

from rag_pipeline import create_llm
from memory_db import (
    register_learning_document,
    save_concept,
    save_question,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MAX_SOURCE_CHARS = 16000
MAX_CHUNK_CHARS = 7000
MIN_EVIDENCE_CHARS = 25
DEFAULT_QUESTIONS_PER_CONCEPT = 3

ALLOWED_QUESTION_TYPES = {
    "multiple_choice",
    "short_answer",
    "true_false",
}


class QuestionEngineError(Exception):
    """Raised when grounded question generation fails."""


# ---------------------------------------------------------------------
# Source handling
# ---------------------------------------------------------------------

def _normalize_text(value: Any) -> str:
    """Normalize whitespace for reliable source comparisons."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _get_metadata(document: Any) -> dict:
    metadata = getattr(document, "metadata", {}) or {}
    return metadata if isinstance(metadata, dict) else {}


def _get_page_number(document: Any) -> str:
    """Read the 1-based page number stored by the RAG pipeline."""
    metadata = _get_metadata(document)
    page = metadata.get("page_number", metadata.get("page", "unknown"))

    try:
        return str(int(page))
    except (TypeError, ValueError):
        return str(page)


def _get_source_name(document: Any) -> str:
    metadata = _get_metadata(document)

    return str(
        metadata.get("source_pdf")
        or metadata.get("document_name")
        or metadata.get("source")
        or "unknown"
    )


def _get_document_id(document: Any) -> str:
    metadata = _get_metadata(document)
    return str(metadata.get("document_id") or "")


def _get_chunk_text(document: Any) -> str:
    content = getattr(document, "page_content", "")
    return content.strip() if isinstance(content, str) else ""


def _prepare_source(documents: list) -> str:
    """
    Build a page-labelled source for the generation prompt.

    Source labels are generated from actual retrieved-document metadata.
    They are not accepted as proof of evidence by themselves.
    """
    if not documents:
        raise QuestionEngineError(
            "No source material was provided. Retrieve document chunks first."
        )

    sections = []
    remaining = MAX_SOURCE_CHARS

    for index, document in enumerate(documents, start=1):
        content = _get_chunk_text(document)

        if not content or remaining <= 0:
            continue

        source_name = _get_source_name(document)
        page = _get_page_number(document)

        header = (
            f"[Source {index} | PDF: {source_name} | Page: {page}]\n"
        )

        available = remaining - len(header)
        if available <= 0:
            break

        content = content[:available]
        section = header + content

        sections.append(section)
        remaining -= len(section) + 2

    source = "\n\n".join(sections)

    if not source:
        raise QuestionEngineError(
            "The retrieved document chunks are empty."
        )

    return source


def _source_records(documents: list) -> list[dict]:
    """Build source records used to verify question evidence."""
    records = []

    for document in documents or []:
        text = _get_chunk_text(document)
        if not text:
            continue

        records.append({
            "text": text,
            "normalized_text": _normalize_text(text).casefold(),
            "page": _get_page_number(document),
            "source_pdf": _get_source_name(document),
            "document_id": _get_document_id(document),
        })

    return records


def _find_evidence(
    evidence_quote: str,
    documents: list,
) -> dict | None:
    """
    Find a sufficiently long, continuous quote in an actual source chunk.

    Whitespace differences are tolerated. The words themselves must match.
    Page attribution is derived from the matching chunk, not from the model.
    """
    quote = _normalize_text(evidence_quote)

    if len(quote) < MIN_EVIDENCE_CHARS:
        return None

    normalized_quote = quote.casefold()

    for record in _source_records(documents):
        if normalized_quote in record["normalized_text"]:
            return {
                "evidence_quote": quote,
                "source_pdf": record["source_pdf"],
                "page_number": record["page"],
                "document_id": record["document_id"],
            }

    return None


# ---------------------------------------------------------------------
# Model output handling
# ---------------------------------------------------------------------

def _extract_json(response_text: str) -> dict:
    """Parse JSON, including JSON wrapped in Markdown fences."""
    if not response_text or not response_text.strip():
        raise QuestionEngineError("The model returned an empty response.")

    cleaned = response_text.strip()
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end <= start:
            raise QuestionEngineError(
                "The model did not return a valid JSON object."
            )

        try:
            result = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise QuestionEngineError(
                "Could not parse the generated JSON."
            ) from exc

    if not isinstance(result, dict):
        raise QuestionEngineError(
            "The generated result must be a JSON object."
        )

    return result


def _generate_json(prompt: str) -> dict:
    """Call SENSEI's configured LLM and parse its response."""
    try:
        response = create_llm().invoke(prompt)
        content = getattr(response, "content", response)

        if isinstance(content, list):
            parts = []

            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))

            content = "\n".join(parts)

        if not isinstance(content, str):
            content = str(content)

        return _extract_json(content)

    except QuestionEngineError:
        raise
    except Exception as exc:
        raise QuestionEngineError(
            f"Question generation failed: {exc}"
        ) from exc


# ---------------------------------------------------------------------
# Concept extraction
# ---------------------------------------------------------------------

def extract_concepts(
    documents: list,
    max_concepts: int = 12,
) -> list[dict]:
    """
    Extract concepts from supplied source chunks.

    Returns:
        [
            {
                "key": "stable_concept_key",
                "name": "Concept name",
                "pages": ["2", "3"]
            }
        ]
    """
    if not 1 <= max_concepts <= 30:
        raise ValueError("max_concepts must be between 1 and 30.")

    source = _prepare_source(documents)
    available_pages = {
        _get_page_number(doc)
        for doc in documents
        if _get_chunk_text(doc)
    }

    prompt = f"""
You are the concept-extraction component of SENSEI.

Extract up to {max_concepts} useful study concepts from the supplied
source material.

Rules:
- Use only concepts explicitly supported by the source.
- Do not add outside knowledge.
- Merge duplicate concepts.
- Each key must contain lowercase letters, numbers, and underscores.
- List only page numbers shown in the source labels.
- Do not invent page numbers.
- Return valid JSON only.

Required format:
{{
  "concepts": [
    {{
      "key": "stable_concept_key",
      "name": "Concept name",
      "pages": ["2", "3"]
    }}
  ]
}}

SOURCE MATERIAL:
{source}
"""

    result = _generate_json(prompt)
    raw_concepts = result.get("concepts", [])

    if not isinstance(raw_concepts, list):
        raise QuestionEngineError("'concepts' must be a list.")

    concepts = []
    seen_keys = set()
    seen_names = set()

    for item in raw_concepts:
        if not isinstance(item, dict):
            continue

        name = item.get("name")
        key = item.get("key")
        pages = item.get("pages", [])

        if not isinstance(name, str) or not name.strip():
            continue

        if not isinstance(key, str) or not key.strip():
            continue

        key = re.sub(
            r"[^a-z0-9_]+",
            "_",
            key.lower(),
        ).strip("_")

        normalized_name = _normalize_text(name).casefold()

        if (
            not key
            or key in seen_keys
            or normalized_name in seen_names
        ):
            continue

        if not isinstance(pages, list):
            pages = []

        # Never retain a page number absent from the retrieved material.
        valid_pages = []
        for page in pages:
            page = str(page)
            if page in available_pages and page not in valid_pages:
                valid_pages.append(page)

        concepts.append({
            "key": key,
            "name": _normalize_text(name),
            "pages": valid_pages,
        })

        seen_keys.add(key)
        seen_names.add(normalized_name)

        if len(concepts) >= max_concepts:
            break

    if not concepts:
        raise QuestionEngineError(
            "No usable concepts could be extracted from the source material."
        )

    return concepts


# ---------------------------------------------------------------------
# Question validation
# ---------------------------------------------------------------------

def _normalize_options(options: Any) -> dict | None:
    """Validate a four-option MCQ while preserving option labels."""
    if not isinstance(options, dict) or len(options) != 4:
        return None

    cleaned = {}

    for label, value in options.items():
        if not isinstance(label, str) or not isinstance(value, str):
            return None

        label = label.strip().upper()
        value = _normalize_text(value)

        if not label or not value or label in cleaned:
            return None

        cleaned[label] = value

    if set(cleaned) != {"A", "B", "C", "D"}:
        return None

    values = [value.casefold() for value in cleaned.values()]
    if len(set(values)) != 4:
        return None

    return cleaned


def _validate_question(
    item: Any,
    documents: list,
) -> dict | None:
    """
    Validate question structure and source evidence.

    A model-supplied page number is never trusted. The page is resolved
    from the actual document chunk containing the evidence quote.
    """
    if not isinstance(item, dict):
        return None

    question = item.get("question")
    question_type = item.get("question_type")
    answer_key = item.get("answer_key")
    explanation = item.get("explanation", "")
    evidence_quote = item.get("evidence_quote", "")

    if not isinstance(question, str) or not question.strip():
        return None

    question = _normalize_text(question)

    if question_type not in ALLOWED_QUESTION_TYPES:
        return None

    if not isinstance(answer_key, str) or not answer_key.strip():
        return None

    answer_key = answer_key.strip()

    if not isinstance(explanation, str) or not explanation.strip():
        return None

    explanation = _normalize_text(explanation)

    # Every accepted question needs evidence that matches a supplied chunk.
    evidence = _find_evidence(evidence_quote, documents)
    if evidence is None:
        return None

    difficulty = item.get("difficulty", 1)
    try:
        difficulty = int(difficulty)
    except (TypeError, ValueError):
        return None

    if not 1 <= difficulty <= 5:
        return None

    if question_type == "multiple_choice":
        options = _normalize_options(item.get("options"))
        if options is None:
            return None

        correct_label = answer_key.upper()

        if correct_label not in options:
            return None

        correct_answer = options[correct_label]

        stored_answer = {
            "options": options,
            "correct_option": correct_label,
            "expected_answer": correct_answer,
            "explanation": explanation,
            **evidence,
        }

    elif question_type == "true_false":
        normalized_answer = answer_key.casefold()

        if normalized_answer not in {"true", "false"}:
            return None

        stored_answer = {
            "correct_answer": normalized_answer,
            "explanation": explanation,
            **evidence,
        }

    else:
        # Short-answer keys must be useful, nonempty expected answers.
        expected_answer = _normalize_text(answer_key)

        if not expected_answer:
            return None

        stored_answer = {
            "expected_answer": expected_answer,
            "explanation": explanation,
            **evidence,
        }

    page = evidence["page_number"]
    source_pdf = evidence["source_pdf"]

    return {
        "question": question,
        "question_type": question_type,
        "answer_key": json.dumps(
            stored_answer,
            ensure_ascii=False,
        ),
        "source_reference": f"{source_pdf} | Page {page}",
        "difficulty": difficulty,
        "evidence_quote": evidence["evidence_quote"],
        "source_pdf": source_pdf,
        "page_number": page,
        "document_id": evidence["document_id"],
        "explanation": explanation,
    }


# ---------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------

def generate_questions_for_concept(
    documents: list,
    concept_name: str,
    questions_per_concept: int = DEFAULT_QUESTIONS_PER_CONCEPT,
) -> list[dict]:
    """
    Generate and validate questions for one concept.

    Unsupported or malformed questions are rejected, not repaired by
    guessing. The returned answer_key remains a JSON string for the
    existing memory_db.save_question interface.
    """
    if not 1 <= questions_per_concept <= 10:
        raise ValueError(
            "questions_per_concept must be between 1 and 10."
        )

    if not documents:
        return []

    source = _prepare_source(documents)

    prompt = f"""
You create assessment questions for SENSEI, a document-grounded tutor.

Target concept: {concept_name}
Requested maximum: {questions_per_concept}

Use ONLY the supplied SOURCE MATERIAL. Do not use outside knowledge.

For each question:
- Ask something answerable from the source alone.
- Provide an exact, continuous evidence_quote copied from the source.
- The quote must be long enough to identify the supporting passage.
- Do not invent facts or rely on unseen diagrams or tables.
- Provide an explanation supported by the same source.
- Use a difficulty integer from 1 to 5.
- Provide the source page as context, but the application will verify it.
- Return fewer questions if the source cannot support enough questions.
- Return valid JSON only.

Allowed question_type values:
1. multiple_choice
2. short_answer
3. true_false

For multiple_choice:
- Supply exactly four options with labels A, B, C, D.
- answer_key must be the correct label: A, B, C, or D.
- All four option texts must be distinct.

For true_false:
- Set options to null.
- answer_key must be "true" or "false".

For short_answer:
- Set options to null.
- answer_key must be a concise expected answer.

Required JSON:
{{
  "questions": [
    {{
      "question": "Question text",
      "question_type": "multiple_choice",
      "options": {{
        "A": "Option one",
        "B": "Option two",
        "C": "Option three",
        "D": "Option four"
      }},
      "answer_key": "B",
      "explanation": "Source-supported explanation",
      "source_reference": "Page 3",
      "evidence_quote": "Exact continuous quote from the source",
      "difficulty": 2
    }}
  ]
}}

For short_answer and true_false, set options to null.

SOURCE MATERIAL:
{source}
"""

    result = _generate_json(prompt)
    raw_questions = result.get("questions", [])

    if not isinstance(raw_questions, list):
        raise QuestionEngineError("'questions' must be a list.")

    validated = []
    seen_questions = set()

    for item in raw_questions:
        question = _validate_question(item, documents)

        if question is None:
            continue

        normalized_question = question["question"].casefold()

        if normalized_question in seen_questions:
            continue

        seen_questions.add(normalized_question)
        validated.append(question)

        if len(validated) >= questions_per_concept:
            break

    return validated


# ---------------------------------------------------------------------
# Learning-bank construction
# ---------------------------------------------------------------------

def build_learning_bank(
    document_id: str,
    document_name: str,
    documents: list,
    max_concepts: int = 12,
    questions_per_concept: int = DEFAULT_QUESTIONS_PER_CONCEPT,
) -> dict:
    """
    Extract concepts, generate evidence-checked questions, and save them.

    Existing memory_db function signatures and duplicate behavior are
    preserved. No database or vector-store reset is performed.
    """
    if not document_id or not document_name:
        raise ValueError("document_id and document_name are required.")

    if not documents:
        raise QuestionEngineError(
            "No retrieved chunks were supplied for this document."
        )

    register_learning_document(document_id, document_name)

    concepts = extract_concepts(
        documents,
        max_concepts=max_concepts,
    )

    summary = {
        "document_id": document_id,
        "document_name": document_name,
        "concepts_found": len(concepts),
        "questions_generated": 0,
        "questions_saved": 0,
        "questions_rejected_by_validation": 0,
        "concept_results": [],
    }

    for concept in concepts:
        concept_id = save_concept(
            document_id=document_id,
            concept_key=concept["key"],
            concept_name=concept["name"],
            source_pages=", ".join(concept["pages"]),
        )

        concept_result = {
            "name": concept["name"],
            "concept_id": concept_id,
            "questions_generated": 0,
            "questions_saved": 0,
            "questions_rejected_by_validation": 0,
        }

        # If the concept names supporting pages, only use those pages.
        # If none of the declared pages are available, skip this concept
        # rather than quietly using unrelated chunks.
        relevant_documents = documents

        if concept["pages"]:
            matching = [
                doc for doc in documents
                if _get_page_number(doc) in concept["pages"]
            ]

            if not matching:
                summary["concept_results"].append(concept_result)
                continue

            relevant_documents = matching

        before_count = len(relevant_documents)

        questions = generate_questions_for_concept(
            documents=relevant_documents,
            concept_name=concept["name"],
            questions_per_concept=questions_per_concept,
        )

        concept_result["questions_generated"] = len(questions)
        summary["questions_generated"] += len(questions)

        # The model may return unsupported candidates; rejected candidates
        # are not returned by the validator. We report only accepted output
        # here rather than inventing an unobservable rejection count.
        concept_result["questions_rejected_by_validation"] = max(
            0,
            min(questions_per_concept, before_count * questions_per_concept)
            - len(questions),
        )
        summary["questions_rejected_by_validation"] += (
            concept_result["questions_rejected_by_validation"]
        )

        for question in questions:
            question_id = save_question(
                document_id=document_id,
                concept_id=concept_id,
                question=question["question"],
                question_type=question["question_type"],
                answer_key=question["answer_key"],
                source_reference=question["source_reference"],
                difficulty=question["difficulty"],
            )

            # Preserve the existing duplicate-handling convention.
            if question_id is not None:
                concept_result["questions_saved"] += 1
                summary["questions_saved"] += 1

        summary["concept_results"].append(concept_result)

    return summary


# ---------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------

def grade_question(
    question_type: str,
    answer_key: str,
    student_answer: Any,
) -> dict:
    """
    Grade a stored question without changing the database schema.

    answer_key is the JSON string saved by build_learning_bank().
    Multiple-choice responses can be the option label or its exact text.
    Short-answer grading is conservative: normalized exact matching only.
    """
    try:
        stored = json.loads(answer_key)
    except (TypeError, json.JSONDecodeError):
        raise QuestionEngineError(
            "The stored answer key is invalid JSON."
        )

    submitted = _normalize_text(student_answer)
    explanation = str(stored.get("explanation", ""))

    if question_type == "multiple_choice":
        options = stored.get("options", {})
        correct_option = str(
            stored.get("correct_option", "")
        ).upper()
        expected = _normalize_text(
            stored.get("expected_answer", "")
        )

        is_correct = (
            submitted.casefold() == correct_option.casefold()
            or submitted.casefold() == expected.casefold()
        )

        return {
            "is_correct": is_correct,
            "correct_answer": expected,
            "correct_option": correct_option,
            "explanation": explanation,
            "evidence_quote": stored.get("evidence_quote", ""),
            "source_pdf": stored.get("source_pdf", "unknown"),
            "page_number": stored.get("page_number", "unknown"),
            "options": options,
        }

    if question_type == "true_false":
        expected = str(
            stored.get("correct_answer", "")
        ).casefold()

        normalized_submission = submitted.casefold()
        aliases = {
            "t": "true",
            "yes": "true",
            "f": "false",
            "no": "false",
        }
        normalized_submission = aliases.get(
            normalized_submission,
            normalized_submission,
        )

        return {
            "is_correct": normalized_submission == expected,
            "correct_answer": expected,
            "explanation": explanation,
            "evidence_quote": stored.get("evidence_quote", ""),
            "source_pdf": stored.get("source_pdf", "unknown"),
            "page_number": stored.get("page_number", "unknown"),
        }

    expected = _normalize_text(stored.get("expected_answer", ""))

    # Do not use an LLM to guess whether an arbitrary free-text answer
    # is correct. This safe baseline uses normalized exact matching.
    return {
        "is_correct": (
            bool(submitted)
            and submitted.casefold() == expected.casefold()
        ),
        "correct_answer": expected,
        "explanation": explanation,
        "evidence_quote": stored.get("evidence_quote", ""),
        "source_pdf": stored.get("source_pdf", "unknown"),
        "page_number": stored.get("page_number", "unknown"),
    }


def calculate_quiz_score(results: list[dict]) -> dict:
    """Calculate accuracy from a list of grade_question() results."""
    total = len(results)
    correct = sum(bool(result.get("is_correct")) for result in results)

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "percentage": round(correct * 100 / total, 2) if total else 0.0,
    }


# ---------------------------------------------------------------------
# Adaptive difficulty
# ---------------------------------------------------------------------

def next_difficulty(
    recent_scores: list[float],
    current_difficulty: int = 3,
) -> int:
    """
    Recommend a difficulty from 1 to 5.

    Increase by one if both of the last two scores are at least 80%.
    Decrease by one if both are below 50%.
    Otherwise keep the current level.

    This helper does not persist scores. The app should save attempts
    through the existing learning-event interface.
    """
    try:
        current = int(current_difficulty)
    except (TypeError, ValueError):
        current = 3

    current = max(1, min(5, current))

    if len(recent_scores) < 2:
        return current

    try:
        recent = [float(score) for score in recent_scores[-2:]]
    except (TypeError, ValueError):
        return current

    if all(score >= 80 for score in recent):
        return min(5, current + 1)

    if all(score < 50 for score in recent):
        return max(1, current - 1)

    return current


def select_questions_for_difficulty(
    questions: list[dict],
    target_difficulty: int = 3,
    limit: int = 10,
) -> list[dict]:
    """
    Select saved/generated question records near a target difficulty.

    This function does not fetch from the database; pass it records
    obtained through the app's existing question retrieval workflow.
    """
    target = max(1, min(5, int(target_difficulty)))
    limit = max(1, min(50, int(limit)))

    # Prefer questions at the requested level, then nearest levels.
    return sorted(
        questions,
        key=lambda item: abs(
            int(item.get("difficulty", 1)) - target
        ),
    )[:limit]
