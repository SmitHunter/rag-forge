"""Evaluation metrics for RAG systems."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_forge.eval.datasets import QAExample
    from rag_forge.retrieval.base import RetrievalResult


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval evaluation."""

    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    num_examples: int
    num_answerable: int = 0
    num_unanswerable: int = 0

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary."""
        return {
            "recall@1": self.recall_at_1,
            "recall@3": self.recall_at_3,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "mrr": self.mrr,
            "num_examples": self.num_examples,
            "num_answerable": self.num_answerable,
            "num_unanswerable": self.num_unanswerable,
        }


@dataclass
class AnswerMetrics:
    """Metrics for answer quality evaluation."""

    faithfulness: float
    answer_correctness: float
    abstention_accuracy: float
    avg_answer_length: float
    num_examples: int

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary."""
        return {
            "faithfulness": self.faithfulness,
            "answer_correctness": self.answer_correctness,
            "abstention_accuracy": self.abstention_accuracy,
            "avg_answer_length": self.avg_answer_length,
            "num_examples": self.num_examples,
        }


def calculate_recall_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
    k: int,
) -> float:
    """Calculate recall@k.

    Measures what fraction of relevant documents were retrieved in top-k results.

    Args:
        retrieved_doc_ids: List of retrieved document IDs in order.
        relevant_doc_ids: List of ground truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Recall@k score between 0 and 1.
    """
    if not relevant_doc_ids:
        return 1.0

    retrieved_set = set(retrieved_doc_ids[:k])
    relevant_set = set(relevant_doc_ids)

    hits = len(retrieved_set & relevant_set)
    return hits / len(relevant_set)


def calculate_mrr(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
) -> float:
    """Calculate Mean Reciprocal Rank.

    MRR is 1/rank where rank is the position of the first relevant document.

    Args:
        retrieved_doc_ids: List of retrieved document IDs in order.
        relevant_doc_ids: List of ground truth relevant document IDs.

    Returns:
        MRR score between 0 and 1.
    """
    if not relevant_doc_ids:
        return 1.0

    relevant_set = set(relevant_doc_ids)

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def calculate_faithfulness(
    answer: str,
    context_chunks: list[str],
) -> float:
    """Calculate faithfulness score (groundedness).

    Measures how well the answer is grounded in the retrieved context.
    Uses a simple heuristic based on n-gram overlap.

    Args:
        answer: Generated answer.
        context_chunks: Retrieved context chunks used for generation.

    Returns:
        Faithfulness score between 0 and 1.
    """
    if not answer or not context_chunks:
        return 0.0

    def extract_ngrams(text: str, n: int) -> set[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < n:
            return {" ".join(words)} if words else set()
        return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

    def extract_key_phrases(text: str) -> set[str]:
        text_lower = text.lower()
        phrases = set()
        for n in [2, 3, 4]:
            phrases.update(extract_ngrams(text_lower, n))
        return phrases

    answer_phrases = extract_key_phrases(answer)
    if not answer_phrases:
        return 1.0

    combined_context = " ".join(context_chunks)
    context_phrases = extract_key_phrases(combined_context)

    if not context_phrases:
        return 0.0

    grounded_phrases = answer_phrases & context_phrases
    return len(grounded_phrases) / len(answer_phrases)


def calculate_answer_correctness(
    generated_answer: str,
    reference_answer: str,
) -> float:
    """Calculate answer correctness using token overlap.

    Uses F1 score based on token overlap between generated and reference answers.

    Args:
        generated_answer: Generated answer from the system.
        reference_answer: Ground truth reference answer.

    Returns:
        Correctness score between 0 and 1.
    """
    if not generated_answer or not reference_answer:
        return 0.0

    def tokenize(text: str) -> set[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "of",
            "in",
            "to",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "and",
            "but",
            "or",
            "nor",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "not",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "just",
            "also",
            "now",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "any",
            "some",
            "no",
            "other",
            "such",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
        }
        return {w for w in words if w not in stop_words and len(w) > 2}

    gen_tokens = tokenize(generated_answer)
    ref_tokens = tokenize(reference_answer)

    if not gen_tokens or not ref_tokens:
        return 0.0

    common_tokens = gen_tokens & ref_tokens

    precision = len(common_tokens) / len(gen_tokens) if gen_tokens else 0
    recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def evaluate_retrieval(
    examples: list[QAExample],
    retrieval_results: list[list[RetrievalResult]],
) -> RetrievalMetrics:
    """Evaluate retrieval quality across a dataset.

    Args:
        examples: QA examples with ground truth.
        retrieval_results: Retrieved results for each example.

    Returns:
        Aggregated retrieval metrics.
    """
    recalls_1 = []
    recalls_3 = []
    recalls_5 = []
    recalls_10 = []
    mrrs = []
    num_answerable = 0
    num_unanswerable = 0

    for example, results in zip(examples, retrieval_results, strict=False):
        retrieved_doc_ids = [r.chunk.document_id for r in results]
        relevant_ids = example.relevant_doc_ids or example.relevant_chunk_ids

        is_unanswerable = (
            not relevant_ids
            or example.metadata.get("type") == "unanswerable"
            or "UNANSWERABLE" in example.answer
        )

        if is_unanswerable:
            num_unanswerable += 1
            continue

        num_answerable += 1
        recalls_1.append(calculate_recall_at_k(retrieved_doc_ids, relevant_ids, 1))
        recalls_3.append(calculate_recall_at_k(retrieved_doc_ids, relevant_ids, 3))
        recalls_5.append(calculate_recall_at_k(retrieved_doc_ids, relevant_ids, 5))
        recalls_10.append(calculate_recall_at_k(retrieved_doc_ids, relevant_ids, 10))
        mrrs.append(calculate_mrr(retrieved_doc_ids, relevant_ids))

    n = len(recalls_1)
    if n == 0:
        return RetrievalMetrics(
            recall_at_1=0.0,
            recall_at_3=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            mrr=0.0,
            num_examples=0,
            num_answerable=0,
            num_unanswerable=num_unanswerable,
        )

    return RetrievalMetrics(
        recall_at_1=sum(recalls_1) / n,
        recall_at_3=sum(recalls_3) / n,
        recall_at_5=sum(recalls_5) / n,
        recall_at_10=sum(recalls_10) / n,
        mrr=sum(mrrs) / n,
        num_examples=n + num_unanswerable,
        num_answerable=num_answerable,
        num_unanswerable=num_unanswerable,
    )


def is_abstention(answer: str) -> bool:
    """Check if an answer is an abstention (says it cannot answer)."""
    abstention_phrases = [
        "cannot answer",
        "can't answer",
        "don't have",
        "do not have",
        "not mentioned",
        "not in the context",
        "no information",
        "not provided",
        "unable to answer",
        "i don't know",
        "not enough information",
        "cannot find",
        "not available",
        "unanswerable",
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in abstention_phrases)


def evaluate_answers(
    examples: list[QAExample],
    generated_answers: list[str],
    context_chunks_list: list[list[str]],
) -> AnswerMetrics:
    """Evaluate answer quality across a dataset.

    Args:
        examples: QA examples with ground truth answers.
        generated_answers: Generated answers for each example.
        context_chunks_list: Context chunks used for each generation.

    Returns:
        Aggregated answer metrics.
    """
    faithfulness_scores = []
    correctness_scores = []
    answer_lengths = []
    abstention_correct = 0
    abstention_total = 0

    for example, answer, context_chunks in zip(
        examples, generated_answers, context_chunks_list, strict=False
    ):
        is_unanswerable = (
            example.metadata.get("type") == "unanswerable" or "UNANSWERABLE" in example.answer
        )

        if is_unanswerable:
            abstention_total += 1
            if is_abstention(answer):
                abstention_correct += 1
            continue

        faithfulness_scores.append(calculate_faithfulness(answer, context_chunks))
        correctness_scores.append(calculate_answer_correctness(answer, example.answer))
        answer_lengths.append(len(answer.split()))

    n = len(faithfulness_scores)
    abstention_acc = abstention_correct / abstention_total if abstention_total > 0 else 1.0

    if n == 0:
        return AnswerMetrics(
            faithfulness=0.0,
            answer_correctness=0.0,
            abstention_accuracy=abstention_acc,
            avg_answer_length=0.0,
            num_examples=abstention_total,
        )

    return AnswerMetrics(
        faithfulness=sum(faithfulness_scores) / n,
        answer_correctness=sum(correctness_scores) / n,
        abstention_accuracy=abstention_acc,
        avg_answer_length=sum(answer_lengths) / n,
        num_examples=n + abstention_total,
    )
