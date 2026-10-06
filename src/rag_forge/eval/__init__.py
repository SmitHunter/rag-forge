"""Evaluation harness for RAG systems."""

from rag_forge.eval.datasets import QADataset, QAExample, load_qa_dataset
from rag_forge.eval.harness import EvalConfig, EvalHarness, EvalResult, default_eval_configs
from rag_forge.eval.metrics import (
    AnswerMetrics,
    RetrievalMetrics,
    calculate_answer_correctness,
    calculate_faithfulness,
    calculate_mrr,
    calculate_recall_at_k,
)

__all__ = [
    "AnswerMetrics",
    "EvalConfig",
    "EvalHarness",
    "EvalResult",
    "QADataset",
    "QAExample",
    "RetrievalMetrics",
    "calculate_answer_correctness",
    "calculate_faithfulness",
    "calculate_mrr",
    "calculate_recall_at_k",
    "default_eval_configs",
    "load_qa_dataset",
]
