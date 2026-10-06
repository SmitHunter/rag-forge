"""Evaluation harness for comparing RAG configurations."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from rag_forge.config import RetrievalStrategy, Settings
from rag_forge.eval.datasets import QADataset
from rag_forge.eval.metrics import (
    AnswerMetrics,
    RetrievalMetrics,
    evaluate_answers,
    evaluate_retrieval,
)
from rag_forge.pipeline.rag import RAGPipeline

if TYPE_CHECKING:
    from rag_forge.retrieval.base import RetrievalResult


@dataclass
class EvalConfig:
    """Configuration for a single evaluation run."""

    name: str
    retrieval_strategy: RetrievalStrategy
    use_reranking: bool = False
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    top_k: int = 5
    rerank_top_k: int = 3

    def to_settings(self, base_settings: Settings) -> Settings:
        """Create Settings from this config."""
        return Settings(
            data_dir=base_settings.data_dir,
            cache_dir=base_settings.cache_dir,
            embedding_provider=base_settings.embedding_provider,
            embedding_model=base_settings.embedding_model,
            llm_provider=base_settings.llm_provider,
            llm_model=base_settings.llm_model,
            openai_api_key=base_settings.openai_api_key,
            anthropic_api_key=base_settings.anthropic_api_key,
            ollama_host=base_settings.ollama_host,
            retrieval_strategy=self.retrieval_strategy,
            retrieval_top_k=self.top_k,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
            use_reranking=self.use_reranking,
            rerank_top_k=self.rerank_top_k,
        )


@dataclass
class EvalResult:
    """Results from a single evaluation run."""

    config_name: str
    retrieval_metrics: RetrievalMetrics
    answer_metrics: AnswerMetrics
    total_time_seconds: float
    avg_latency_ms: float
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "config_name": self.config_name,
            "retrieval_metrics": self.retrieval_metrics.to_dict(),
            "answer_metrics": self.answer_metrics.to_dict(),
            "total_time_seconds": self.total_time_seconds,
            "avg_latency_ms": self.avg_latency_ms,
            "metadata": self.metadata,
        }


def default_eval_configs() -> list[EvalConfig]:
    """Shared retrieval configurations used by the CLI and run_eval.py."""
    return [
        EvalConfig(
            name="BM25 Only",
            retrieval_strategy=RetrievalStrategy.BM25,
            top_k=5,
        ),
        EvalConfig(
            name="Dense Only",
            retrieval_strategy=RetrievalStrategy.DENSE,
            top_k=5,
        ),
        EvalConfig(
            name="Hybrid (BM25=0.3)",
            retrieval_strategy=RetrievalStrategy.HYBRID,
            bm25_weight=0.3,
            dense_weight=0.7,
            top_k=5,
        ),
        EvalConfig(
            name="Hybrid (BM25=0.5)",
            retrieval_strategy=RetrievalStrategy.HYBRID,
            bm25_weight=0.5,
            dense_weight=0.5,
            top_k=5,
        ),
        EvalConfig(
            name="Dense + Rerank",
            retrieval_strategy=RetrievalStrategy.DENSE,
            use_reranking=True,
            top_k=10,
            rerank_top_k=5,
        ),
        EvalConfig(
            name="Hybrid + Rerank",
            retrieval_strategy=RetrievalStrategy.HYBRID,
            bm25_weight=0.3,
            dense_weight=0.7,
            use_reranking=True,
            top_k=10,
            rerank_top_k=5,
        ),
    ]


class EvalHarness:
    """Evaluation harness for comparing RAG configurations."""

    def __init__(
        self,
        base_settings: Settings,
        documents: list[dict[str, str]],
        qa_dataset: QADataset,
    ) -> None:
        self.base_settings = base_settings
        self.documents = documents
        self.qa_dataset = qa_dataset
        self.results: list[EvalResult] = []
        self.console = Console()

    def run_evaluation(
        self,
        configs: list[EvalConfig],
        show_progress: bool = True,
    ) -> list[EvalResult]:
        """Run evaluation across multiple configurations.

        Args:
            configs: List of configurations to evaluate.
            show_progress: Whether to show progress.

        Returns:
            List of evaluation results.
        """
        self.results = []

        for config in configs:
            self.console.print(f"\n[bold blue]Evaluating: {config.name}[/bold blue]")
            result = self._evaluate_config(config, show_progress)
            self.results.append(result)

        return self.results

    def _evaluate_config(self, config: EvalConfig, show_progress: bool) -> EvalResult:
        """Evaluate a single configuration."""
        settings = config.to_settings(self.base_settings)

        pipeline = RAGPipeline(settings)
        pipeline.ingest_documents(self.documents, show_progress=show_progress)
        pipeline.build_index(show_progress=show_progress)

        examples = list(self.qa_dataset)
        retrieval_results: list[list[RetrievalResult]] = []
        generated_answers: list[str] = []
        context_chunks_list: list[list[str]] = []
        latencies: list[float] = []

        example_iter = tqdm(examples, desc="Running queries") if show_progress else examples
        for example in example_iter:
            start_time = time.perf_counter()

            response = pipeline.query(example.question, top_k=config.top_k)

            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

            retrieval_results.append(response.retrieval_results)
            generated_answers.append(response.answer)
            context_chunks_list.append([r.chunk.text for r in response.retrieval_results])

        retrieval_metrics = evaluate_retrieval(examples, retrieval_results)
        answer_metrics = evaluate_answers(examples, generated_answers, context_chunks_list)

        return EvalResult(
            config_name=config.name,
            retrieval_metrics=retrieval_metrics,
            answer_metrics=answer_metrics,
            total_time_seconds=sum(latencies) / 1000,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            metadata={
                "retrieval_strategy": config.retrieval_strategy.value,
                "use_reranking": config.use_reranking,
                "top_k": config.top_k,
                "num_documents": len(self.documents),
                "num_questions": len(examples),
            },
        )

    def print_results_table(self) -> None:
        """Print a formatted results table."""
        if not self.results:
            self.console.print("[yellow]No results to display[/yellow]")
            return

        table = Table(title="RAG Evaluation Results", show_header=True, header_style="bold magenta")

        table.add_column("Configuration", style="cyan", no_wrap=True)
        table.add_column("R@1", justify="right")
        table.add_column("R@5", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column("Faith.", justify="right")
        table.add_column("Latency", justify="right")

        for result in self.results:
            table.add_row(
                result.config_name,
                f"{result.retrieval_metrics.recall_at_1:.3f}",
                f"{result.retrieval_metrics.recall_at_5:.3f}",
                f"{result.retrieval_metrics.mrr:.3f}",
                f"{result.answer_metrics.faithfulness:.3f}",
                f"{result.avg_latency_ms:.0f}ms",
            )

        self.console.print(table)

    def get_results_markdown(self) -> str:
        """Get results as a Markdown table."""
        if not self.results:
            return "No results to display."

        lines = [
            "| Configuration | Recall@1 | Recall@5 | MRR | Faithfulness | Latency |",
            "|---------------|----------|----------|-----|--------------|---------|",
        ]

        for result in self.results:
            lines.append(
                f"| {result.config_name} "
                f"| {result.retrieval_metrics.recall_at_1:.3f} "
                f"| {result.retrieval_metrics.recall_at_5:.3f} "
                f"| {result.retrieval_metrics.mrr:.3f} "
                f"| {result.answer_metrics.faithfulness:.3f} "
                f"| {result.avg_latency_ms:.0f}ms |"
            )

        return "\n".join(lines)

    def save_results(self, path: Path | str) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "num_configurations": len(self.results),
                "num_documents": len(self.documents),
                "num_questions": len(self.qa_dataset),
                "best_recall_at_5": max(
                    (r.retrieval_metrics.recall_at_5 for r in self.results), default=0
                ),
                "best_mrr": max((r.retrieval_metrics.mrr for r in self.results), default=0),
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_results(cls, path: Path | str) -> list[EvalResult]:
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)

        results = []
        for r in data["results"]:
            retrieval_metrics = RetrievalMetrics(
                recall_at_1=r["retrieval_metrics"]["recall@1"],
                recall_at_3=r["retrieval_metrics"]["recall@3"],
                recall_at_5=r["retrieval_metrics"]["recall@5"],
                recall_at_10=r["retrieval_metrics"]["recall@10"],
                mrr=r["retrieval_metrics"]["mrr"],
                num_examples=r["retrieval_metrics"]["num_examples"],
                num_answerable=r["retrieval_metrics"].get("num_answerable", 0),
                num_unanswerable=r["retrieval_metrics"].get("num_unanswerable", 0),
            )
            answer_metrics = AnswerMetrics(
                faithfulness=r["answer_metrics"]["faithfulness"],
                answer_correctness=r["answer_metrics"]["answer_correctness"],
                abstention_accuracy=r["answer_metrics"].get("abstention_accuracy", 0.0),
                avg_answer_length=r["answer_metrics"]["avg_answer_length"],
                num_examples=r["answer_metrics"]["num_examples"],
            )
            results.append(
                EvalResult(
                    config_name=r["config_name"],
                    retrieval_metrics=retrieval_metrics,
                    answer_metrics=answer_metrics,
                    total_time_seconds=r["total_time_seconds"],
                    avg_latency_ms=r["avg_latency_ms"],
                    metadata=r.get("metadata", {}),
                )
            )

        return results
