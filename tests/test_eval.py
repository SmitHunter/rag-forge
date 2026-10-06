"""Tests for evaluation components."""

from pathlib import Path

import pytest

from rag_forge.config import EmbeddingProvider, LLMProvider, RetrievalStrategy, Settings
from rag_forge.eval.datasets import QADataset, QAExample
from rag_forge.eval.harness import EvalConfig, EvalHarness, default_eval_configs
from rag_forge.eval.metrics import (
    calculate_answer_correctness,
    calculate_faithfulness,
    calculate_mrr,
    calculate_recall_at_k,
    is_abstention,
)


class TestRetrievalMetrics:
    """Tests for retrieval metrics."""

    def test_recall_at_k_perfect(self) -> None:
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2"]

        recall = calculate_recall_at_k(retrieved, relevant, k=3)

        assert recall == 1.0

    def test_recall_at_k_partial(self) -> None:
        retrieved = ["doc1", "doc3", "doc4"]
        relevant = ["doc1", "doc2"]

        recall = calculate_recall_at_k(retrieved, relevant, k=3)

        assert recall == 0.5

    def test_recall_at_k_none(self) -> None:
        retrieved = ["doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc2"]

        recall = calculate_recall_at_k(retrieved, relevant, k=3)

        assert recall == 0.0

    def test_recall_at_k_respects_k(self) -> None:
        retrieved = ["doc3", "doc1", "doc2"]
        relevant = ["doc1", "doc2"]

        recall_1 = calculate_recall_at_k(retrieved, relevant, k=1)
        recall_3 = calculate_recall_at_k(retrieved, relevant, k=3)

        assert recall_1 == 0.0
        assert recall_3 == 1.0

    def test_mrr_first_position(self) -> None:
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1"]

        mrr = calculate_mrr(retrieved, relevant)

        assert mrr == 1.0

    def test_mrr_second_position(self) -> None:
        retrieved = ["doc2", "doc1", "doc3"]
        relevant = ["doc1"]

        mrr = calculate_mrr(retrieved, relevant)

        assert mrr == 0.5

    def test_mrr_not_found(self) -> None:
        retrieved = ["doc2", "doc3", "doc4"]
        relevant = ["doc1"]

        mrr = calculate_mrr(retrieved, relevant)

        assert mrr == 0.0


class TestAnswerMetrics:
    """Tests for answer quality metrics."""

    def test_faithfulness_high_overlap(self) -> None:
        answer = "Python is a programming language known for simplicity."
        context = ["Python is a programming language known for its simplicity and readability."]

        score = calculate_faithfulness(answer, context)

        assert score > 0.5

    def test_faithfulness_no_overlap(self) -> None:
        answer = "The weather is sunny today in California."
        context = ["Python is a programming language known for its simplicity."]

        score = calculate_faithfulness(answer, context)

        assert score < 0.3

    def test_answer_correctness_exact_match(self) -> None:
        generated = "Python is a programming language."
        reference = "Python is a programming language."

        score = calculate_answer_correctness(generated, reference)

        assert score == 1.0

    def test_answer_correctness_partial_match(self) -> None:
        generated = "Python is popular for data science."
        reference = "Python is a programming language used for data science."

        score = calculate_answer_correctness(generated, reference)

        assert 0.3 < score < 1.0

    def test_answer_correctness_no_match(self) -> None:
        generated = "The capital of France is Paris."
        reference = "Python is a programming language."

        score = calculate_answer_correctness(generated, reference)

        assert score < 0.3

    def test_abstention_detection(self) -> None:
        assert is_abstention("I cannot answer this based on the provided context.")
        assert is_abstention("This information is not in the context.")
        assert not is_abstention("The answer is 42.")
        assert is_abstention("I don't have enough information to answer.")


class TestQADataset:
    """Tests for QA dataset handling."""

    def test_create_example(self) -> None:
        example = QAExample(
            id="q1",
            question="What is Python?",
            answer="A programming language.",
            relevant_doc_ids=["doc1"],
        )

        assert example.id == "q1"
        assert example.question == "What is Python?"

    def test_example_to_from_dict(self) -> None:
        example = QAExample(
            id="q1",
            question="What is Python?",
            answer="A programming language.",
            relevant_doc_ids=["doc1"],
        )

        d = example.to_dict()
        restored = QAExample.from_dict(d)

        assert restored.id == example.id
        assert restored.question == example.question
        assert restored.relevant_doc_ids == example.relevant_doc_ids

    def test_dataset_save_load(self, tmp_path: Path) -> None:
        examples = [
            QAExample(id="q1", question="Q1?", answer="A1", relevant_doc_ids=["doc1"]),
            QAExample(id="q2", question="Q2?", answer="A2", relevant_doc_ids=["doc2"]),
        ]
        dataset = QADataset(name="Test", description="Test dataset", examples=examples)

        path = tmp_path / "test_qa.json"
        dataset.save(path)
        loaded = QADataset.load(path)

        assert len(loaded) == 2
        assert loaded.name == "Test"
        assert loaded.examples[0].question == "Q1?"


class TestEvalHarness:
    """Tests for the evaluation harness."""

    @pytest.fixture
    def test_documents(self) -> list[dict[str, str]]:
        return [
            {
                "id": "doc1",
                "title": "Python Guide",
                "text": "Python is a programming language. It is known for simplicity. Python is widely used.",
            },
            {
                "id": "doc2",
                "title": "ML Guide",
                "text": "Machine learning is AI. It uses data to learn. ML is popular.",
            },
        ]

    @pytest.fixture
    def test_qa_dataset(self) -> QADataset:
        return QADataset(
            name="Test QA",
            description="Test",
            examples=[
                QAExample(
                    id="q1",
                    question="What is Python?",
                    answer="Python is a programming language.",
                    relevant_doc_ids=["doc1"],
                ),
            ],
        )

    def test_default_eval_configs_match_readme(self) -> None:
        configs = default_eval_configs()
        names = [c.name for c in configs]
        assert names == [
            "BM25 Only",
            "Dense Only",
            "Hybrid (BM25=0.3)",
            "Hybrid (BM25=0.5)",
            "Dense + Rerank",
            "Hybrid + Rerank",
        ]

    def test_eval_config_to_settings(self) -> None:
        base = Settings()
        config = EvalConfig(
            name="Test",
            retrieval_strategy=RetrievalStrategy.BM25,
            use_reranking=True,
        )

        settings = config.to_settings(base)

        assert settings.retrieval_strategy == RetrievalStrategy.BM25
        assert settings.use_reranking is True

    def test_run_single_evaluation(
        self, test_documents: list[dict[str, str]], test_qa_dataset: QADataset, tmp_path: Path
    ) -> None:
        base_settings = Settings(
            data_dir=tmp_path / "data",
            cache_dir=tmp_path / "cache",
            embedding_provider=EmbeddingProvider.OFFLINE,
            llm_provider=LLMProvider.OFFLINE,
        )
        config = EvalConfig(
            name="BM25 Test",
            retrieval_strategy=RetrievalStrategy.BM25,
        )

        harness = EvalHarness(base_settings, test_documents, test_qa_dataset)
        results = harness.run_evaluation([config], show_progress=False)

        assert len(results) == 1
        assert results[0].config_name == "BM25 Test"
        assert results[0].retrieval_metrics.num_examples == 1

    def test_results_markdown(
        self, test_documents: list[dict[str, str]], test_qa_dataset: QADataset, tmp_path: Path
    ) -> None:
        base_settings = Settings(
            data_dir=tmp_path / "data",
            cache_dir=tmp_path / "cache",
            embedding_provider=EmbeddingProvider.OFFLINE,
            llm_provider=LLMProvider.OFFLINE,
        )
        config = EvalConfig(name="Test", retrieval_strategy=RetrievalStrategy.BM25)

        harness = EvalHarness(base_settings, test_documents, test_qa_dataset)
        harness.run_evaluation([config], show_progress=False)
        markdown = harness.get_results_markdown()

        assert "Test" in markdown
        assert "Recall@1" in markdown
        assert "|" in markdown
