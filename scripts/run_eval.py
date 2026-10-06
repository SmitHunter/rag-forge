#!/usr/bin/env python3
"""Run the RAG evaluation harness and output results.

This script runs a comprehensive evaluation comparing different retrieval
configurations and outputs the results in various formats.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_forge.config import EmbeddingProvider, LLMProvider, Settings
from rag_forge.eval.datasets import QADataset
from rag_forge.eval.harness import EvalHarness, default_eval_configs


def load_documents(data_dir: Path) -> list[dict[str, str]]:
    """Load documents from the data directory."""
    import json

    docs_path = data_dir / "documents.json"
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents not found at {docs_path}. Run download_data.py first.")

    with open(docs_path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation harness")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing documents",
    )
    parser.add_argument(
        "--qa-file",
        type=Path,
        default=Path("data/qa_dataset.json"),
        help="Path to QA dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results.json"),
        help="Output file for results",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Alias for --llm-provider offline (retrieval-only baseline)",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        choices=["sentence_transformers", "offline", "openai"],
        default="sentence_transformers",
        help="Embedding provider to use",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["offline", "local", "openai", "anthropic"],
        default="offline",
        help="LLM provider for answer generation (default: offline template)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override LLM model name (e.g. llama3.2:1b)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated config names to run (default: all). "
        "Example: 'Dense Only,Hybrid (BM25=0.5)'",
    )
    args = parser.parse_args()

    print("Loading documents...")
    documents = load_documents(args.data_dir)
    print(f"Loaded {len(documents)} documents")

    print("Loading QA dataset...")
    qa_dataset = QADataset.load(args.qa_file)
    print(f"Loaded {len(qa_dataset)} QA examples")

    embedding_provider = EmbeddingProvider(args.embedding_provider)
    llm_provider = LLMProvider.OFFLINE if args.offline else LLMProvider(args.llm_provider)

    settings_kwargs: dict[str, object] = {
        "data_dir": args.data_dir,
        "embedding_provider": embedding_provider,
        "llm_provider": llm_provider,
    }
    if args.llm_model:
        settings_kwargs["llm_model"] = args.llm_model

    base_settings = Settings(**settings_kwargs)
    configs = default_eval_configs()
    if args.configs:
        wanted = {name.strip() for name in args.configs.split(",") if name.strip()}
        known = {c.name for c in configs}
        unknown = wanted - known
        if unknown:
            parser.error(
                f"Unknown config(s): {', '.join(sorted(unknown))}. "
                f"Known: {', '.join(sorted(known))}"
            )
        configs = [c for c in configs if c.name in wanted]

    print("\nStarting evaluation...")
    print("=" * 60)

    harness = EvalHarness(base_settings, documents, qa_dataset)
    harness.run_evaluation(configs)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60 + "\n")

    harness.print_results_table()

    print("\n" + "-" * 60)
    print("MARKDOWN TABLE (for README)")
    print("-" * 60 + "\n")
    print(harness.get_results_markdown())

    harness.save_results(args.output)
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
