"""Command-line interface for RAG Forge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag_forge.config import (
    ChunkingStrategy,
    EmbeddingProvider,
    LLMProvider,
    RetrievalStrategy,
    Settings,
)
from rag_forge.eval.datasets import QADataset
from rag_forge.eval.harness import EvalHarness, default_eval_configs
from rag_forge.pipeline.rag import RAGPipeline
from rag_forge.providers.local_llm_provider import OllamaNotAvailableError, check_ollama_available

app = typer.Typer(
    name="rag-forge",
    help="Production-grade RAG document QA with evaluation harness.",
    add_completion=False,
)
console = Console()


def load_documents(data_dir: Path) -> list[dict[str, str]]:
    """Load documents from a data directory."""
    documents = []
    docs_path = data_dir / "documents.json"

    if docs_path.exists():
        with open(docs_path) as f:
            documents = json.load(f)
    else:
        for txt_file in data_dir.glob("*.txt"):
            with open(txt_file) as f:
                content = f.read()
            documents.append(
                {
                    "id": txt_file.stem,
                    "title": txt_file.stem.replace("_", " ").title(),
                    "text": content,
                }
            )

    return documents


@app.command()
def ingest(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Directory containing documents"),
    ] = Path("data"),
    cache_dir: Annotated[
        Path,
        typer.Option("--cache-dir", "-c", help="Directory for index cache"),
    ] = Path(".cache"),
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", help="Target chunk size in tokens"),
    ] = 512,
    chunk_overlap: Annotated[
        int,
        typer.Option("--chunk-overlap", help="Overlap between chunks"),
    ] = 50,
    chunking_strategy: Annotated[
        ChunkingStrategy,
        typer.Option("--chunking", help="Chunking strategy"),
    ] = ChunkingStrategy.FIXED_SIZE,
    retrieval_strategy: Annotated[
        RetrievalStrategy,
        typer.Option("--retrieval", help="Retrieval strategy"),
    ] = RetrievalStrategy.HYBRID,
    embedding_provider: Annotated[
        EmbeddingProvider,
        typer.Option("--embedding-provider", help="Embedding provider"),
    ] = EmbeddingProvider.SENTENCE_TRANSFORMERS,
) -> None:
    """Ingest documents and build the retrieval index."""
    settings = Settings(
        data_dir=data_dir,
        cache_dir=cache_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunking_strategy=chunking_strategy,
        retrieval_strategy=retrieval_strategy,
        embedding_provider=embedding_provider,
    )

    documents = load_documents(data_dir)
    if not documents:
        console.print(f"[red]No documents found in {data_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Found {len(documents)} documents[/green]")

    pipeline = RAGPipeline(settings)
    chunks = pipeline.ingest_documents(documents)
    console.print(f"[green]Created {len(chunks)} chunks[/green]")

    pipeline.build_index()
    pipeline.save_index()
    console.print(f"[green]Index saved to {cache_dir}[/green]")


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to answer")],
    cache_dir: Annotated[
        Path,
        typer.Option("--cache-dir", "-c", help="Directory with index cache"),
    ] = Path(".cache"),
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Number of chunks to retrieve"),
    ] = 5,
    retrieval_strategy: Annotated[
        RetrievalStrategy,
        typer.Option("--retrieval", help="Retrieval strategy"),
    ] = RetrievalStrategy.HYBRID,
    llm_provider: Annotated[
        LLMProvider,
        typer.Option("--llm-provider", help="LLM provider"),
    ] = LLMProvider.OFFLINE,
    llm_model: Annotated[
        str | None,
        typer.Option("--llm-model", help="LLM model name (e.g. llama3.2:1b, gpt-4o-mini)"),
    ] = None,
    embedding_provider: Annotated[
        EmbeddingProvider,
        typer.Option("--embedding-provider", help="Embedding provider"),
    ] = EmbeddingProvider.SENTENCE_TRANSFORMERS,
    show_sources: Annotated[
        bool,
        typer.Option("--show-sources", "-s", help="Show source citations"),
    ] = True,
    use_reranking: Annotated[
        bool,
        typer.Option("--rerank", help="Use cross-encoder reranking"),
    ] = False,
) -> None:
    """Answer a question using RAG."""
    if llm_provider == LLMProvider.LOCAL:
        available, message = check_ollama_available()
        if not available:
            console.print(f"[red]{message}[/red]")
            raise typer.Exit(1)
        console.print(f"[dim]{message}[/dim]")

    settings_kwargs: dict[str, object] = {
        "cache_dir": cache_dir,
        "retrieval_strategy": retrieval_strategy,
        "retrieval_top_k": top_k,
        "llm_provider": llm_provider,
        "embedding_provider": embedding_provider,
        "use_reranking": use_reranking,
    }
    if llm_model:
        settings_kwargs["llm_model"] = llm_model

    settings = Settings(**settings_kwargs)
    pipeline = RAGPipeline(settings)

    if not pipeline.index_exists():
        console.print("[red]No index found. Run 'rag-forge ingest' first.[/red]")
        raise typer.Exit(1)

    pipeline.load_index()

    try:
        with console.status("[bold green]Thinking..."):
            response = pipeline.query(question, top_k=top_k)
    except OllamaNotAvailableError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    console.print(Panel(response.answer, title="Answer", border_style="green"))

    if show_sources:
        table = Table(title="Sources", show_header=True, header_style="bold cyan")
        table.add_column("Source", style="cyan", no_wrap=True)
        table.add_column("Document", style="magenta")
        table.add_column("Score", justify="right")
        table.add_column("Preview", max_width=50)

        for citation in response.citations:
            table.add_row(
                f"[{citation.source_number}]",
                citation.document_title,
                f"{citation.relevance_score:.3f}",
                citation.text_snippet[:100] + "..."
                if len(citation.text_snippet) > 100
                else citation.text_snippet,
            )

        console.print(table)


@app.command()
def evaluate(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Directory containing documents"),
    ] = Path("data"),
    qa_file: Annotated[
        Path,
        typer.Option("--qa-file", "-q", help="Path to QA dataset JSON"),
    ] = Path("data/qa_dataset.json"),
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file for results"),
    ] = None,
    embedding_provider: Annotated[
        EmbeddingProvider,
        typer.Option("--embedding-provider", help="Embedding provider"),
    ] = EmbeddingProvider.SENTENCE_TRANSFORMERS,
    llm_provider: Annotated[
        LLMProvider,
        typer.Option("--llm-provider", help="LLM provider"),
    ] = LLMProvider.OFFLINE,
    llm_model: Annotated[
        str | None,
        typer.Option("--llm-model", help="LLM model name (e.g. llama3.2:1b)"),
    ] = None,
) -> None:
    """Run evaluation across multiple retrieval configurations."""
    if not qa_file.exists():
        console.print(f"[red]QA dataset not found: {qa_file}[/red]")
        raise typer.Exit(1)

    documents = load_documents(data_dir)
    if not documents:
        console.print(f"[red]No documents found in {data_dir}[/red]")
        raise typer.Exit(1)

    qa_dataset = QADataset.load(qa_file)
    console.print(f"[green]Loaded {len(qa_dataset)} QA examples[/green]")

    settings_kwargs: dict[str, object] = {
        "data_dir": data_dir,
        "embedding_provider": embedding_provider,
        "llm_provider": llm_provider,
    }
    if llm_model:
        settings_kwargs["llm_model"] = llm_model

    base_settings = Settings(**settings_kwargs)
    harness = EvalHarness(base_settings, documents, qa_dataset)
    harness.run_evaluation(default_eval_configs())

    harness.print_results_table()

    console.print("\n[bold]Markdown Table:[/bold]")
    console.print(harness.get_results_markdown())

    if output:
        harness.save_results(output)
        console.print(f"\n[green]Results saved to {output}[/green]")


@app.command()
def serve(
    cache_dir: Annotated[
        Path,
        typer.Option("--cache-dir", "-c", help="Directory with index cache"),
    ] = Path(".cache"),
    host: Annotated[
        str,
        typer.Option("--host", help="Server host"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Server port"),
    ] = 8000,
    retrieval_strategy: Annotated[
        RetrievalStrategy,
        typer.Option("--retrieval", help="Retrieval strategy"),
    ] = RetrievalStrategy.HYBRID,
    llm_provider: Annotated[
        LLMProvider,
        typer.Option("--llm-provider", help="LLM provider"),
    ] = LLMProvider.OFFLINE,
    llm_model: Annotated[
        str | None,
        typer.Option("--llm-model", help="LLM model name (e.g. llama3.2:1b)"),
    ] = None,
    embedding_provider: Annotated[
        EmbeddingProvider,
        typer.Option("--embedding-provider", help="Embedding provider"),
    ] = EmbeddingProvider.SENTENCE_TRANSFORMERS,
) -> None:
    """Start the FastAPI web server."""
    import os

    os.environ["RAG_FORGE_CACHE_DIR"] = str(cache_dir)
    os.environ["RAG_FORGE_RETRIEVAL_STRATEGY"] = retrieval_strategy.value
    os.environ["RAG_FORGE_LLM_PROVIDER"] = llm_provider.value
    os.environ["RAG_FORGE_EMBEDDING_PROVIDER"] = embedding_provider.value
    if llm_model:
        os.environ["RAG_FORGE_LLM_MODEL"] = llm_model

    import uvicorn

    console.print(f"[green]Starting server at http://{host}:{port}[/green]")
    uvicorn.run("rag_forge.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
