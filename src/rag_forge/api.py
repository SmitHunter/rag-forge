"""FastAPI web interface for RAG Forge."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rag_forge.config import Settings
from rag_forge.pipeline.rag import RAGPipeline

app = FastAPI(
    title="RAG Forge",
    description="Document Question-Answering with RAG",
    version="0.1.0",
)

pipeline: RAGPipeline | None = None


class QueryRequest(BaseModel):
    """Request body for query endpoint."""

    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)


class CitationResponse(BaseModel):
    """Citation in response."""

    source_number: int
    document_title: str
    text_snippet: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Response from query endpoint."""

    answer: str
    citations: list[CitationResponse]
    model: str
    total_tokens: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    indexed_chunks: int
    retrieval_strategy: str
    llm_provider: str


def get_pipeline() -> RAGPipeline:
    """Get or initialize the RAG pipeline."""
    global pipeline
    if pipeline is None:
        settings = Settings()
        pipeline = RAGPipeline(settings)
        if pipeline.index_exists():
            pipeline.load_index()
    return pipeline


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Check service health and index status."""
    p = get_pipeline()
    return HealthResponse(
        status="healthy" if p.chunk_count > 0 else "no_index",
        indexed_chunks=p.chunk_count,
        retrieval_strategy=p.settings.retrieval_strategy.value,
        llm_provider=p.settings.llm_provider.value,
    )


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    """Answer a question using RAG."""
    p = get_pipeline()

    if p.chunk_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Run 'rag-forge ingest' first.",
        )

    response = p.query(request.question, top_k=request.top_k)

    return QueryResponse(
        answer=response.answer,
        citations=[
            CitationResponse(
                source_number=c.source_number,
                document_title=c.document_title,
                text_snippet=c.text_snippet,
                relevance_score=c.relevance_score,
            )
            for c in response.citations
        ],
        model=response.model,
        total_tokens=response.total_tokens,
    )


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Forge - Document QA</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .loading { animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="container mx-auto px-4 py-8 max-w-4xl">
        <header class="mb-8">
            <h1 class="text-3xl font-bold text-gray-900">RAG Forge</h1>
            <p class="text-gray-600 mt-2">Document Question-Answering System</p>
        </header>

        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <form id="queryForm" class="space-y-4">
                <div>
                    <label for="question" class="block text-sm font-medium text-gray-700 mb-2">
                        Your Question
                    </label>
                    <textarea
                        id="question"
                        name="question"
                        rows="3"
                        class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        placeholder="Ask a question about the documents..."
                    ></textarea>
                </div>
                <div class="flex items-center gap-4">
                    <button
                        type="submit"
                        class="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition-colors"
                    >
                        Ask Question
                    </button>
                    <label class="text-sm text-gray-600">
                        Sources:
                        <select id="topK" class="ml-2 px-2 py-1 border rounded">
                            <option value="3">3</option>
                            <option value="5" selected>5</option>
                            <option value="10">10</option>
                        </select>
                    </label>
                </div>
            </form>
        </div>

        <div id="loading" class="hidden">
            <div class="bg-blue-50 rounded-lg p-6 text-center">
                <p class="text-blue-600 loading">Searching documents and generating answer...</p>
            </div>
        </div>

        <div id="results" class="hidden space-y-6">
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-lg font-semibold text-gray-900 mb-3">Answer</h2>
                <div id="answer" class="text-gray-700 whitespace-pre-wrap"></div>
            </div>

            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-lg font-semibold text-gray-900 mb-3">Sources</h2>
                <div id="sources" class="space-y-4"></div>
            </div>
        </div>

        <div id="error" class="hidden">
            <div class="bg-red-50 border border-red-200 rounded-lg p-6">
                <p class="text-red-600" id="errorMessage"></p>
            </div>
        </div>

        <footer class="mt-12 text-center text-gray-500 text-sm">
            <p>Powered by RAG Forge | <span id="status">Loading...</span></p>
        </footer>
    </div>

    <script>
        function escapeHtml(str) {
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        }

        async function checkHealth() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                document.getElementById('status').textContent =
                    `${data.indexed_chunks} chunks indexed | ${data.retrieval_strategy} retrieval | ${data.llm_provider} LLM`;
            } catch (e) {
                document.getElementById('status').textContent = 'Service unavailable';
            }
        }

        document.getElementById('queryForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const question = document.getElementById('question').value.trim();
            if (!question) return;

            const topK = parseInt(document.getElementById('topK').value);

            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('results').classList.add('hidden');
            document.getElementById('error').classList.add('hidden');

            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, top_k: topK })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Query failed');
                }

                const data = await response.json();

                document.getElementById('answer').textContent = data.answer;

                const sourcesDiv = document.getElementById('sources');
                sourcesDiv.innerHTML = data.citations.map(c => `
                    <div class="border-l-4 border-blue-400 pl-4 py-2">
                        <div class="flex justify-between items-start">
                            <span class="font-medium text-gray-900">[${c.source_number}] ${escapeHtml(c.document_title)}</span>
                            <span class="text-sm text-gray-500">Score: ${c.relevance_score.toFixed(3)}</span>
                        </div>
                        <p class="text-sm text-gray-600 mt-1">${escapeHtml(c.text_snippet)}</p>
                    </div>
                `).join('');

                document.getElementById('results').classList.remove('hidden');
            } catch (error) {
                document.getElementById('errorMessage').textContent = error.message;
                document.getElementById('error').classList.remove('hidden');
            } finally {
                document.getElementById('loading').classList.add('hidden');
            }
        });

        checkHealth();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def read_root() -> HTMLResponse:
    """Serve the web UI."""
    return HTMLResponse(content=HTML_TEMPLATE)
