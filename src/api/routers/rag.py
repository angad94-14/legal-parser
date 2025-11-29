"""
RAG API endpoints.

Handles document indexing and semantic search/Q&A.

Endpoints:
- POST /rag/index - Index contracts into vector store
- GET /rag/search - Semantic search
- POST /rag/query - Question answering (RAG)
- GET /rag/stats - Vector store statistics

Design Rationale:
- Indexing is async (can take time for many docs)
- Search is fast (pure vector search)
- Query involves LLM (slower, costs money)

Interview Note: Shows understanding of RAG architecture.
"""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from typing import Dict
import uuid
from datetime import datetime
import logging
import tempfile
from pathlib import Path

from src.api.models import (
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    QueryRequest,
    QueryResponse,
    Source,
    RAGStatsResponse,
)
from src.rag.indexer import RAGIndexer
from src.rag.retriever import RAGRetriever
from src.utils.rag_config import RAGConfig
from src.parsers.pdf_parser import PDFParser

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# In-memory storage for indexing jobs
# Production: Use Redis or database
INDEXING_JOBS: Dict[str, Dict] = {}


@router.post("/index", response_model=IndexResponse, status_code=status.HTTP_202_ACCEPTED)
async def index_contracts(
        request: IndexRequest,
        background_tasks: BackgroundTasks
):
    """
    Index contracts into RAG system.

    **Process:**
    1. Download PDFs from URLs
    2. Parse PDFs → Extract text
    3. Chunk text
    4. Generate embeddings
    5. Store in vector database

    **Cost:** ~$0.002 per contract (embeddings only)
    **Time:** ~2 seconds per contract

    **Note:** Returns immediately with job_id. Processing happens in background.

    **Example:**
```bash
    curl -X POST "http://localhost:8000/rag/index" \\
         -H "Content-Type: application/json" \\
         -d '{
           "urls": [
             "https://example.com/contract1.pdf",
             "https://example.com/contract2.pdf"
           ]
         }'
```

    Args:
        request: IndexRequest with PDF URLs
        background_tasks: FastAPI background tasks

    Returns:
        IndexResponse with job_id (202 Accepted)
    """
    job_id = f"idx_{uuid.uuid4().hex[:12]}"

    logger.info(f"[{job_id}] Starting indexing for {len(request.urls)} documents")

    # Create job record
    INDEXING_JOBS[job_id] = {
        "status": "pending",
        "total_documents": len(request.urls),
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "created_at": datetime.utcnow()
    }

    # Queue background task
    # Rationale: Don't block API response waiting for indexing
    background_tasks.add_task(
        process_indexing,
        job_id,
        request.urls,
        request.metadata
    )

    return IndexResponse(
        job_id=job_id,
        total_documents=len(request.urls),
        total_chunks=0,  # Will be updated when processing completes
        successful=0,
        failed=0,
        cost_estimate=len(request.urls) * 0.002  # Rough estimate
    )


async def process_indexing(job_id: str, urls: list, metadata: dict = None):
    """
    Background task to process indexing.

    Rationale: Long-running task shouldn't block API response.
    - User gets job_id immediately
    - Can poll for status
    - Processing happens in background

    Args:
        job_id: Unique job ID
        urls: List of PDF URLs
        metadata: Optional metadata to attach
    """
    logger.info(f"[{job_id}] Background indexing started")

    try:
        # Update status
        INDEXING_JOBS[job_id]["status"] = "processing"

        # Initialize indexer
        config = RAGConfig()
        indexer = RAGIndexer(config)

        # Download and index each PDF
        pdf_paths = []

        for i, url in enumerate(urls, 1):
            try:
                # Download PDF
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.get(str(url), timeout=30.0)
                    response.raise_for_status()
                    pdf_content = response.content

                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(pdf_content)
                    pdf_paths.append(Path(tmp_file.name))

                logger.info(f"[{job_id}] Downloaded {i}/{len(urls)}")

            except Exception as e:
                logger.error(f"[{job_id}] Failed to download {url}: {e}")
                INDEXING_JOBS[job_id]["failed"] += 1

        # Index all PDFs
        if pdf_paths:
            results = indexer.index_multiple_contracts(pdf_paths)

            INDEXING_JOBS[job_id]["successful"] = results["successful"]
            INDEXING_JOBS[job_id]["failed"] += results["failed"]
            INDEXING_JOBS[job_id]["total_chunks"] = results["total_chunks"]

        # Cleanup temp files
        for path in pdf_paths:
            try:
                path.unlink()
            except:
                pass

        # Update status
        INDEXING_JOBS[job_id]["status"] = "completed"
        logger.info(
            f"[{job_id}] Indexing complete: "
            f"{INDEXING_JOBS[job_id]['successful']} successful, "
            f"{INDEXING_JOBS[job_id]['failed']} failed"
        )

    except Exception as e:
        logger.error(f"[{job_id}] Indexing failed: {e}", exc_info=True)
        INDEXING_JOBS[job_id]["status"] = "failed"
        INDEXING_JOBS[job_id]["error"] = str(e)


@router.get("/index/{job_id}")
async def get_indexing_status(job_id: str):
    """
    Get status of indexing job.

    **Use case:** Poll for indexing progress

    **Example:**
```bash
    curl "http://localhost:8000/rag/index/idx_abc123"
```

    Args:
        job_id: Unique job ID

    Returns:
        Job status and progress
    """
    if job_id not in INDEXING_JOBS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    return INDEXING_JOBS[job_id]


@router.post("/search", response_model=SearchResponse)
async def search_contracts(request: SearchRequest):
    """
    Semantic search across indexed contracts.

    **Process:**
    1. Embed query
    2. Vector similarity search
    3. Return top-k chunks

    **Cost:** ~$0.00001 (embedding only, no LLM)
    **Time:** ~100-200ms

    **Example:**
```bash
    curl -X POST "http://localhost:8000/rag/search" \\
         -H "Content-Type: application/json" \\
         -d '{
           "query": "governing law",
           "top_k": 5
         }'
```

    Args:
        request: SearchRequest with query

    Returns:
        SearchResponse with ranked results
    """
    logger.info(f"Search query: '{request.query}' (top_k={request.top_k})")

    try:
        # Initialize retriever
        retriever = RAGRetriever()

        # Perform search
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            filter_metadata=request.filters
        )

        # Convert to API format
        search_results = [
            SearchResult(
                text=r["text"],
                score=r["score"],
                metadata=r["metadata"]
            )
            for r in results
        ]

        logger.info(f"Found {len(search_results)} results")

        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results)
        )

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse)
async def query_contracts(request: QueryRequest):
    """
    Answer questions using RAG (retrieval + generation).

    **Process:**
    1. Embed question
    2. Retrieve relevant chunks
    3. Generate answer with LLM
    4. Return answer + sources

    **Cost:** ~$0.01 per query (LLM generation)
    **Time:** ~2-5 seconds

    **Example:**
```bash
    curl -X POST "http://localhost:8000/rag/query" \\
         -H "Content-Type: application/json" \\
         -d '{
           "question": "Which contracts have California governing law?",
           "return_sources": true
         }'
```

    Args:
        request: QueryRequest with question

    Returns:
        QueryResponse with answer and sources
    """
    logger.info(f"Query: '{request.question}'")

    try:
        # Initialize retriever
        retriever = RAGRetriever()

        # Generate answer
        result = retriever.answer(
            query=request.question,
            return_sources=request.return_sources
        )

        # Convert sources to API format
        sources = []
        if request.return_sources and "sources" in result:
            sources = [
                Source(
                    text=s["text"],
                    metadata=s["metadata"]
                )
                for s in result["sources"]
            ]

        logger.info(f"Generated answer ({len(result['answer'])} chars)")

        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=sources,
            metadata={
                "num_sources": len(sources)
            }
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@router.get("/stats", response_model=RAGStatsResponse)
async def get_rag_stats():
    """
    Get RAG system statistics.

    **Returns:**
    - Total indexed chunks
    - Approximate document count
    - Collection info
    - Embedding model

    **Example:**
```bash
    curl "http://localhost:8000/rag/stats"
```

    Returns:
        RAGStatsResponse with system stats
    """
    try:
        config = RAGConfig()
        indexer = RAGIndexer(config)

        stats = indexer.get_stats()

        return RAGStatsResponse(
            total_chunks=stats["total_chunks"],
            total_documents=len(stats.get("sample_contracts", [])),
            collection_name=stats["collection_name"],
            embedding_model=stats["embedding_model"]
        )

    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )