"""
RAG API endpoints.

UPDATED: Uses singleton retriever with conversation memory.
"""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks, UploadFile, File
from typing import Dict, Optional, Any, List
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
INDEXING_JOBS: Dict[str, Dict] = {}

# ============================================================================
# NEW: Singleton Retriever with Memory
# ============================================================================

# Module-level singleton for conversation memory
# Rationale: Single-user demo - all requests share same memory
# For multi-user: would use session-based retriever instances
_retriever_instance: Optional[RAGRetriever] = None


def get_retriever() -> RAGRetriever:
    """
    Get or create the singleton RAGRetriever instance.

    Rationale: Single retriever = shared conversation memory
    - Memory persists across API requests
    - Enables multi-turn conversations
    - Simple for single-user demo

    For production multi-user:
    - Would create retriever per session
    - Store in session dict: {session_id: retriever}
    - Clean up inactive sessions

    Returns:
        Singleton RAGRetriever instance with memory
    """
    global _retriever_instance

    if _retriever_instance is None:
        logger.info("Creating singleton RAGRetriever with conversation memory")
        config = RAGConfig()
        _retriever_instance = RAGRetriever(config)

    return _retriever_instance


# ============================================================================
# Endpoints (with memory support)
# ============================================================================

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


@router.post("/index/upload", status_code=status.HTTP_200_OK)
async def index_contracts_upload(
        files: List[UploadFile] = File(..., description="PDF contracts to index")
):
    """
    Index multiple contracts via direct file upload (synchronous).

    **For async indexing from URLs, use POST /rag/index instead.**

    **Process:**
    1. Upload PDF files
    2. Parse and chunk
    3. Generate embeddings
    4. Store in vector database

    **Time:** ~2-3 seconds per contract

    Args:
        files: List of PDF files

    Returns:
        Indexing results
    """
    import tempfile

    try:
        # Create RAGIndexer directly (separate from retriever)
        config = RAGConfig()
        indexer = RAGIndexer(config)  # ← Create indexer

        # Create temp directory for uploaded files
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Save all uploaded files
            saved_files = []
            for file in files:
                if not file.filename.lower().endswith('.pdf'):
                    logger.warning(f"Skipping non-PDF file: {file.filename}")
                    continue

                file_path = tmp_path / file.filename
                content = await file.read()
                file_path.write_bytes(content)
                saved_files.append(file_path)

            if not saved_files:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid PDF files provided"
                )

            # Index all files using indexer.index_multiple_contracts()
            logger.info(f"Indexing {len(saved_files)} contracts...")

            results = indexer.index_multiple_contracts(saved_files)

            # Get updated stats
            stats = indexer.get_stats()

            logger.info(
                f"Successfully indexed {results['successful']} contracts "
                f"({results['failed']} failed)"
            )

            return {
                "status": "success",
                "indexed_files": results['successful'],
                "failed_files": results['failed'],
                "filenames": [f.name for f in saved_files],
                "total_documents": len(stats["sample_contracts"]),  # ← Use sample_contracts length
                "total_chunks": stats["total_chunks"]
            }

    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}"
        )

@router.post("/search", response_model=SearchResponse)
async def search_contracts(request: SearchRequest):
    """
    Semantic search across indexed contracts.

    Note: Search does NOT use conversation memory.
    Pure search operations don't need conversational context.
    """
    logger.info(f"Search query: '{request.query}' (top_k={request.top_k})")

    try:
        # Use singleton retriever (but search doesn't use memory)
        retriever = get_retriever()

        # Clean filters
        filter_metadata = clean_filters(request.filters)

        # Perform search
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            filter_metadata=filter_metadata
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
    Answer questions using RAG with conversation memory.

    UPDATED: Now supports multi-turn conversations!

    Follow-up questions automatically use context from previous exchanges.

    Example conversation:
        Q1: "Which contracts have California law?"
        A1: "TechCorp, Euromedia, and Cybergy"

        Q2: "What are their payment terms?"  ← Uses memory!
        A2: "TechCorp: Net 30, Euromedia: Net 60, Cybergy: Net 45"

    The API automatically:
    - Loads conversation history
    - Includes it in LLM context
    - Saves the new exchange

    Memory: Last 5 exchanges (10 messages)
    """
    logger.info(f"Query: '{request.question}'")

    try:
        # Use singleton retriever (with memory!)
        retriever = get_retriever()

        # Get memory stats for logging
        mem_stats = retriever.get_memory_stats()
        logger.info(f"Memory state: {mem_stats['num_exchanges']} exchanges in history")

        # Generate answer (memory handled automatically!)
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
                    metadata=s["metadata"],
                    score=s["score"]
                )
                for s in result["sources"]
            ]

        logger.info(f"Generated answer ({len(result['answer'])} chars) with memory")

        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=sources,
            metadata={
                "num_sources": len(sources),
                "conversation_turns": mem_stats["num_exchanges"]
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
    """Get RAG system statistics."""
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


# ============================================================================
# NEW: Memory Management Endpoints
# ============================================================================

@router.post("/memory/clear")
async def clear_conversation_memory():
    """
    Clear conversation memory.

    Starts a fresh conversation by clearing all history.

    Use cases:
    - User wants to start new topic
    - Reset after demo
    - Clear after testing

    Example:
        curl -X POST http://localhost:8000/rag/memory/clear
    """
    try:
        retriever = get_retriever()
        retriever.clear_memory()

        logger.info("Conversation memory cleared via API")

        return {
            "status": "success",
            "message": "Conversation memory cleared"
        }

    except Exception as e:
        logger.error(f"Failed to clear memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear memory: {str(e)}"
        )


@router.get("/memory/stats")
async def get_memory_stats():
    """
    Get conversation memory statistics.

    Returns:
        Current memory state (number of messages, exchanges, etc.)

    Example:
        curl http://localhost:8000/rag/memory/stats

        Response:
        {
          "num_messages": 10,
          "num_exchanges": 5,
          "window_size": 5,
          "memory_full": true
        }
    """
    try:
        retriever = get_retriever()
        stats = retriever.get_memory_stats()

        # Add memory full indicator
        stats["memory_full"] = stats["num_exchanges"] >= stats["window_size"]

        return stats

    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get memory stats: {str(e)}"
        )


# ============================================================================
# Helper Functions
# ============================================================================

def clean_filters(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Clean filter dict for ChromaDB (remove Swagger placeholders)."""
    if not filters:
        return None

    cleaned = {
        k: v for k, v in filters.items()
        if v and not k.startswith("additionalProp")
    }

    return cleaned if cleaned else None