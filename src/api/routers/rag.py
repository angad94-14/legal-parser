"""
RAG API endpoints.

UPDATED: Uses singleton retriever with conversation memory.
"""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from typing import Dict, Optional, Any
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

# ... (keep all existing indexing endpoints unchanged) ...

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