"""
API request/response models using Pydantic.

Design Rationale:
- Separate API models from internal models
- API models = external contract (stable)
- Internal models = implementation details (can change)

This separation allows us to:
- Version the API independently
- Change internal models without breaking API
- Add API-specific validation
- Transform data between layers

Interview Note: This shows you understand API design principles.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic.config import ConfigDict



# ============================================================================
# EXTRACTION API MODELS
# ============================================================================

class ExtractionStatus(str, Enum):
    """Status of an extraction job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ExtractRequest(BaseModel):
    """
    Request to extract clauses from a contract.

    Rationale: Accept either file upload or URL.
    - File upload: User uploads PDF directly
    - URL: User provides link to PDF (we download it)
    """
    url: Optional[HttpUrl] = Field(
        default=None,
        description="URL to PDF contract (alternative to file upload)"
    )

    extract_tables: bool = Field(
        default=False,
        description="Whether to extract tables from PDF"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "https://example.com/contract.pdf",
                    "extract_tables": False
                }
            ]
        }
    }


class ClauseResponse(BaseModel):
    """
    Response for a single extracted clause.

    Rationale: Simplified version of ExtractedClause for API.
    - Only includes what users need
    - Hides internal implementation details
    """
    text: Optional[str] = Field(
        default=None,
        description="Original text from contract"
    )

    answer: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Normalized/simplified answer"
    )

    confidence: str = Field(
        description="Confidence level: high, medium, low"
    )

    page_number: Optional[int] = Field(
        default=None,
        description="Page number where clause appears"
    )


class ExtractResponse(BaseModel):
    """
    Response containing extracted contract data.

    Rationale: Wraps the extracted clauses with metadata.
    - extraction_id: For tracking/retrieval
    - status: For async processing
    - timestamp: For auditing
    - clauses: The actual extracted data
    """
    extraction_id: str = Field(
        description="Unique ID for this extraction"
    )

    filename: str = Field(
        description="Name of the processed file"
    )

    status: ExtractionStatus = Field(
        description="Status of extraction job"
    )

    timestamp: datetime = Field(
        description="When extraction was performed"
    )

    clauses: Dict[str, Optional[ClauseResponse]] = Field(
        description="Extracted clauses by type"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extraction metadata (model, cost, etc.)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "extraction_id": "ext_abc123",
                    "filename": "contract.pdf",
                    "status": "completed",
                    "timestamp": "2024-11-27T10:30:00Z",
                    "clauses": {
                        "governing_law": {
                            "text": "This Agreement shall be governed by California law",
                            "answer": "California",
                            "confidence": "high",
                            "page_number": 5
                        }
                    },
                    "metadata": {
                        "model": "gpt-4o-mini",
                        "token_count": 8234,
                        "cost": 0.0089
                    }
                }
            ]
        }
    }


class BatchExtractRequest(BaseModel):
    """
    Request to extract from multiple contracts.

    Rationale: Batch processing is more efficient than N individual requests.
    - Single API call
    - Better resource utilization
    - Progress tracking
    """
    urls: List[HttpUrl] = Field(
        description="List of PDF URLs to process"
    )

    extract_tables: bool = Field(
        default=False,
        description="Whether to extract tables"
    )


class BatchExtractResponse(BaseModel):
    """
    Response for batch extraction.

    Rationale: Returns batch job ID for async processing.
    User can poll for results.
    """
    batch_id: str = Field(
        description="Unique ID for this batch job"
    )

    total_documents: int = Field(
        description="Total number of documents to process"
    )

    status: str = Field(
        description="Batch job status"
    )

    results: List[ExtractResponse] = Field(
        default_factory=list,
        description="Individual extraction results (populated when complete)"
    )


# ============================================================================
# RAG API MODELS
# ============================================================================

class IndexRequest(BaseModel):
    """
    Request to index contracts into RAG system.

    Rationale: Allows users to add documents to searchable corpus.
    """
    urls: List[HttpUrl] = Field(
        description="List of PDF URLs to index"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata to attach to documents"
    )


class IndexResponse(BaseModel):
    """
    Response for indexing operation.

    Rationale: Provides feedback on indexing success.
    """
    job_id: str = Field(
        description="Unique ID for this indexing job"
    )

    total_documents: int = Field(
        description="Number of documents indexed"
    )

    total_chunks: int = Field(
        description="Number of chunks created"
    )

    successful: int = Field(
        description="Number of successfully indexed documents"
    )

    failed: int = Field(
        description="Number of failed documents"
    )

    cost_estimate: float = Field(
        description="Estimated cost in USD"
    )


class SearchRequest(BaseModel):
    """
    Request to search indexed contracts.

    Rationale: Simple search without answer generation.
    Fast and cheap for exploration.
    """
    query: str = Field(
        description="Search query"
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return (1-20)"
    )

    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters"
    )


class SearchResult(BaseModel):
    """
    Single search result.

    Rationale: Contains chunk text + metadata for citation.
    """
    text: str = Field(
        description="Text content of chunk"
    )

    score: float = Field(
        description="Similarity score (0-1)"
    )

    metadata: Dict[str, Any] = Field(
        description="Chunk metadata (filename, page, etc.)"
    )


class SearchResponse(BaseModel):
    """
    Response for search request.

    Rationale: Returns ranked list of relevant chunks.
    """
    query: str = Field(
        description="Original query"
    )

    results: List[SearchResult] = Field(
        description="Search results ranked by relevance"
    )

    total_results: int = Field(
        description="Number of results returned"
    )


class QueryRequest(BaseModel):
    """
    Request for RAG question answering.

    Rationale: Full RAG (retrieval + generation) for natural language answers.
    """
    question: str = Field(
        description="Question to answer"
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve"
    )

    return_sources: bool = Field(
        default=True,
        description="Whether to return source documents"
    )

    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters"
    )


class Source(BaseModel):
    """
    Source document for citation.

    Rationale: Provides provenance for answers.
    """
    text: str = Field(
        description="Source text"
    )

    metadata: Dict[str, Any] = Field(
        description="Source metadata (filename, page, etc.)"
    )


class QueryResponse(BaseModel):
    """
    Response for RAG query.

    Rationale: Contains answer + sources for verification.
    """
    question: str = Field(
        description="Original question"
    )

    answer: str = Field(
        description="Generated answer"
    )

    sources: List[Source] = Field(
        default_factory=list,
        description="Source documents used for answer"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Query metadata (model, cost, etc.)"
    )


class RAGStatsResponse(BaseModel):
    """
    Response for RAG system statistics.

    Rationale: Observability - know what's in your system.
    """
    total_chunks: int = Field(
        description="Total number of indexed chunks"
    )

    total_documents: int = Field(
        description="Approximate number of documents (based on metadata)"
    )

    collection_name: str = Field(
        description="Name of vector store collection"
    )

    embedding_model: str = Field(
        description="Embedding model used"
    )


# ============================================================================
# COMMON MODELS
# ============================================================================

class HealthResponse(BaseModel):
    """
    Health check response.

    Rationale: Standard health endpoint for monitoring.
    - Load balancers check this
    - Monitoring systems check this
    - Deployment scripts check this
    """
    status: str = Field(
        description="Service status: healthy, degraded, unhealthy"
    )

    version: str = Field(
        description="API version"
    )

    timestamp: datetime = Field(
        description="Current server time"
    )

    dependencies: Dict[str, str] = Field(
        description="Status of dependencies (vector DB, LLM API, etc.)"
    )


class ErrorResponse(BaseModel):
    """
    Standard error response.

    Rationale: Consistent error format across all endpoints.
    - Makes client error handling easier
    - Provides debugging context
    """
    error: str = Field(
        description="Error type/code"
    )

    message: str = Field(
        description="Human-readable error message"
    )

    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )

    timestamp: datetime = Field(
        description="When error occurred"
    )