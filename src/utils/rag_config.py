"""
Configuration for RAG (Retrieval-Augmented Generation) system.

Design Philosophy:
- Separate RAG config from extraction config (different concerns)
- Chunking strategy is critical for RAG quality
- Embedding model choice affects cost and quality
- Retrieval parameters (top-k) affect answer relevance

Interview Note: This shows you understand RAG has its own
hyperparameters separate from extraction.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class RAGConfig(BaseModel):
    """
    Configuration for RAG pipeline.

    Rationale: RAG has many tunable parameters:
    - Chunk size (too small = lost context, too large = noisy retrieval)
    - Overlap (preserve context across chunks)
    - Top-k (how many chunks to retrieve)
    - Embedding model (quality vs cost)
    """

    # Chunking configuration
    chunk_size: int = Field(
        default=1000,
        description="Size of text chunks in tokens"
    )
    # Rationale: 1000 tokens ≈ 750 words
    # - Large enough to preserve context
    # - Small enough for focused retrieval
    # - Fits comfortably in LLM context with 5 chunks (5000 tokens)

    chunk_overlap: int = Field(
        default=200,
        description="Overlap between chunks in tokens"
    )
    # Rationale: 200 token overlap prevents splitting clauses
    # Example: Clause starts in chunk 1, ends in chunk 2
    # Overlap ensures we capture the full clause

    # Embedding configuration
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )
    # Rationale: text-embedding-3-small
    # - Cost: $0.02 / 1M tokens (cheap!)
    # - Dimensions: 1536 (good quality)
    # - Alternative: text-embedding-3-large (better quality, 3x cost)

    embedding_dimensions: int = Field(
        default=1536,
        description="Dimension of embedding vectors"
    )

    # Retrieval configuration
    retrieval_top_k: int = Field(
        default=5,
        description="Number of chunks to retrieve"
    )
    # Rationale: 5 chunks = 5000 tokens of context
    # - Enough context to answer most questions
    # - Not so much that LLM gets overwhelmed
    # - Can adjust based on query complexity

    retrieval_score_threshold: Optional[float] = Field(
        default=None,
        description="Minimum similarity score for retrieval (0-1)"
    )
    # Rationale: Filter out irrelevant chunks
    # - None = return top-k regardless of score
    # - 0.7 = only return chunks with >70% similarity
    # - Higher threshold = more precise but might miss relevant chunks

    # Generation configuration
    generation_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for answer generation"
    )

    generation_temperature: float = Field(
        default=0.0,
        description="Temperature for answer generation"
    )
    # Rationale: 0 = deterministic answers (good for Q&A)

    generation_max_tokens: int = Field(
        default=1000,
        description="Max tokens for generated answer"
    )

    # Vector store configuration
    vector_store_path: str = Field(
        default="./data/vector_store/chroma",
        description="Path to persist vector store"
    )
    # Rationale: Local persistence
    # - Data survives restarts
    # - Can version control (with .gitignore)
    # - Easy to deploy (just copy directory)

    collection_name: str = Field(
        default="contract_chunks",
        description="Name of ChromaDB collection"
    )

    # Metadata filtering
    enable_metadata_filtering: bool = Field(
        default=True,
        description="Whether to support metadata filtering in queries"
    )
    # Rationale: Allows queries like "Find in contracts from 2023"
    # Metadata: filename, date, contract_type, etc.


# Default configuration instance
DEFAULT_RAG_CONFIG = RAGConfig()