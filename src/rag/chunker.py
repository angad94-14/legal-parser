"""
Document chunking for RAG.

Rationale: Chunking is critical for RAG quality.
- Too large: Noisy, irrelevant content in chunks
- Too small: Lost context, incomplete information
- No overlap: Clauses split across chunks

Design: Use LangChain's RecursiveCharacterTextSplitter
- Tries to split on paragraph boundaries first
- Falls back to sentence boundaries
- Then word boundaries
- Preserves semantic units
"""

from typing import List, Dict, Any
from pathlib import Path
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.utils.rag_config import RAGConfig, DEFAULT_RAG_CONFIG
from src.utils.tokens import count_tokens

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Chunks documents for RAG indexing.

    Design Pattern: Single Responsibility
    - Only does chunking (not parsing, not embedding)
    - Reusable across different document types
    - Configurable via RAGConfig

    Interview Note: Shows you understand separation of concerns.
    Chunking is separate from parsing and embedding.
    """

    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG):
        """
        Initialize chunker.

        Args:
            config: RAG configuration

        Rationale: Dependency injection for testability.
        """
        self.config = config

        # Create LangChain text splitter
        # Rationale: RecursiveCharacterTextSplitter is smart:
        # - Tries to split on \n\n (paragraphs) first
        # - Falls back to \n (lines)
        # - Falls back to spaces (words)
        # - Preserves semantic units better than naive splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=count_tokens,  # Use token count, not characters
            separators=["\n\n", "\n", ". ", " ", ""],  # Split hierarchy
        )

        logger.info(
            f"DocumentChunker initialized: "
            f"chunk_size={config.chunk_size}, "
            f"overlap={config.chunk_overlap}"
        )

    def chunk_text(
            self,
            text: str,
            metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Chunk text into LangChain Documents.

        Args:
            text: Raw text to chunk
            metadata: Metadata to attach to all chunks

        Returns:
            List of LangChain Document objects

        Rationale: Returns LangChain Document objects (not just strings)
        because Documents carry metadata (filename, page, etc.)
        which is essential for citation and filtering.

        Example:
            text = "This is a contract..."
            metadata = {"filename": "contract.pdf", "page": 1}
            chunks = chunker.chunk_text(text, metadata)
            # chunks[0].page_content = "This is a contract..."
            # chunks[0].metadata = {"filename": "contract.pdf", "page": 1, "chunk_index": 0}
        """
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided to chunk_text")
            return []

        # Initialize metadata
        base_metadata = metadata or {}

        # Split text into chunks
        # Rationale: text_splitter.create_documents automatically:
        # - Splits text according to configured strategy
        # - Creates Document objects
        # - Attaches metadata to each chunk
        documents = self.text_splitter.create_documents(
            texts=[text],
            metadatas=[base_metadata]
        )

        # Add chunk-specific metadata
        # Rationale: Track chunk index for debugging and ordering
        for i, doc in enumerate(documents):
            doc.metadata["chunk_index"] = i
            doc.metadata["chunk_token_count"] = count_tokens(doc.page_content)

        logger.info(
            f"Chunked text into {len(documents)} chunks "
            f"(avg {sum(d.metadata['chunk_token_count'] for d in documents) / len(documents):.0f} tokens/chunk)"
        )

        return documents

    def chunk_contract(
            self,
            contract_text: str,
            filename: str,
            additional_metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Chunk a contract with contract-specific metadata.

        Args:
            contract_text: Full contract text
            filename: Contract filename
            additional_metadata: Additional metadata (extracted clauses, etc.)

        Returns:
            List of Document chunks with rich metadata

        Rationale: Convenience method for chunking contracts.
        Automatically adds contract-specific metadata.

        Metadata includes:
        - filename: For citation
        - document_type: "contract"
        - Any extracted clause data (optional)

        This metadata enables queries like:
        - "Find governing law clauses in all contracts"
        - "Search only in contracts from 2023"
        """
        metadata = {
            "filename": filename,
            "document_type": "contract",
            "source": "cuad",
        }

        # Add any additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)

        return self.chunk_text(contract_text, metadata)

    def chunk_multiple_contracts(
            self,
            contracts: List[Dict[str, Any]]
    ) -> List[Document]:
        """
        Chunk multiple contracts at once.

        Args:
            contracts: List of dicts with 'text', 'filename', and optional metadata

        Returns:
            All chunks from all contracts

        Example:
            contracts = [
                {"text": "contract1...", "filename": "c1.pdf"},
                {"text": "contract2...", "filename": "c2.pdf"},
            ]
            all_chunks = chunker.chunk_multiple_contracts(contracts)

        Rationale: Batch chunking for efficient indexing.
        Process all contracts at once rather than one-by-one.
        """
        all_chunks = []

        for contract in contracts:
            text = contract.get("text", "")
            filename = contract.get("filename", "unknown.pdf")
            metadata = contract.get("metadata", {})

            chunks = self.chunk_contract(text, filename, metadata)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked {len(contracts)} contracts into {len(all_chunks)} total chunks"
        )

        return all_chunks


# Convenience function
def chunk_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
) -> List[str]:
    """
    Simple function to chunk text without metadata.

    Args:
        text: Text to chunk
        chunk_size: Chunk size in tokens
        chunk_overlap: Overlap in tokens

    Returns:
        List of text chunks (strings)

    Rationale: Sometimes you just want strings, not Document objects.
    Useful for quick testing or simple use cases.
    """
    config = RAGConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunker = DocumentChunker(config)
    documents = chunker.chunk_text(text)
    return [doc.page_content for doc in documents]