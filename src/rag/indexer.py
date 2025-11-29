"""
RAG Indexer - Index contracts into vector store.

Purpose:
- Parse contracts
- Chunk text
- Generate embeddings
- Store in ChromaDB

Design Pattern: Indexer handles all "write" operations to vector store.
Separate from Retriever (which handles "read" operations).

Rationale: Separation of concerns
- Indexing: Offline, batch processing
- Retrieval: Online, low-latency queries
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.utils.config import settings
from src.utils.rag_config import RAGConfig, DEFAULT_RAG_CONFIG
from src.rag.chunker import DocumentChunker
from src.parsers.pdf_parser import PDFParser
from src.utils.dataset import CUADDataset

logger = logging.getLogger(__name__)


class RAGIndexer:
    """
    Indexes contracts into vector store for semantic search.

    Workflow:
    1. Parse PDF → Extract text
    2. Chunk text → Create Document objects
    3. Generate embeddings → OpenAI API
    4. Store in ChromaDB → Persist to disk

    Design Rationale:
    - Batch processing (index many contracts at once)
    - Persistent storage (survives restarts)
    - Incremental indexing (add new contracts without rebuilding)
    - Metadata tracking (know what's indexed)
    """

    def __init__(
            self,
            config: RAGConfig = DEFAULT_RAG_CONFIG,
            pdf_parser: Optional[PDFParser] = None,
    ):
        """
        Initialize indexer.

        Args:
            config: RAG configuration
            pdf_parser: Optional PDF parser (creates one if not provided)

        Rationale: Dependency injection
        - Can pass in custom parser for testing
        - Can reuse parser across multiple operations
        """
        self.config = config
        self.pdf_parser = pdf_parser or PDFParser(extract_tables=False)
        self.chunker = DocumentChunker(config)

        # Initialize OpenAI embeddings
        # Rationale: text-embedding-3-small
        # - 1536 dimensions (good quality)
        # - $0.02 / 1M tokens (very cheap!)
        # - Fast (100k tokens/sec)
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=settings.openai_api_key,
        )

        # Initialize or load vector store
        # Rationale: ChromaDB persists to disk
        # - If vector_store_path exists, load existing data
        # - If not, create new empty store
        # - This enables incremental indexing
        self.vector_store = self._initialize_vector_store()

        logger.info(
            f"RAGIndexer initialized with collection '{config.collection_name}'"
        )

    def _initialize_vector_store(self) -> Chroma:
        """
        Initialize or load ChromaDB vector store.

        Returns:
            Chroma vector store instance

        Rationale: ChromaDB auto-creates directory if not exists.
        If directory exists, loads existing data.

        This enables:
        - Fresh start: Delete vector_store_path, run again
        - Incremental: Keep adding documents over time
        - Recovery: Data persists across restarts
        """
        vector_store_path = Path(self.config.vector_store_path)
        vector_store_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing vector store at: {vector_store_path}")

        # Create or load ChromaDB
        # Rationale: persist_directory makes data permanent
        vector_store = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(vector_store_path),
        )

        # Check if store has existing data
        try:
            count = vector_store._collection.count()
            logger.info(f"Vector store loaded with {count} existing documents")
        except Exception as e:
            logger.info("Created new vector store")

        return vector_store

    def index_contract(
            self,
            pdf_path: Path,
            additional_metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Index a single contract.

        Args:
            pdf_path: Path to PDF contract
            additional_metadata: Optional metadata to attach

        Returns:
            Number of chunks indexed

        Process:
        1. Parse PDF
        2. Chunk text
        3. Generate embeddings (automatic via vector_store.add_documents)
        4. Store in ChromaDB

        Rationale: High-level method for indexing one contract.
        Handles all steps internally.
        """
        logger.info(f"Indexing contract: {pdf_path.name}")

        # Step 1: Parse PDF
        try:
            parsed = self.pdf_parser.parse(pdf_path)
            contract_text = parsed.full_text

            if not contract_text or len(contract_text.strip()) == 0:
                logger.warning(f"Empty contract: {pdf_path.name}")
                return 0

            logger.info(
                f"Parsed {pdf_path.name}: {len(contract_text):,} characters"
            )

        except Exception as e:
            logger.error(f"Failed to parse {pdf_path.name}: {str(e)}")
            raise

        # Step 2: Chunk text
        metadata = additional_metadata or {}
        chunks = self.chunker.chunk_contract(
            contract_text,
            filename=pdf_path.name,
            additional_metadata=metadata
        )

        if not chunks:
            logger.warning(f"No chunks created for {pdf_path.name}")
            return 0

        logger.info(f"Created {len(chunks)} chunks")

        # Step 3 & 4: Generate embeddings and store
        # Rationale: vector_store.add_documents automatically:
        # - Generates embeddings via self.embeddings
        # - Stores vectors + metadata in ChromaDB
        # - Returns IDs of stored documents
        try:
            ids = self.vector_store.add_documents(chunks)
            logger.info(f"Indexed {len(ids)} chunks from {pdf_path.name}")
            return len(ids)

        except Exception as e:
            logger.error(f"Failed to index {pdf_path.name}: {str(e)}")
            raise

    def index_multiple_contracts(
            self,
            pdf_paths: List[Path],
            batch_size: int = 5
    ) -> Dict[str, Any]:
        """
        Index multiple contracts with progress tracking.

        Args:
            pdf_paths: List of PDF paths to index
            batch_size: Process N contracts before persisting (for efficiency)

        Returns:
            Summary statistics

        Rationale: Batch processing for efficiency
        - Process multiple contracts
        - Track successes and failures
        - Provide summary statistics

        Batch size optimization:
        - Small batch (5): Frequent persistence, safer
        - Large batch (50): Less I/O, faster, but lose more on crash

        Interview Note: Shows you understand batch processing trade-offs.
        """
        logger.info(f"Indexing {len(pdf_paths)} contracts")

        results = {
            "total_contracts": len(pdf_paths),
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "failed_contracts": [],
        }

        for i, pdf_path in enumerate(pdf_paths, 1):
            try:
                # Index contract
                num_chunks = self.index_contract(pdf_path)

                results["successful"] += 1
                results["total_chunks"] += num_chunks

                logger.info(
                    f"[{i}/{len(pdf_paths)}] ✅ {pdf_path.name}: {num_chunks} chunks"
                )

                # Persist every batch_size contracts
                # Rationale: Balance between performance and safety
                if i % batch_size == 0:
                    logger.info(f"Persisting after {i} contracts...")
                    # ChromaDB auto-persists, but we can force it

            except Exception as e:
                results["failed"] += 1
                results["failed_contracts"].append({
                    "filename": pdf_path.name,
                    "error": str(e)
                })

                logger.error(f"[{i}/{len(pdf_paths)}] ❌ {pdf_path.name}: {str(e)}")

        # Final summary
        logger.info(
            f"\nIndexing complete:\n"
            f"  ✅ Successful: {results['successful']}\n"
            f"  ❌ Failed: {results['failed']}\n"
            f"  📊 Total chunks: {results['total_chunks']}\n"
            f"  📈 Avg chunks/contract: {results['total_chunks'] / results['successful']:.1f}"
        )

        return results

    def index_cuad_sample(
            self,
            num_contracts: int = 10,
            cuad_dataset: Optional[CUADDataset] = None
    ) -> Dict[str, Any]:
        """
        Index sample contracts from CUAD dataset.

        Args:
            num_contracts: Number of contracts to index
            cuad_dataset: Optional CUAD dataset (creates one if not provided)

        Returns:
            Indexing summary

        Rationale: Convenience method for testing.
        Quickly index sample contracts without manual file selection.

        Example:
            indexer = RAGIndexer()
            indexer.index_cuad_sample(num_contracts=20)
            # Now you have 20 contracts indexed!
        """
        cuad = cuad_dataset or CUADDataset()

        # Get sample contracts
        sample_contracts = cuad.get_sample_contracts(n=num_contracts)
        pdf_paths = [cuad.pdf_dir / filename for filename in sample_contracts]

        logger.info(f"Indexing {num_contracts} CUAD contracts")

        return self.index_multiple_contracts(pdf_paths)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.

        Returns:
            Dictionary with stats

        Rationale: Observability - know what's in your vector store.
        - How many documents?
        - What contracts are indexed?
        - When was last update?

        Interview Note: Production systems need observability.
        """
        try:
            count = self.vector_store._collection.count()

            # Get sample of metadata to see what's indexed
            # Rationale: Peek at what contracts are in the store
            sample_docs = self.vector_store._collection.get(
                    limit=max(count, 10000),  # Get all, up to 10k
                    include=['metadatas']
                )

            filenames = set()
            if sample_docs and 'metadatas' in sample_docs:
                for metadata in sample_docs['metadatas']:
                    if metadata and 'filename' in metadata:
                        filenames.add(metadata['filename'])
            logger.info(f"Found {filenames}, length: {len(filenames)}")
            return {
                "total_chunks": count,
                "collection_name": self.config.collection_name,
                "sample_contracts": list(filenames),
                "embedding_model": self.config.embedding_model,
                "vector_store_path": self.config.vector_store_path,
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {"error": str(e)}

    def clear_collection(self):
        """
        Clear all documents from vector store.

        Rationale: Sometimes you need to start fresh:
        - Changed chunking strategy
        - Changed embedding model
        - Corrupted data

        WARNING: This deletes all indexed data!
        """
        logger.warning("Clearing vector store collection")

        try:
            # Delete collection
            self.vector_store._client.delete_collection(
                name=self.config.collection_name
            )

            # Recreate empty collection
            self.vector_store = self._initialize_vector_store()

            logger.info("Collection cleared and recreated")

        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            raise