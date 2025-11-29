"""
Demo script for RAG system.

Usage:
    # Index contracts
    poetry run python scripts/demo_rag.py --index --num-contracts 10

    # Search
    poetry run python scripts/demo_rag.py --search "governing law"

    # Answer question
    poetry run python scripts/demo_rag.py --query "Which contracts have California law?"
"""
# Load .env into environment variables
from dotenv import load_dotenv
load_dotenv()

import argparse
from pathlib import Path

from src.rag.indexer import RAGIndexer
from src.rag.retriever import RAGRetriever
from src.utils.rag_config import RAGConfig


def index_contracts(num_contracts: int = 10):
    """Index sample CUAD contracts."""
    print(f"\n{'='*70}")
    print(f"INDEXING {num_contracts} CONTRACTS")
    print(f"{'='*70}\n")

    config = RAGConfig()
    indexer = RAGIndexer(config)

    # Index contracts
    results = indexer.index_cuad_sample(num_contracts=num_contracts)

    print(f"\n{'='*70}")
    print("INDEXING COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Successful: {results['successful']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"📊 Total chunks: {results['total_chunks']}")
    print(f"📈 Avg chunks/contract: {results['total_chunks'] / results['successful']:.1f}")

    # Show stats
    stats = indexer.get_stats()
    print(f"\n📚 Vector Store Stats:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Sample contracts: {', '.join(stats['sample_contracts'][:3])}...")


def search_contracts(query: str, top_k: int = 5):
    """Search for relevant chunks."""
    print(f"\n{'='*70}")
    print(f"SEARCHING: {query}")
    print(f"{'='*70}\n")

    retriever = RAGRetriever()
    results = retriever.search(query, top_k=top_k)

    print(f"Found {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['metadata']['filename']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Text: {result['text'][:200]}...")
        print()


def answer_question(query: str):
    """Answer question using RAG with modern LCEL."""
    print(f"\n{'='*70}")
    print(f"QUESTION: {query}")
    print(f"{'='*70}\n")

    retriever = RAGRetriever()
    result = retriever.answer(query, return_sources=True)

    print(f"ANSWER:")
    print(result['answer'])
    print()

    # Check if sources exist (they should!)
    if 'sources' in result and result['sources']:
        print(f"{'─'*70}")
        print(f"SOURCES ({result['num_sources']}):")
        print(f"{'─'*70}")

        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. {source['metadata']['filename']}")
            print(f"   {source['text'][:150]}...")
    else:
        print("(No sources returned)")


def main():
    parser = argparse.ArgumentParser(description="RAG Demo")
    parser.add_argument("--index", action="store_true", help="Index contracts")
    parser.add_argument("--num-contracts", type=int, default=10, help="Number of contracts to index")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--query", type=str, help="Question to answer")
    parser.add_argument("--top-k", type=int, default=5, help="Number of search results")

    args = parser.parse_args()

    if args.index:
        index_contracts(args.num_contracts)

    elif args.search:
        search_contracts(args.search, args.top_k)

    elif args.query:
        answer_question(args.query)

    else:
        print("Usage:")
        print("  Index:  poetry run python scripts/demo_rag.py --index --num-contracts 10")
        print("  Search: poetry run python scripts/demo_rag.py --search 'governing law'")
        print("  Query:  poetry run python scripts/demo_rag.py --query 'Which contracts have CA law?'")


if __name__ == "__main__":
    main()