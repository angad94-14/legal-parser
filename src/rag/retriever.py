"""
RAG Retriever - Search and answer questions using modern LCEL.

Modern Approach: Uses LangChain Expression Language (LCEL)
- ChatPromptTemplate for chat-style prompts
- Runnable chains for composition
- Better type safety and flexibility

Rationale: LCEL is the modern LangChain pattern (2024+)
- More maintainable
- Better error messages
- Easier to customize
- Industry standard
"""

from typing import List, Dict, Any, Optional
import logging

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from src.utils.rag_config import RAGConfig, DEFAULT_RAG_CONFIG
from src.utils.config import settings

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Retrieves information from indexed contracts using modern LCEL.

    Design Pattern: LCEL (LangChain Expression Language)
    - Chains are composable with | operator
    - Type-safe
    - Easy to debug

    Two main operations:
    1. Search: Find relevant chunks (pure retrieval)
    2. Answer: Generate answer from chunks (RAG)

    Interview Note: Using modern LCEL shows you stay current with
    best practices and aren't just copying old tutorials.
    """

    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG):
        """
        Initialize retriever with modern LCEL approach.

        Args:
            config: RAG configuration
        """
        self.config = config

        # Initialize embeddings (same model as indexing!)
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=settings.openai_api_key
        )

        # Load vector store
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=config.vector_store_path,
        )

        # Initialize LLM for answer generation
        self.llm = ChatOpenAI(
            model=config.generation_model,
            temperature=config.generation_temperature,
            max_tokens=config.generation_max_tokens,
        )

        # Create retriever from vector store
        # Rationale: Retriever is a standard interface in LangChain
        # - Has .invoke() method
        # - Can be used in LCEL chains with | operator
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": config.retrieval_top_k}
        )

        # Create RAG chain using modern LCEL
        self.rag_chain = self._create_rag_chain()

        logger.info("RAGRetriever initialized with modern LCEL")

    def _create_rag_chain(self):
        """
        Create RAG chain using LCEL (LangChain Expression Language).

        Returns:
            Runnable chain for RAG

        Rationale: LCEL uses | operator to compose chains.

        Chain flow:
        1. Input: {"question": "What is X?"}
        2. RunnableParallel retrieves context and passes question
        3. Prompt formats context + question
        4. LLM generates answer
        5. StrOutputParser extracts text from LLM response

        """

        # Define prompt template with chat format
        # Rationale: ChatPromptTemplate supports system/user roles
        # Better for modern chat models (GPT-4, Claude, etc.)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a legal contract analysis assistant. 
Use the following contract excerpts to answer questions accurately.

**Instructions:**
- Base your answer ONLY on the provided contract excerpts
- Cite specific contracts and sections you reference
- If the answer is not in the excerpts, say "I don't have enough information to answer this based on the provided contracts."
- Be precise and concise"""),
            ("human", """Contract Excerpts:
{context}

Question: {question}

Answer:""")
        ])

        # Helper function to format retrieved documents
        # Rationale: Retrieved docs need to be converted to string
        def format_docs(docs):
            """Format retrieved documents into context string."""
            return "\n\n".join(doc.page_content for doc in docs)

        # Build LCEL chain
        # Rationale: | operator composes runnables
        # RunnableParallel runs retriever and question in parallel
        # Then pipes to prompt → LLM → parser
        rag_chain = (
            RunnableParallel(
                context=self.retriever | format_docs,  # Retrieve & format
                question=RunnablePassthrough()          # Pass question through
            )
            | prompt           # Format prompt with context + question
            | self.llm         # Generate answer
            | StrOutputParser() # Extract string from LLM response
        )

        return rag_chain

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks (retrieval only, no generation).

        Args:
            query: Search query
            top_k: Number of results (overrides config)
            filter_metadata: Filter by metadata

        Returns:
            List of matching chunks with metadata and scores

        Rationale: Pure vector search without LLM generation.
        Fast and cheap for exploratory queries.
        """
        k = top_k or self.config.retrieval_top_k

        logger.info(f"Searching for: '{query}' (top_k={k})")

        # Perform similarity search with scores
        if filter_metadata and self.config.enable_metadata_filtering:
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                filter=filter_metadata
            )
        else:
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k
            )

        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            })

        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results

    def answer(
        self,
        query: str,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer question using RAG (modern LCEL approach).

        Args:
            query: User question
            return_sources: Whether to return source documents

        Returns:
            Dictionary with answer and optional sources

        Rationale: Full RAG with modern LCEL chain.

        Process:
        1. Chain retrieves relevant chunks
        2. Formats prompt with chunks + question
        3. LLM generates answer
        4. Returns answer + sources

        Example:
            result = retriever.answer("Which contracts have CA law?")
            print(result['answer'])
            for source in result['sources']:
                print(f"  - {source['metadata']['filename']}")
        """
        logger.info(f"Answering query: '{query}'")

        # Generate answer using LCEL chain
        # Rationale: .invoke() runs the full chain
        # Input: question string
        # Output: answer string
        answer = self.rag_chain.invoke(query)

        response = {
            "query": query,
            "answer": answer,
        }

        # Retrieve sources if requested
        # Rationale: LCEL chain doesn't automatically return sources
        # We need to retrieve them separately
        if return_sources:
            # Use retriever directly to get source docs
            source_docs = self.retriever.invoke(query)

            sources = []
            for doc in source_docs:
                sources.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                })

            response["sources"] = sources
            response["num_sources"] = len(sources)

        logger.info(f"Generated answer ({len(answer)} chars)")
        return response

    def answer_with_context(
        self,
        query: str,
        context_chunks: List[str]
    ) -> str:
        """
        Generate answer given explicit context chunks (bypass retrieval).

        Args:
            query: User question
            context_chunks: Pre-selected context chunks

        Returns:
            Generated answer

        Rationale: Manual control over retrieval.
        Useful for:
        - Custom retrieval logic
        - Hybrid search
        - Debugging
        """
        context = "\n\n".join(context_chunks)

        # Create ad-hoc prompt for direct LLM call
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal contract assistant. Answer based on the provided excerpts."),
            ("human", f"""Contract Excerpts:
{context}

Question: {query}

Answer:""")
        ])

        # Create simple chain: prompt → LLM → parser
        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({})

    def stream_answer(self, query: str):
        """
        Stream answer in real-time (for UI responsiveness).

        Args:
            query: User question

        Yields:
            Chunks of the answer as they're generated

        Rationale: Streaming improves perceived latency.
        User sees partial answer immediately instead of waiting
        for complete response.

        Example:
            for chunk in retriever.stream_answer("What is X?"):
                print(chunk, end="", flush=True)

        Interview Note: Shows you understand UX considerations.
        Streaming is critical for good chat experiences.
        """
        logger.info(f"Streaming answer for: '{query}'")

        # LCEL chains support streaming with .stream()
        for chunk in self.rag_chain.stream(query):
            yield chunk


# Convenience functions

def search_contracts(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Quick search function.

    Example:
        results = search_contracts("payment terms")
        for r in results:
            print(r['metadata']['filename'])
    """
    retriever = RAGRetriever()
    return retriever.search(query, top_k=top_k)


def answer_question(query: str) -> str:
    """
    Quick answer function.

    Example:
        answer = answer_question("What is the governing law?")
        print(answer)
    """
    retriever = RAGRetriever()
    result = retriever.answer(query, return_sources=False)
    return result["answer"]