"""
RAG Retriever - Search and answer questions using modern LCEL.

UPDATED: Now includes conversation memory for multi-turn conversations.
"""

from typing import List, Dict, Any, Optional
import logging

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationBufferWindowMemory

from src.utils.config import settings
from src.utils.rag_config import RAGConfig, DEFAULT_RAG_CONFIG

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Retrieves information from indexed contracts using modern LCEL.

    UPDATED: Now includes conversation memory for follow-up questions.

    Design Pattern: LCEL (LangChain Expression Language) with memory
    - Chains are composable with | operator
    - Memory tracks last K conversation turns
    - Type-safe and easy to debug

    Memory Strategy:
    - Uses ConversationBufferWindowMemory (k=5)
    - Keeps last 5 exchanges (10 messages)
    - Balances context vs token cost
    - Prevents memory overflow

    Note: Single-user mode - all requests share same memory.
    For multi-user: would need session-based memory instances.

    Interview Note: Using windowed memory shows you understand
    token limits and cost optimization.
    """

    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG):
        """
        Initialize retriever with modern LCEL approach and conversation memory.

        Args:
            config: RAG configuration
        """
        self.config = config

        # Initialize embeddings (same model as indexing!)
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=settings.openai_api_key
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
            openai_api_key=settings.openai_api_key
        )

        # Create retriever from vector store
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": config.retrieval_top_k}
        )

        # NEW: Add conversation memory
        # Rationale: Keep last 5 exchanges (10 messages)
        # - Enough for context continuity
        # - Won't hit token limits
        # - Keeps costs reasonable
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Keep last 5 exchanges
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"
        )

        # Create RAG chain using modern LCEL with memory support
        self.rag_chain = self._create_rag_chain()

        logger.info("RAGRetriever initialized with LCEL and conversation memory (k=5)")

    def _create_rag_chain(self):
        """
        Create RAG chain using LCEL with conversation memory support.

        Returns:
            Runnable chain for RAG with memory

        Rationale: Uses RunnablePassthrough.assign to handle dict inputs.

        Input format:
        {
            "question": "What is X?",
            "chat_history": [HumanMessage(...), AIMessage(...), ...]
        }

        Chain flow:
        1. Receive dict with question + chat_history
        2. Extract question → retrieve context → add to dict
        3. Pass question + chat_history + context to prompt
        4. LLM generates answer with full context
        5. Parse and return answer string
        """

        # Define prompt template with chat history
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a legal contract analysis assistant. 
    Use the following contract excerpts to answer questions accurately.

    **Instructions:**
    - Base your answer ONLY on the provided contract excerpts
    - Consider the conversation history for context on follow-up questions
    - Cite specific contracts and sections you reference
    - If the answer is not in the excerpts, say "I don't have enough information to answer this based on the provided contracts."
    - Be precise and concise"""),

            # Chat history placeholder
            MessagesPlaceholder(variable_name="chat_history"),

            ("human", """Contract Excerpts:
    {context}

    Question: {question}

    Answer:""")
        ])

        # Helper function to format retrieved documents
        def format_docs(docs):
            """Format retrieved documents into context string."""
            return "\n\n".join(doc.page_content for doc in docs)

        # Build LCEL chain
        # RunnablePassthrough.assign adds fields to input dict
        rag_chain = (
                RunnablePassthrough.assign(
                    # Add context field by retrieving based on question
                    context=lambda x: format_docs(self.retriever.invoke(x["question"]))
                )
                # Now input dict has: question, chat_history, context
                | prompt
                | self.llm
                | StrOutputParser()
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

        Note: This does NOT use conversation memory.
        Pure search operations don't need conversational context.

        Args:
            query: Search query
            top_k: Number of results (overrides config)
            filter_metadata: Filter by metadata

        Returns:
            List of matching chunks with metadata and scores
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
        Answer question using RAG with conversation memory.

        UPDATED: Now includes conversation history in LLM context.

        Args:
            query: User question
            return_sources: Whether to return source documents

        Returns:
            Dictionary with answer and optional sources

        Process:
        1. Load conversation history from memory
        2. Chain retrieves relevant chunks
        3. Formats prompt with history + chunks + question
        4. LLM generates answer (aware of conversation context)
        5. Save interaction to memory
        6. Return answer + sources

        Example:
            # First question
            result = retriever.answer("Which contracts have CA law?")
            # Answer: "TechCorp, Euromedia, Cybergy"

            # Follow-up (uses memory!)
            result = retriever.answer("What are their payment terms?")
            # Answer: "TechCorp: Net 30, Euromedia: Net 60, Cybergy: Net 45"
            # ↑ LLM knows "their" refers to the 3 contracts from previous Q!
        """
        logger.info(f"Answering query: '{query}'")

        # Load conversation history from memory
        memory_vars = self.memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])

        logger.info(f"Using {len(chat_history)} messages from conversation history")

        # Generate answer using LCEL chain with history
        # The chain expects: question (str) and chat_history (list of messages)
        answer = self.rag_chain.invoke({
            "question": query,
            "chat_history": chat_history
        })

        # Save this interaction to memory
        # Rationale: Memory will be used for next question
        self.memory.save_context(
            {"question": query},
            {"answer": answer}
        )

        response = {
            "query": query,
            "answer": answer,
        }

        # Retrieve sources if requested
        if return_sources:
            # Use retriever directly to get source docs
            # source_docs = self.retriever.invoke(query)
            source_docs = self.vector_store.similarity_search_with_score(
                query,
                k=self.config.retrieval_top_k
            )
            sources = []
            for doc, score in source_docs:
                sources.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })

            response["sources"] = sources
            response["num_sources"] = len(sources)

        logger.info(f"Generated answer ({len(answer)} chars) with memory context")
        return response

    def answer_with_context(
        self,
        query: str,
        context_chunks: List[str]
    ) -> str:
        """
        Generate answer given explicit context chunks (bypass retrieval).

        Note: This does NOT use conversation memory.
        Used for custom retrieval scenarios where memory isn't needed.

        Args:
            query: User question
            context_chunks: Pre-selected context chunks

        Returns:
            Generated answer
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

        UPDATED: Now includes conversation memory.

        Args:
            query: User question

        Yields:
            Chunks of the answer as they're generated

        Note: Memory is saved after streaming completes.
        """
        logger.info(f"Streaming answer for: '{query}'")

        # Load conversation history
        memory_vars = self.memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])

        # Stream the response
        full_answer = ""
        for chunk in self.rag_chain.stream({
            "question": query,
            "chat_history": chat_history
        }):
            full_answer += chunk
            yield chunk

        # Save to memory after streaming completes
        self.memory.save_context(
            {"question": query},
            {"answer": full_answer}
        )

    def clear_memory(self):
        """
        Clear conversation memory.

        Use cases:
        - Start fresh conversation
        - Reset after testing
        - Clear after inactivity

        Example:
            retriever.clear_memory()
        """
        self.memory.clear()
        logger.info("Conversation memory cleared")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current memory state.

        Returns:
            Dictionary with memory info

        Example:
            stats = retriever.get_memory_stats()
            # {
            #   "num_messages": 10,
            #   "num_exchanges": 5,
            #   "window_size": 5
            # }
        """
        memory_vars = self.memory.load_memory_variables({})
        messages = memory_vars.get("chat_history", [])

        return {
            "num_messages": len(messages),
            "num_exchanges": len(messages) // 2,
            "window_size": self.memory.k
        }


# Convenience functions (unchanged)

def search_contracts(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Quick search function."""
    retriever = RAGRetriever()
    return retriever.search(query, top_k=top_k)


def answer_question(query: str) -> str:
    """Quick answer function."""
    retriever = RAGRetriever()
    result = retriever.answer(query, return_sources=False)
    return result["answer"]