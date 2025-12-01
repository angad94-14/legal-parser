"""
Legal Contract Parser - Streamlit UI

A simple, clean interface for:
1. Chat: Ask questions about indexed contracts (RAG)
2. Extract: Upload PDFs to extract metadata

Author: [Your Name]
"""

import streamlit as st
import requests
from typing import Dict, Any, List
import os
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

# API endpoint (configure based on environment)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Legal Contract Parser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# Styling
# ============================================================================

# Custom CSS for better appearance
st.markdown("""
<style>
    /* Main title */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }

    /* Subtitle */
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }

    /* Chat messages */
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    .assistant-message {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    /* Confidence badges */
    .confidence-high {
        color: #2e7d32;
        font-weight: bold;
    }

    .confidence-medium {
        color: #f57c00;
        font-weight: bold;
    }

    .confidence-low {
        color: #c62828;
        font-weight: bold;
    }

    /* Stats box */
    .stats-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper Functions
# ============================================================================
def display_sources(sources: List[Dict], max_chars: int = 500):
    """Helper function to display sources consistently."""
    for i, source in enumerate(sources, 1):
        # Get score
        score = source.get('score', None)

        # Header with filename and score
        filename = source['metadata'].get('filename', 'Unknown')
        if score is not None:
            st.markdown(f"**{i}. {filename}** (Similarity: {score:.3f})")
        else:
            st.markdown(f"**{i}. {filename}**")

        # Clean and display text
        text = source.get("text", "")
        # Remove extra whitespace and newlines
        clean_text = " ".join(text.split())
        preview = clean_text[:max_chars] + "..." if len(clean_text) > max_chars else clean_text

        # Use st.info() - automatically adapts to theme!
        st.info(preview)

        # OR use blockquote markdown (also adapts):
        # st.markdown(f"> {preview}")

        # Add separator between sources (except last one)
        if i < len(sources):
            st.markdown("---")

def get_rag_stats() -> Dict[str, Any]:
    """Get RAG system statistics from API."""
    try:
        response = requests.get(f"{API_URL}/rag/stats")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to get stats: {str(e)}")
        return {}


def search_contracts(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search contracts (pure vector search)."""
    try:
        response = requests.post(
            f"{API_URL}/rag/search",
            json={"query": query, "top_k": top_k}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return {}


def query_contracts(question: str, return_sources: bool = True) -> Dict[str, Any]:
    """Ask question using RAG (with LLM generation)."""
    try:
        response = requests.post(
            f"{API_URL}/rag/query",
            json={
                "question": question,
                "return_sources": return_sources
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Query failed: {str(e)}")
        return {}


def extract_contract(file) -> Dict[str, Any]:
    """Extract metadata from uploaded contract."""
    try:
        files = {"file": (file.name, file, "application/pdf")}
        response = requests.post(f"{API_URL}/extract", files=files)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Extraction failed: {str(e)}")
        return {}


def clear_memory():
    """Clear conversation memory."""
    try:
        response = requests.post(f"{API_URL}/rag/memory/clear")
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Failed to clear memory: {str(e)}")
        return False


def get_memory_stats() -> Dict[str, Any]:
    """Get conversation memory statistics."""
    try:
        response = requests.get(f"{API_URL}/rag/memory/stats")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {}


def get_confidence_badge(confidence: str) -> str:
    """Return HTML for confidence badge."""
    if confidence == "high":
        return '<span class="confidence-high">🟢 High Confidence</span>'
    elif confidence == "medium":
        return '<span class="confidence-medium">🟡 Medium Confidence</span>'
    else:
        return '<span class="confidence-low">🔴 Low Confidence</span>'


# ============================================================================
# Main App
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-title">📄 Legal Contract Parser</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered contract analysis and Q&A system</p>', unsafe_allow_html=True)

    # Tabs
    tab1, tab2 = st.tabs(["💬 Chat", "📄 Extract"])

    # ========================================================================
    # Tab 1: Chat Interface
    # ========================================================================

    with tab1:
        st.markdown("### Ask Questions About Contracts")

        # Get RAG stats
        stats = get_rag_stats()
        mem_stats = get_memory_stats()

        # Display stats in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Indexed Contracts",
                stats.get("total_documents", "N/A")
            )

        with col2:
            st.metric(
                "Total Chunks",
                stats.get("total_chunks", "N/A")
            )

        with col3:
            st.metric(
                "Conversation Turns",
                mem_stats.get("num_exchanges", 0)
            )

        st.markdown("---")

        # Example questions
        st.markdown("**💡 Example Questions:**")
        example_col1, example_col2, example_col3 = st.columns(3)

        with example_col1:
            if st.button("Which contracts have California law?"):
                st.session_state.example_question = "Which contracts have California law?"

        with example_col2:
            if st.button("What are the payment terms?"):
                st.session_state.example_question = "What are the payment terms?"

        with example_col3:
            if st.button("Show me termination clauses"):
                st.session_state.example_question = "Show me termination clauses"

        st.markdown("---")

        # Initialize chat history in session state
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # Display chat history
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if "sources" in message and message["sources"]:
                    with st.expander(f"📎 Sources ({len(message['sources'])})"):
                        display_sources(message["sources"], max_chars=500)

        # Chat input
        user_input = st.chat_input("Ask a question about your contracts...")

        # Handle example question click
        if "example_question" in st.session_state:
            user_input = st.session_state.example_question
            del st.session_state.example_question

        # Process user input
        if user_input:
            # Add user message to chat
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_input
            })

            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)

            # Get response from API
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = query_contracts(user_input, return_sources=True)

                if result:
                    answer = result.get("answer", "Sorry, I couldn't generate an answer.")
                    sources = result.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander(f"📎 Sources ({len(sources)})"):
                            display_sources(sources, max_chars=500)

                    # Add assistant message to chat
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

        # Clear conversation button
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear Conversation"):
                if clear_memory():
                    st.session_state.chat_messages = []
                    st.success("Conversation cleared!")
                    st.rerun()

    # ========================================================================
    # Tab 2: Extract Interface
    # ========================================================================

    with tab2:
        st.markdown("### Extract Contract Metadata")

        st.markdown("""
        Upload a contract PDF to extract structured information like:
        - Document name and parties
        - Dates (agreement, effective, expiration)
        - Governing law and jurisdiction
        - Payment terms
        - Key clauses (termination, renewal, liability, etc.)
        """)

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Contract (PDF)",
            type=["pdf"],
            help="Upload a contract PDF to extract metadata"
        )

        # Extract button
        if uploaded_file:
            st.info(f"📄 File: {uploaded_file.name} ({uploaded_file.size:,} bytes)")

            if st.button("🔍 Extract Metadata", type="primary"):
                with st.spinner("Extracting metadata... This may take 5-10 seconds..."):
                    result = extract_contract(uploaded_file)

                if result:
                    st.success("✅ Extraction Complete!")

                    # Display metadata
                    st.markdown("---")
                    st.markdown(f"**Extraction ID:** `{result.get('extraction_id')}`")
                    st.markdown(f"**Status:** {result.get('status')}")
                    st.markdown(f"**Timestamp:** {result.get('timestamp')}")

                    # Display cost/time if available
                    metadata = result.get("metadata", {})
                    if metadata:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model", metadata.get("model", "N/A"))
                        with col2:
                            st.metric("Tokens", f"{metadata.get('token_count', 0):,}")
                        with col3:
                            cost = metadata.get("total_cost", 0)
                            st.metric("Cost", f"${cost:.4f}")

                    st.markdown("---")

                    # Display extracted clauses
                    st.markdown("### 📋 Extracted Clauses")

                    clauses = result.get("clauses", {})

                    if clauses:
                        for clause_name, clause_data in clauses.items():
                            if clause_data:  # Only show non-null clauses
                                # Format clause name (remove underscores, title case)
                                display_name = clause_name.replace("_", " ").title()

                                with st.expander(f"**{display_name}**", expanded=True):
                                    # Answer
                                    answer = clause_data.get("answer")
                                    if answer:
                                        if isinstance(answer, list):
                                            st.markdown("**Value:**")
                                            for item in answer:
                                                st.markdown(f"- {item}")
                                        else:
                                            st.markdown(f"**Value:** {answer}")

                                    # Confidence and page
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        confidence = clause_data.get("confidence", "unknown")
                                        st.markdown(get_confidence_badge(confidence), unsafe_allow_html=True)
                                    with col2:
                                        page = clause_data.get("page_number")
                                        if page:
                                            st.markdown(f"📄 Page {page}")

                                    # Original text
                                    text = clause_data.get("text")
                                    if text:
                                        st.markdown("**Original Text:**")
                                        st.caption(text)
                    else:
                        st.warning("No clauses were extracted from this contract.")

                    # Download JSON button
                    st.markdown("---")
                    st.download_button(
                        label="📥 Download JSON",
                        data=str(result),
                        file_name=f"{uploaded_file.name}_extracted.json",
                        mime="application/json"
                    )


# ============================================================================
# Sidebar (Optional Info)
# ============================================================================

with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Legal Contract Parser** uses AI to:
    - 💬 Answer questions about contracts
    - 📄 Extract structured metadata
    - 🔍 Search contract corpus semantically

    **Tech Stack:**
    - FastAPI backend
    - OpenAI GPT-4o-mini
    - LangChain + ChromaDB
    - Streamlit UI

    **Note:** This is a demo system with 30+ pre-indexed CUAD contracts.
    """)

    st.markdown("---")

    # API status
    st.markdown("### 🔌 API Status")
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")

    st.caption(f"API: `{API_URL}`")

# ============================================================================
# Run App
# ============================================================================

if __name__ == "__main__":
    main()