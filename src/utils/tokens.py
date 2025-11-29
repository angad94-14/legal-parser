"""
Token counting utilities for OpenAI models.

Rationale: Token limits and costs are based on tokens, not characters.
We need accurate token counting to:
- Prevent exceeding context windows
- Estimate costs before making API calls
- Optimize chunk sizes for RAG

Why tiktoken?
- Official OpenAI tokenizer
- Accurate (same tokenization as API)
- Fast (written in Rust)
"""

import tiktoken
from typing import Optional


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens in text for a specific model.

    Args:
        text: Text to tokenize
        model: Model name (determines tokenizer)

    Returns:
        Number of tokens

    Rationale: Different models use different tokenizers:
    - GPT-4, GPT-4o, GPT-3.5-turbo: cl100k_base
    - GPT-3 (davinci, ada, etc.): p50k_base

    Always use the correct tokenizer for accurate counts!

    Example:
        text = "Hello, world!"
        count = count_tokens(text)  # 4 tokens

    Interview Note: Shows you understand tokenization.
    Common mistake: Using len(text.split()) (word count ≠ token count)
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Model not found, use default
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def truncate_to_tokens(
        text: str,
        max_tokens: int,
        model: str = "gpt-4o"
) -> str:
    """
    Truncate text to fit within token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model: Model name

    Returns:
        Truncated text (or original if under limit)

    Rationale: When text exceeds limits, we need to truncate smartly.
    This preserves as much content as possible within token budget.

    Note: This truncates from the end. For contracts, you might want
    to truncate from the middle or use smarter strategies.

    Example:
        long_text = "..." * 100000  # Very long
        truncated = truncate_to_tokens(long_text, max_tokens=5000)
        # Now fits in 5000 tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    # Truncate tokens and decode back to text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def estimate_tokens(text: str) -> int:
    """
    Fast token estimation without loading encoder.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count

    Rationale: Sometimes you need a quick estimate without
    loading the full tokenizer (which takes ~100ms first time).

    Rule of thumb: ~4 characters = 1 token for English text.
    This is approximate but useful for quick checks.

    Example:
        text = "Short text"
        estimate = estimate_tokens(text)  # Fast, approximate
        exact = count_tokens(text)        # Slow, accurate
    """
    # Rough estimate: 1 token ≈ 4 characters for English
    return len(text) // 4