"""
Test extraction service with real LLM calls.

Testing Strategy:
- Integration tests: Test full extraction pipeline with real API calls
- Unit tests: Test individual methods with mocked responses
- Cost awareness: Real API calls cost money, so we test strategically

Rationale for test structure:
- Use pytest fixtures for reusable setup
- Mark expensive tests (so they can be skipped in CI)
- Test both success and failure cases
- Verify cost tracking works

Interview Note: This shows you understand:
- Different types of tests (unit vs integration)
- Cost-conscious testing (don't waste API calls)
- Real-world constraints (API can fail, costs money)
"""

import pytest
from pathlib import Path
from openai import OpenAI

from src.extractors.extraction_service import ExtractionService, ExtractionError
from src.utils.extraction_config import ExtractionConfig
from src.utils.config import settings
from src.utils.dataset import CUADDataset
from src.parsers.pdf_parser import PDFParser


# Fixtures - reusable test setup
# Rationale: Don't repeat yourself, setup once, use in multiple tests

@pytest.fixture
def openai_client():
    """
    Create OpenAI client for testing.

    Rationale: Real client for integration tests.
    Uses actual API key from settings.

    Note: This will make real API calls (costs money!)
    That's why we mark tests with @pytest.mark.integration
    """
    return OpenAI(api_key=settings.openai_api_key)


@pytest.fixture
def extraction_config():
    """
    Create test-specific extraction config.

    Rationale: Use cheaper model for tests (gpt-4o-mini).
    Saves money while still testing real behavior.
    """
    return ExtractionConfig(
        model="gpt-4o-mini",  # Cheap model for testing
        temperature=0.0,  # Deterministic results
        max_tokens=4000,
        track_tokens=True,  # Test cost tracking
    )


@pytest.fixture
def extraction_service(openai_client, extraction_config):
    """
    Create extraction service for testing.

    Rationale: Dependency injection in action!
    We inject test-specific config and client.
    """
    return ExtractionService(openai_client, extraction_config)


@pytest.fixture
def cuad_dataset():
    """Get CUAD dataset for testing with real contracts."""
    return CUADDataset()


@pytest.fixture
def pdf_parser():
    """Get PDF parser for extracting text."""
    return PDFParser(extract_tables=False)  # Faster without tables


# Unit Tests - Fast, no API calls

def test_extraction_service_init(openai_client, extraction_config):
    """
    Test service initialization.

    Rationale: Verify dependency injection works.
    This is a unit test (no API calls).
    """
    service = ExtractionService(openai_client, extraction_config)

    assert service.client == openai_client
    assert service.config == extraction_config
    assert service._extraction_count == 0
    assert service._total_tokens_used == 0
    assert service._total_cost == 0.0


def test_get_metrics_initial(extraction_service):
    """
    Test metrics tracking before any extractions.

    Rationale: Verify metrics start at zero.
    """
    metrics = extraction_service.get_metrics()

    assert metrics["extraction_count"] == 0
    assert metrics["total_tokens_used"] == 0
    assert metrics["total_cost"] == 0.0
    assert metrics["avg_cost_per_extraction"] == 0.0
    assert metrics["model"] == "gpt-4o-mini"


def test_extract_empty_text_fails(extraction_service):
    """
    Test that empty text raises error.

    Rationale: Fail-fast validation.
    Better to raise clear error than waste API call.
    """
    with pytest.raises(ExtractionError, match="Contract text is empty"):
        extraction_service.extract("")


def test_extract_very_long_text_fails(extraction_service):
    """
    Test that extremely long text raises error.

    Rationale: Prevent exceeding token limits.
    """
    # Create very long text (>100k tokens)
    very_long_text = "word " * 50000  # ~100k tokens

    with pytest.raises(ExtractionError, match="exceeds token limit"):
        extraction_service.extract(
            very_long_text,
            max_tokens_allowed=10000  # Set low limit for test
        )


# Integration Tests - Expensive, real API calls
# Mark with @pytest.mark.integration so they can be skipped

@pytest.mark.integration
def test_extract_simple_text(extraction_service):
    """
    Test extraction with simple contract text.

    Rationale: Verify basic extraction works end-to-end.
    This makes a REAL API call!

    What this tests:
    - Prompt building works
    - API call succeeds
    - Response parsing works
    - Pydantic validation works
    - Cost tracking works

    This is an integration test because it tests the full pipeline.
    """
    # Simple contract text for testing
    contract_text = """
    MASTER SERVICES AGREEMENT

    This Agreement is entered into as of January 15, 2024 ("Effective Date")
    by and between TechCorp Solutions Inc., a Delaware corporation ("Provider"),
    and Enterprise Client LLC, a California corporation ("Client").

    1. TERM AND TERMINATION
    This Agreement shall commence on the Effective Date and continue for a period
    of one (1) year, unless terminated earlier in accordance with this Section.

    2. GOVERNING LAW
    This Agreement shall be governed by and construed in accordance with the laws
    of the State of California, without regard to its conflict of law provisions.

    3. PAYMENT TERMS
    Client shall pay Provider a monthly fee of $10,000, payable within thirty (30)
    days of receipt of invoice.
    """

    # Run extraction
    result = extraction_service.extract(
        contract_text,
        filename="test_simple_contract.txt"
    )

    # Verify result structure
    assert result is not None
    assert result.filename == "test_simple_contract.txt"

    # Verify some clauses were extracted
    # Note: We don't assert exact values because LLM output can vary
    # Instead, we check that reasonable extractions happened
    assert result.document_name is not None
    assert "MASTER SERVICES AGREEMENT" in result.document_name.text.upper()

    assert result.parties is not None
    assert len(result.parties.answer) >= 2  # Should find both parties

    assert result.effective_date is not None
    assert "2024" in result.effective_date.text  # Should find date

    assert result.governing_law is not None
    assert "California" in result.governing_law.answer

    assert result.payment_terms is not None
    assert "10,000" in result.payment_terms.text or "10000" in result.payment_terms.text

    # Verify metrics were tracked
    metrics = extraction_service.get_metrics()
    assert metrics["extraction_count"] == 1
    assert metrics["total_tokens_used"] > 0
    assert metrics["total_cost"] > 0

    print(f"\n✅ Extraction successful!")
    print(f"📊 Metrics: {metrics}")
    print(f"📄 Found {len(result.get_clause_names())} clauses")


@pytest.mark.integration
def test_extract_real_cuad_contract(
        extraction_service,
        cuad_dataset,
        pdf_parser
):
    """
    Test extraction with real CUAD contract.

    Rationale: Most important test!
    - Uses real contract from dataset
    - Tests full PDF → Extraction pipeline
    - Validates against actual legal document

    This is the "money test" - if this works, your system works.

    Cost: ~$0.01 per run (acceptable for testing)
    """
    # Get a sample contract
    sample_contracts = cuad_dataset.get_sample_contracts(n=1)
    contract_filename = sample_contracts[0]

    print(f"\n📄 Testing with: {contract_filename}")

    # Parse PDF
    pdf_path = cuad_dataset.pdf_dir / contract_filename
    parsed_pdf = pdf_parser.parse(pdf_path)

    print(f"📊 Contract has {len(parsed_pdf.full_text)} characters")

    # Extract clauses
    result = extraction_service.extract(
        parsed_pdf.full_text,
        filename=contract_filename
    )

    # Verify extraction
    assert result is not None
    assert result.filename == contract_filename

    # Verify we extracted some clauses
    clause_names = result.get_clause_names()
    print(f"✅ Extracted {len(clause_names)} clauses: {clause_names}")

    # Most contracts should have at least these basic clauses
    # (but we don't assert because not all contracts have all clauses)
    if result.document_name:
        print(f"📝 Document Name: {result.document_name.answer}")

    if result.parties:
        print(f"👥 Parties: {result.parties.answer}")

    if result.governing_law:
        print(f"⚖️  Governing Law: {result.governing_law.answer}")

    # Verify metrics
    metrics = extraction_service.get_metrics()
    print(f"\n💰 Cost: ${metrics['total_cost']:.4f}")
    print(f"🔢 Tokens used: {metrics['total_tokens_used']:,}")

    assert len(clause_names) > 0, "Should extract at least one clause"


@pytest.mark.integration
@pytest.mark.slow
def test_extract_multiple_contracts(
        extraction_service,
        cuad_dataset,
        pdf_parser
):
    """
    Test extraction with multiple contracts.

    Rationale: Verify service handles multiple extractions.
    Tests:
    - Metrics accumulation
    - Service state management
    - Consistent results

    Marked as @pytest.mark.slow because it's expensive (3 API calls).
    Run this occasionally, not on every test run.

    Cost: ~$0.03 per run
    """
    # Get 3 sample contracts
    sample_contracts = cuad_dataset.get_sample_contracts(n=3)

    results = []
    for contract_filename in sample_contracts:
        print(f"\n📄 Processing: {contract_filename}")

        # Parse and extract
        pdf_path = cuad_dataset.pdf_dir / contract_filename
        parsed_pdf = pdf_parser.parse(pdf_path)

        result = extraction_service.extract(
            parsed_pdf.full_text,
            filename=contract_filename
        )

        results.append(result)
        print(f"✅ Extracted {len(result.get_clause_names())} clauses")

    # Verify all extractions succeeded
    assert len(results) == 3

    # Verify metrics accumulated
    metrics = extraction_service.get_metrics()
    assert metrics["extraction_count"] == 3
    assert metrics["total_cost"] > 0

    print(f"\n ===  Final Metrics: ====")
    print(f"  Total extractions: {metrics['extraction_count']}")
    print(f"  Total tokens: {metrics['total_tokens_used']:,}")
    print(f"  Total cost: ${metrics['total_cost']:.4f}")
    print(f"  Avg cost per contract: ${metrics['avg_cost_per_extraction']:.4f}")


# Helper function to run specific tests manually
if __name__ == "__main__":
    """
    Run tests manually for debugging.

    Usage:
        poetry run python tests/test_extraction_service.py

    Rationale: Sometimes you want to run one test manually
    without pytest's overhead.
    """
    pytest.main([
        __file__,
        "-v",
        "-s",  # Show print statements
        "-m", "integration",  # Only run integration tests
    ])