"""
Test contract data models.

Rationale: Even data models need tests because:
1. Pydantic validation might have bugs in our validators
2. Schema changes can break downstream code
3. Tests serve as executable documentation
4. Easy to catch breaking changes early
"""

import pytest
from src.models.contract import (
    ExtractedContract,
    DocumentName,
    Parties,
    GoverningLaw,
    ConfidenceLevel,
)


def test_extracted_clause_basic():
    """
    Test basic clause extraction model.

    Learning: Start with simplest test - can we create the model?
    """
    clause = DocumentName(
        text="MASTER SERVICES AGREEMENT",
        answer="Master Services Agreement",
        confidence=ConfidenceLevel.HIGH,
        page_number=1
    )

    assert clause.text == "MASTER SERVICES AGREEMENT"
    assert clause.answer == "Master Services Agreement"
    assert clause.confidence == ConfidenceLevel.HIGH
    assert clause.page_number == 1


def test_parties_with_list():
    """
    Test that Parties accepts list of strings.

    Learning: Test specialized behavior (Parties has List[str] answer).
    """
    parties = Parties(
        text="between TechCorp Inc. and Client LLC",
        answer=["TechCorp Inc.", "Client LLC"],
        confidence=ConfidenceLevel.HIGH
    )

    assert len(parties.answer) == 2
    assert "TechCorp Inc." in parties.answer


def test_extracted_contract_minimal():
    """
    Test creating contract with minimal data.

    Learning: All fields are Optional - contract should work with just filename.
    """
    contract = ExtractedContract(
        filename="test.pdf"
    )

    assert contract.filename == "test.pdf"
    assert contract.document_name is None  # Optional fields are None


def test_extracted_contract_with_clauses():
    """
    Test creating contract with multiple clauses.

    Learning: Verify we can populate multiple clause types.
    """
    contract = ExtractedContract(
        filename="test.pdf",
        document_name=DocumentName(
            text="MASTER SERVICES AGREEMENT",
            answer="MSA",
            confidence=ConfidenceLevel.HIGH
        ),
        governing_law=GoverningLaw(
            text="governed by California law",
            answer="California",
            confidence=ConfidenceLevel.HIGH
        )
    )

    assert contract.document_name.answer == "MSA"
    assert contract.governing_law.answer == "California"


def test_to_dict_excludes_none():
    """
    Test that to_dict() excludes None values for cleaner JSON.

    Learning: API responses should be clean - no null spam.
    """
    contract = ExtractedContract(
        filename="test.pdf",
        document_name=DocumentName(
            text="Test Agreement",
            answer="Test",
            confidence=ConfidenceLevel.HIGH
        )
        # All other fields are None
    )

    result = contract.to_dict()

    assert "filename" in result
    assert "document_name" in result
    # None fields should be excluded
    assert "parties" not in result
    assert "governing_law" not in result


def test_get_clause_names():
    """
    Test getting list of extracted clause names.

    Learning: Useful for debugging and metrics.
    """
    contract = ExtractedContract(
        filename="test.pdf",
        document_name=DocumentName(text="Test", answer="Test"),
        governing_law=GoverningLaw(text="CA law", answer="California")
    )

    clause_names = contract.get_clause_names()

    assert "document_name" in clause_names
    assert "governing_law" in clause_names
    assert "parties" not in clause_names  # Not set


def test_confidence_enum():
    """
    Test that confidence levels serialize correctly.

    Learning: Enums need special handling for JSON serialization.
    """
    clause = DocumentName(
        text="Agreement",
        answer="Agreement",
        confidence=ConfidenceLevel.LOW
    )

    # Should serialize as string, not Enum object
    assert clause.confidence == ConfidenceLevel.LOW
    assert clause.confidence.value == "low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])