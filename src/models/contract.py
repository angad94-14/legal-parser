"""
Data models for contract extraction.

Design principle: Define the OUTPUT schema first, then build extraction logic.
This ensures:
1. Type safety (Pydantic validation)
2. Clear API contracts
3. Easy testing (mock data must match schema)
4. Self-documenting code
"""

from typing import Optional, List, Dict, Any
from datetime import date
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from pydantic.config import ConfigDict


class ConfidenceLevel(str, Enum):
    """
    Confidence in extraction accuracy.

    Rationale: Not all extractions are equally certain. Tracking confidence helps:
    - Identify which clauses need human review
    - Improve prompts for low-confidence extractions
    - Monitor model performance over time
    """
    HIGH = "high"  # 90%+ confident
    MEDIUM = "medium"  # 70-90% confident
    LOW = "low"  # <70% confident


class ExtractedClause(BaseModel):
    """
    Base model for any extracted clause.

    Design pattern: All clauses share common fields (text, confidence, page).
    This makes it easy to:
    - Track where information came from (provenance)
    - Assess extraction quality
    - Debug extraction errors
    """
    text: Optional[str] = Field(
        None,
        description="Raw text of the clause as it appears in contract"
    )
    answer: Optional[str] = Field(
        None,
        description="Normalized/extracted answer (e.g., 'California' from 'governed by California law')"
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Confidence level in this extraction"
    )
    page_number: Optional[int] = Field(
        None,
        description="Page where this clause was found"
    )


class DocumentName(ExtractedClause):
    """
    Document name/title extraction.

    Example: "MASTER SERVICES AGREEMENT" from header
    """
    pass


class Parties(ExtractedClause):
    """
    Contract parties extraction.

    Rationale: Parties are critical - we extract both raw text AND parsed list.
    - text: "between TechCorp Inc. and Client LLC"
    - answer: ["TechCorp Inc.", "Client LLC"]

    This dual approach:
    - Preserves original wording (for legal accuracy)
    - Provides structured data (for programmatic use)
    """
    answer: Optional[List[str]] = Field(
        None,
        description="List of party names extracted from the clause"
    )


class AgreementDate(ExtractedClause):
    """Date when agreement was signed/executed."""
    answer: Optional[str] = Field(
        None,
        description="Date in YYYY-MM-DD format if possible, otherwise as written"
    )


class EffectiveDate(ExtractedClause):
    """Date when agreement becomes effective."""
    answer: Optional[str] = None


class ExpirationDate(ExtractedClause):
    """Date when agreement expires."""
    answer: Optional[str] = None


class GoverningLaw(ExtractedClause):
    """
    Jurisdiction/governing law.

    Example: "California" from "This Agreement shall be governed by California law"

    Rationale: Normalization is key here.
    - text: "This Agreement shall be governed by the laws of the State of California"
    - answer: "California"

    LLM extracts normalized form for easier filtering/searching.
    """
    pass


class PaymentTerms(ExtractedClause):
    """
    Payment terms and conditions.

    Rationale: Often contains complex nested information:
    - Payment schedule
    - Amount
    - Frequency
    - Late fees

    We extract the full clause text, LLM can parse sub-components if needed.
    """
    pass


class TerminationClause(ExtractedClause):
    """How and when the agreement can be terminated."""
    pass


class RenewalTerm(ExtractedClause):
    """Terms for automatic renewal or extension."""
    pass


class NoticeToTerminate(ExtractedClause):
    """Notice period required to terminate agreement."""
    pass


class LiabilityClause(ExtractedClause):
    """Limitation of liability provisions."""
    pass


class IndemnityClause(ExtractedClause):
    """Indemnification obligations."""
    pass


class ConfidentialityClause(ExtractedClause):
    """Confidentiality and NDA provisions."""
    pass


class NonCompete(ExtractedClause):
    """Non-compete restrictions."""
    pass


class IPOwnership(ExtractedClause):
    """Intellectual property ownership terms."""
    pass


class ExtractedContract(BaseModel):
    """
    Complete extracted contract data.

    Design rationale:
    - Each clause is Optional (not all contracts have all clauses)
    - Metadata tracks extraction quality
    - to_dict() enables easy JSON serialization for API responses

    This structure makes it easy to:
    - Add new clause types (just add a field)
    - Track extraction quality (metadata)
    - Compare against ground truth (CUAD validation)
    """

    # Document metadata
    filename: str = Field(..., description="Original PDF filename")

    # Extracted clauses (15 types to start)
    document_name: Optional[DocumentName] = None
    parties: Optional[Parties] = None
    agreement_date: Optional[AgreementDate] = None
    effective_date: Optional[EffectiveDate] = None
    expiration_date: Optional[ExpirationDate] = None
    governing_law: Optional[GoverningLaw] = None
    payment_terms: Optional[PaymentTerms] = None
    termination_clause: Optional[TerminationClause] = None
    renewal_term: Optional[RenewalTerm] = None
    notice_to_terminate: Optional[NoticeToTerminate] = None
    liability_clause: Optional[LiabilityClause] = None
    indemnity_clause: Optional[IndemnityClause] = None
    confidentiality_clause: Optional[ConfidentialityClause] = None
    non_compete: Optional[NonCompete] = None
    ip_ownership: Optional[IPOwnership] = None

    # Extraction metadata
    extraction_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the extraction process"
    )

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "filename": "TechCorp_MSA.pdf",
                "document_name": {
                    "text": "MASTER SERVICES AGREEMENT",
                    "answer": "Master Services Agreement",
                    "confidence": "high"
                },
                "parties": {
                    "text": "between TechCorp Inc. and Client LLC",
                    "answer": ["TechCorp Inc.", "Client LLC"],
                    "confidence": "high"
                },
                "governing_law": {
                    "text": "governed by California law",
                    "answer": "California",
                    "confidence": "high"
                }
            }
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Rationale: Pydantic's .model_dump() is great, but we want custom formatting:
        - Exclude None values (cleaner JSON)
        - Include only extracted clauses (not empty objects)
        - Easy to extend with custom serialization logic
        """
        return self.model_dump(exclude_none=True, exclude_unset=True)

    def get_clause_names(self) -> List[str]:
        """
        Get list of clause types that were extracted (non-None).

        Useful for:
        - Debugging (which clauses were found?)
        - Metrics (extraction coverage %)
        - Validation (compare against CUAD ground truth)
        """
        clause_fields = [
            'document_name', 'parties', 'agreement_date', 'effective_date',
            'expiration_date', 'governing_law', 'payment_terms',
            'termination_clause', 'renewal_term', 'notice_to_terminate',
            'liability_clause', 'indemnity_clause', 'confidentiality_clause',
            'non_compete', 'ip_ownership'
        ]
        return [field for field in clause_fields if getattr(self, field) is not None]