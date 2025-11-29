"""
Configuration for contract extraction.

Design Rationale:
- Separates "what to extract" from "how to extract"
- Makes it easy to experiment (change model, temperature, clause types)
- Enables prompt versioning (track in Git)
- Different configs for dev/prod (fast models vs. accurate models)

"""

from typing import List, Dict
from pydantic import BaseModel, Field


class ExtractionConfig(BaseModel):
    """
    Configuration for LLM-based extraction.

    Design Pattern: All LLM parameters in one place.
    Makes it easy to:
    - A/B test different models
    - Optimize cost vs. quality
    - Track what configuration produced what results
    """

    # Model configuration
    model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use. Options: gpt-4o, gpt-4o-mini, gpt-4-turbo"
    )
    # Rationale: gpt-4o-mini is 60x cheaper than gpt-4, good enough for most extractions

    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature. 0 = deterministic, 2 = creative"
    )
    # Rationale: Use 0 for extraction (we want consistency, not creativity)

    max_tokens: int = Field(
        default=4000,
        description="Maximum tokens in response"
    )
    # Rationale: Contract extractions can be long (15 clauses * ~100 tokens each = ~1500 tokens)

    # Extraction configuration
    clause_types: List[str] = Field(
        default=[
            "document_name",
            "parties",
            "agreement_date",
            "effective_date",
            "expiration_date",
            "governing_law",
            "payment_terms",
            "termination_clause",
            "renewal_term",
            "notice_to_terminate",
            "liability_clause",
            "indemnity_clause",
            "confidentiality_clause",
            "non_compete",
            "ip_ownership",
        ],
        description="List of clause types to extract"
    )
    # Rationale: Easy to add/remove clause types without changing code

    # Prompt configuration
    system_prompt: str = Field(
        default="""You are an expert legal contract analyst specializing in commercial agreements.

Your task is to extract specific clauses from contracts with high accuracy and consistency.

For each clause type:
1. Search the entire contract for relevant text
2. Extract the exact text where the clause appears
3. Provide a normalized/simplified answer
4. Rate your confidence: high (90%+), medium (70-90%), low (<70%)
5. Note the page number if available

If a clause is not present in the contract, return null for that clause type.

Be precise and thorough. Legal accuracy is critical.""",
        description="System prompt that defines the LLM's role"
    )
    # Rationale:
    # - Role-playing improves quality ("expert legal analyst")
    # - Clear instructions reduce ambiguity
    # - Confidence rating helps identify uncertain extractions

    include_definitions: bool = Field(
        default=True,
        description="Whether to include clause type definitions in prompt"
    )
    # Rationale: Definitions help LLM understand what to look for

    # Output configuration
    include_confidence: bool = Field(
        default=True,
        description="Whether to ask LLM for confidence scores"
    )

    include_page_numbers: bool = Field(
        default=True,
        description="Whether to ask LLM to identify page numbers"
    )

    # Cost tracking
    track_tokens: bool = Field(
        default=True,
        description="Whether to track token usage for cost analysis"
    )
    # Rationale: MLOps best practice - always track costs in production


class ClauseDefinitions:
    """
    Definitions for each clause type.

    Rationale: LLMs perform better with clear definitions.
    These definitions are included in the prompt to help the model
    understand exactly what we're looking for.

    Design Pattern: Centralized knowledge base.
    - Easy to update definitions
    - Consistent across all extractions
    - Can be reviewed by legal team
    """

    DEFINITIONS: Dict[str, str] = {
        "document_name": """
            The title or name of the contract document.
            Usually found at the top of the first page.
            Example: "MASTER SERVICES AGREEMENT" or "SOFTWARE LICENSE AGREEMENT"
        """,

        "parties": """
            All legal entities entering into the agreement.
            Usually listed at the beginning: "between [Party A] and [Party B]"
            Extract as a list of company/entity names.
            Example: ["TechCorp Inc.", "Client Services LLC"]
        """,

        "agreement_date": """
            The date when the agreement was signed or executed.
            Often phrased as: "dated as of", "executed on", "entered into on"
            Example: "January 15, 2024" or "2024-01-15"
        """,

        "effective_date": """
            The date when the agreement becomes legally effective.
            May be different from agreement_date.
            Look for: "effective as of", "shall become effective"
            Example: "February 1, 2024"
        """,

        "expiration_date": """
            The date when the agreement expires or terminates.
            Look for: "shall expire on", "termination date", "ends on"
            Example: "December 31, 2025"
        """,

        "governing_law": """
            The jurisdiction or state whose laws govern the agreement.
            Usually in a "Governing Law" section near the end.
            Look for: "governed by the laws of", "subject to the laws of"
            Extract just the jurisdiction name.
            Example: "California" or "New York"
        """,

        "payment_terms": """
            Terms describing how and when payments are made.
            Includes: payment amounts, schedules, methods, late fees.
            Look for: "payment terms", "fees", "invoicing", "net 30 days"
            Example: "Client shall pay $10,000 monthly, net 30 days"
        """,

        "termination_clause": """
            Conditions under which the agreement can be terminated.
            Look for: "termination", "may be terminated", "either party may terminate"
            Include both for-cause and convenience termination terms.
        """,

        "renewal_term": """
            Terms for automatic renewal or extension of the agreement.
            Look for: "automatically renew", "renewal term", "extend for"
            Example: "Agreement renews automatically for successive 1-year terms"
        """,

        "notice_to_terminate": """
            How much notice is required to terminate the agreement.
            Look for: "upon [X] days notice", "written notice of termination"
            Example: "90 days written notice"
        """,

        "liability_clause": """
            Limitations on liability and damages.
            Look for: "limitation of liability", "cap on damages", "shall not exceed"
            Example: "Liability shall not exceed fees paid in prior 12 months"
        """,

        "indemnity_clause": """
            Indemnification obligations (who protects whom from what).
            Look for: "indemnify", "hold harmless", "defend against"
            Example: "Provider shall indemnify Client against third-party IP claims"
        """,

        "confidentiality_clause": """
            Confidentiality and non-disclosure obligations.
            Look for: "confidential information", "NDA", "non-disclosure"
            Include: what's confidential, how long, permitted uses
        """,

        "non_compete": """
            Non-competition restrictions.
            Look for: "non-compete", "shall not compete", "competitive business"
            Include: duration, geographic scope, restricted activities
        """,

        "ip_ownership": """
            Intellectual property ownership and assignment terms.
            Look for: "intellectual property", "IP rights", "work for hire", "ownership"
            Example: "All IP created under this agreement belongs to Client"
        """,
    }

    @classmethod
    def get_definition(cls, clause_type: str) -> str:
        """
        Get definition for a specific clause type.

        Args:
            clause_type: Name of the clause type

        Returns:
            Definition string, or empty string if not found

        Rationale: Helper method for building prompts dynamically.
        """
        return cls.DEFINITIONS.get(clause_type, "").strip()

    @classmethod
    def get_all_definitions(cls) -> str:
        """
        Get formatted string of all definitions for prompt.

        Returns:
            Formatted string with all clause definitions

        Rationale: Easy to inject into prompt template.
        Used when include_definitions=True in config.
        """
        definitions = []
        for clause_type, definition in cls.DEFINITIONS.items():
            definitions.append(f"**{clause_type}**: {definition}")
        return "\n\n".join(definitions)


# Default configuration instance
# Rationale: Singleton pattern - one default config, easy to import and use
DEFAULT_CONFIG = ExtractionConfig()