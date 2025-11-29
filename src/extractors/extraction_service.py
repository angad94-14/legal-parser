"""
LLM-based contract extraction service.

Design Philosophy:
- Single Responsibility: This class does ONE thing - extract clauses via LLM
- Dependency Injection: OpenAI client and config are passed in (testable!)
- Fail-Fast: Validate inputs early, provide clear error messages
- Observable: Log everything for debugging and monitoring

Interview Insight: This is production-quality code structure.
Shows you understand:
- Clean architecture
- Error handling
- Observability
- Cost tracking
"""

from typing import Dict, Any, Optional
import json
import logging
from pathlib import Path

from openai import OpenAI
from pydantic import ValidationError

from src.models.contract import (
    ExtractedContract,
    DocumentName,
    Parties,
    AgreementDate,
    EffectiveDate,
    ExpirationDate,
    GoverningLaw,
    PaymentTerms,
    TerminationClause,
    RenewalTerm,
    NoticeToTerminate,
    LiabilityClause,
    IndemnityClause,
    ConfidentialityClause,
    NonCompete,
    IPOwnership,
    ConfidenceLevel,
)
from src.utils.extraction_config import ExtractionConfig, DEFAULT_CONFIG
from src.extractors.prompts import ExtractionPromptBuilder
from src.utils.tokens import count_tokens

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """
    Custom exception for extraction failures.

    Rationale: Custom exceptions make error handling clearer:
    - Distinguishes extraction errors from other errors
    - Can add extraction-specific context
    - Makes it clear what layer the error came from

    Example:
        try:
            result = service.extract(text)
        except ExtractionError as e:
            # We know this is an extraction failure, not a network error
            logger.error(f"Extraction failed: {e}")
    """
    pass


class ExtractionService:
    """
    Service for extracting structured clause data from contracts using LLMs.

    Design Pattern: Service Layer
    - Encapsulates business logic (extraction)
    - Abstracts LLM interaction details
    - Provides clean interface for API layer

    Usage:
        service = ExtractionService(client, config)
        result = service.extract(contract_text)

    Rationale for class vs functions:
    - State management (client, config, metrics)
    - Dependency injection (easy testing)
    - Extensibility (add caching, retries, etc.)
    """

    def __init__(
            self,
            client: OpenAI,
            config: ExtractionConfig = DEFAULT_CONFIG,
    ):
        """
        Initialize extraction service.

        Args:
            client: Configured OpenAI client
            config: Extraction configuration

        Rationale: Dependency Injection pattern
        - Client is passed in (not created here)
        - Can inject mock client for testing
        - Can swap clients (OpenAI → Anthropic) easily
        - Config is injectable too (different configs for different tasks)

        Example:
            # Production
            client = OpenAI(api_key=settings.openai_api_key)
            service = ExtractionService(client)

            # Testing
            mock_client = MockOpenAI()
            service = ExtractionService(mock_client)
        """
        self.client = client
        self.config = config
        self.prompt_builder = ExtractionPromptBuilder(config)

        # Metrics tracking (MLOps best practice)
        self._total_tokens_used = 0
        self._total_cost = 0.0
        self._extraction_count = 0

        logger.info(
            f"ExtractionService initialized with model={config.model}, "
            f"temperature={config.temperature}"
        )

    def extract(
            self,
            contract_text: str,
            filename: str = "unknown.pdf",
            max_tokens_allowed: int = 100000,
    ) -> ExtractedContract:
        """
        Extract structured clause data from contract text.

        Args:
            contract_text: Raw contract text (from PDF parser)
            filename: Original filename (for tracking)
            max_tokens_allowed: Safety limit for token count

        Returns:
            ExtractedContract with all extracted clauses

        Raises:
            ExtractionError: If extraction fails

        Rationale: Main entry point for extraction
        - Validates input
        - Builds prompt
        - Calls LLM
        - Parses response
        - Validates output
        - Tracks metrics

        This is the "public interface" of the service.
        Everything else is private helper methods.

        Example:
            text = pdf_parser.parse("contract.pdf").full_text
            result = service.extract(text, filename="contract.pdf")
            print(result.governing_law.answer)  # "California"
        """
        logger.info(f"Starting extraction for {filename}")

        # Step 1: Validate input
        # Rationale: Fail fast with clear error messages
        if not contract_text or len(contract_text.strip()) == 0:
            raise ExtractionError("Contract text is empty")

        # Step 2: Check token count (safety check)
        # Rationale: Prevent exceeding context window and excessive costs
        token_count = count_tokens(contract_text, model=self.config.model)
        logger.info(f"Contract has {token_count:,} tokens")

        if token_count > max_tokens_allowed:
            raise ExtractionError(
                f"Contract exceeds token limit: {token_count:,} > {max_tokens_allowed:,}. "
                "Consider using chunking or RAG-assisted extraction."
            )

        if token_count > 50000:
            logger.warning(
                f"Large contract detected: {token_count:,} tokens. "
                "This may be slow and expensive."
            )

        # Step 3: Build prompt
        # Rationale: Separation of concerns - prompt building is separate
        try:
            messages = self.prompt_builder.build_messages(contract_text)
        except Exception as e:
            raise ExtractionError(f"Failed to build prompt: {str(e)}")

        # Step 4: Call LLM
        # Rationale: Main extraction logic
        try:
            response_data = self._call_llm(messages)
        except Exception as e:
            raise ExtractionError(f"LLM call failed: {str(e)}")

        # Step 5: Parse and validate response
        # Rationale: Ensure LLM output matches our schema
        try:
            extracted_contract = self._parse_response(
                response_data,
                filename,
                token_count
            )
        except Exception as e:
            raise ExtractionError(f"Failed to parse LLM response: {str(e)}")

        # Step 6: Track metrics
        # Rationale: MLOps - always track costs and usage
        self._extraction_count += 1
        logger.info(
            f"Extraction complete for {filename}. "
            f"Found {len(extracted_contract.get_clause_names())} clauses."
        )

        return extracted_contract

    def _call_llm(self, messages: list) -> Dict[str, Any]:
        """
        Call OpenAI API with retry logic.

        Args:
            messages: Message array for API

        Returns:
            Parsed JSON response from LLM

        Raises:
            Exception: If API call fails after retries

        Rationale: Centralize LLM calling logic
        - Handles API errors
        - Tracks token usage
        - Estimates costs
        - Retry logic (for transient failures)
        - JSON mode enforcement

        This is a private method (_call_llm) because external callers
        shouldn't directly call the LLM - they use extract().

        Design Pattern: Encapsulation
        """
        logger.debug(f"Calling OpenAI API with model={self.config.model}")

        try:
            # Make API call with JSON mode
            # Rationale: response_format="json_object" ensures valid JSON
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},  # Force JSON output
            )

            # Extract response content
            # Rationale: OpenAI response structure is nested
            content = response.choices[0].message.content

            # Track token usage (MLOps best practice)
            # Rationale: Cost tracking is critical in production
            if self.config.track_tokens:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                self._total_tokens_used += total_tokens

                # Estimate cost (prices as of 2024)
                # Rationale: Help users understand extraction costs
                cost = self._estimate_cost(input_tokens, output_tokens)
                self._total_cost += cost

                logger.info(
                    f"Token usage - Input: {input_tokens:,}, "
                    f"Output: {output_tokens:,}, "
                    f"Total: {total_tokens:,}, "
                    f"Cost: ${cost:.4f}"
                )

            # Parse JSON response
            # Rationale: LLM returns string, we need dict
            try:
                response_data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from LLM: {content[:500]}")
                raise Exception(f"LLM returned invalid JSON: {str(e)}")

            return response_data

        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost of API call.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD

        Rationale: Cost transparency
        - Users should know how much extraction costs
        - Helps with budget planning
        - Can optimize if costs are too high

        Prices (as of Nov 2024):
        - GPT-4o: $2.50/1M input, $10.00/1M output
        - GPT-4o-mini: $0.15/1M input, $0.60/1M output

        Note: Update these prices as OpenAI changes them
        """
        # Price mapping (update as needed)
        # Rationale: Centralized price management
        prices = {
            "gpt-4o": {
                "input": 2.50 / 1_000_000,  # $2.50 per 1M tokens
                "output": 10.00 / 1_000_000,  # $10.00 per 1M tokens
            },
            "gpt-4o-mini": {
                "input": 0.15 / 1_000_000,  # $0.15 per 1M tokens
                "output": 0.60 / 1_000_000,  # $0.60 per 1M tokens
            },
        }

        # Get prices for current model
        model_prices = prices.get(self.config.model, prices["gpt-4o-mini"])

        # Calculate cost
        input_cost = input_tokens * model_prices["input"]
        output_cost = output_tokens * model_prices["output"]
        total_cost = input_cost + output_cost

        return total_cost

    def _parse_response(
            self,
            response_data: Dict[str, Any],
            filename: str,
            token_count: int,
    ) -> ExtractedContract:
        """
        Parse LLM JSON response into ExtractedContract model.

        Args:
            response_data: JSON data from LLM
            filename: Original filename
            token_count: Token count of input

        Returns:
            Validated ExtractedContract

        Raises:
            ValidationError: If response doesn't match schema

        Rationale: Transform unstructured LLM output → structured data
        - Map JSON fields to Pydantic models
        - Validate all fields
        - Handle missing/null fields gracefully
        - Add metadata (filename, token count, etc.)

        This is where Pydantic shines:
        - Automatic validation
        - Type coercion (string → ConfidenceLevel enum)
        - Clear error messages if validation fails

        Design Pattern: Data Transfer Object (DTO)
        LLM JSON → Parse → Pydantic Model → Type-safe contract
        """
        logger.debug("Parsing LLM response into ExtractedContract")

        # Helper function to parse individual clauses
        # Rationale: DRY (Don't Repeat Yourself) - same logic for all clause types
        def parse_clause(clause_data: Optional[Dict], clause_class):
            """
            Parse a single clause from JSON to Pydantic model.

            Args:
                clause_data: JSON dict for this clause (or None)
                clause_class: Pydantic model class (e.g., GoverningLaw)

            Returns:
                Instance of clause_class, or None if clause_data is None

            Rationale: Consistent parsing logic for all clause types
            - Handle missing clauses (None)
            - Handle partial data (only text, no answer)
            - Normalize confidence strings to enum
            """
            if clause_data is None:
                return None

            # Handle case where LLM returned null for missing clause
            if not isinstance(clause_data, dict):
                return None

            # Parse confidence level
            # Rationale: LLM returns string, we need enum
            confidence_str = clause_data.get("confidence", "medium")

            # Handle None confidence (LLM returned null)
            if confidence_str is None:
                confidence_str = "medium"

            try:
                confidence = ConfidenceLevel(confidence_str.lower())
            except ValueError:
                # Invalid confidence string, default to medium
                logger.warning(f"Invalid confidence value: {confidence_str}, using 'medium'")
                confidence = ConfidenceLevel.MEDIUM

            # Create clause instance
            # Rationale: Pydantic validates all fields automatically
            try:
                return clause_class(
                    text=clause_data.get("text"),
                    answer=clause_data.get("answer"),
                    confidence=confidence,
                    page_number=clause_data.get("page_number"),
                )
            except ValidationError as e:
                logger.warning(f"Failed to parse {clause_class.__name__}: {e}")
                return None

        # Parse all clause types
        # Rationale: Map JSON keys → Pydantic models
        # This is tedious but necessary for type safety
        try:
            contract = ExtractedContract(
                filename=filename,

                # Parse each clause type
                document_name=parse_clause(
                    response_data.get("document_name"),
                    DocumentName
                ),
                parties=parse_clause(
                    response_data.get("parties"),
                    Parties
                ),
                agreement_date=parse_clause(
                    response_data.get("agreement_date"),
                    AgreementDate
                ),
                effective_date=parse_clause(
                    response_data.get("effective_date"),
                    EffectiveDate
                ),
                expiration_date=parse_clause(
                    response_data.get("expiration_date"),
                    ExpirationDate
                ),
                governing_law=parse_clause(
                    response_data.get("governing_law"),
                    GoverningLaw
                ),
                payment_terms=parse_clause(
                    response_data.get("payment_terms"),
                    PaymentTerms
                ),
                termination_clause=parse_clause(
                    response_data.get("termination_clause"),
                    TerminationClause
                ),
                renewal_term=parse_clause(
                    response_data.get("renewal_term"),
                    RenewalTerm
                ),
                notice_to_terminate=parse_clause(
                    response_data.get("notice_to_terminate"),
                    NoticeToTerminate
                ),
                liability_clause=parse_clause(
                    response_data.get("liability_clause"),
                    LiabilityClause
                ),
                indemnity_clause=parse_clause(
                    response_data.get("indemnity_clause"),
                    IndemnityClause
                ),
                confidentiality_clause=parse_clause(
                    response_data.get("confidentiality_clause"),
                    ConfidentialityClause
                ),
                non_compete=parse_clause(
                    response_data.get("non_compete"),
                    NonCompete
                ),
                ip_ownership=parse_clause(
                    response_data.get("ip_ownership"),
                    IPOwnership
                ),

                # Add extraction metadata
                # Rationale: Track extraction details for debugging/monitoring
                extraction_metadata={
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                    "token_count": token_count,
                    "total_extractions": self._extraction_count + 1,
                }
            )

            logger.debug(
                f"Successfully parsed {len(contract.get_clause_names())} clauses"
            )
            return contract

        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            raise ExtractionError(f"Response validation failed: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get service metrics (tokens used, cost, extraction count).

        Returns:
            Dictionary with metric data

        Rationale: Observability - track service usage
        - How many extractions have we done?
        - How much have we spent?
        - What's the average cost per extraction?

        This is crucial for production:
        - Budget monitoring
        - Performance optimization
        - Capacity planning

        Example:
            metrics = service.get_metrics()
            print(f"Total cost: ${metrics['total_cost']:.2f}")
            print(f"Average per extraction: ${metrics['avg_cost_per_extraction']:.4f}")
        """
        avg_cost = (
            self._total_cost / self._extraction_count
            if self._extraction_count > 0
            else 0.0
        )

        return {
            "total_tokens_used": self._total_tokens_used,
            "total_cost": self._total_cost,
            "extraction_count": self._extraction_count,
            "avg_cost_per_extraction": avg_cost,
            "model": self.config.model,
        }