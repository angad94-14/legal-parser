"""
Extraction API endpoints.

Handles contract clause extraction via LLM.

Endpoints:
- POST /extract - Extract clauses from single contract
- POST /extract/batch - Extract from multiple contracts
- GET /extract/{extraction_id} - Get extraction results

Design Rationale:
- Synchronous extraction (user waits for result)
- In production: Use async task queue (Celery, RQ)
- Store results for retrieval
- Provide progress tracking

Interview Note: Shows you understand API design for ML services.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from typing import Dict
import uuid
from datetime import datetime
import logging
from pathlib import Path
import tempfile

from openai import OpenAI

from src.api.models import (
    ExtractRequest,
    ExtractResponse,
    ClauseResponse,
    ExtractionStatus,
    BatchExtractRequest,
    BatchExtractResponse,
)
from src.extractors.extraction_service import ExtractionService
from src.utils.extraction_config import ExtractionConfig
from src.parsers.pdf_parser import PDFParser
from src.utils.config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# In-memory storage for extractions
# Rationale: Simple storage for MVP
# Production: Use Redis, PostgreSQL, or S3
EXTRACTIONS: Dict[str, ExtractResponse] = {}


def convert_to_api_response(
        extracted_contract,
        extraction_id: str,
        filename: str
) -> ExtractResponse:
    """
    Convert internal ExtractedContract to API response.

    Rationale: Separation between internal models and API models.
    - Internal model can change without breaking API
    - API model only exposes what clients need
    - Transformation layer for data mapping

    Args:
        extracted_contract: ExtractedContract from extraction service
        extraction_id: Unique ID for this extraction
        filename: Original filename

    Returns:
        ExtractResponse for API
    """
    # Convert clauses to API format
    clauses = {}

    for clause_name in extracted_contract.get_clause_names():
        clause = getattr(extracted_contract, clause_name)

        if clause:
            clauses[clause_name] = ClauseResponse(
                text=clause.text,
                answer=clause.answer,
                confidence=clause.confidence.value,
                page_number=clause.page_number
            )

    # Build response
    return ExtractResponse(
        extraction_id=extraction_id,
        filename=filename,
        status=ExtractionStatus.COMPLETED,
        timestamp=datetime.utcnow(),
        clauses=clauses,
        metadata=extracted_contract.extraction_metadata
    )


@router.post("/", response_model=ExtractResponse, status_code=status.HTTP_200_OK)
async def extract_contract(file: UploadFile = File(...)):
    """
    Extract clauses from a single contract PDF.

    **Process:**
    1. Receive PDF file upload
    2. Parse PDF → Extract text
    3. Extract clauses with LLM
    4. Return structured JSON

    **Cost:** ~$0.01 per contract
    **Time:** ~5-10 seconds

    **Example:**
```bash
    curl -X POST "http://localhost:8000/extract/" \\
         -F "file=@contract.pdf"
```

    Args:
        file: PDF file upload

    Returns:
        ExtractResponse with extracted clauses

    Raises:
        HTTPException: If extraction fails
    """
    extraction_id = f"ext_{uuid.uuid4().hex[:12]}"

    logger.info(f"[{extraction_id}] Starting extraction for: {file.filename}")

    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )

    try:
        # Save uploaded file to temporary location
        # Rationale: PDF parser needs file path, not file bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        # Parse PDF
        logger.info(f"[{extraction_id}] Parsing PDF...")
        pdf_parser = PDFParser(extract_tables=False)
        parsed_pdf = pdf_parser.parse(tmp_path)

        logger.info(
            f"[{extraction_id}] Parsed {len(parsed_pdf.full_text):,} characters"
        )

        # Extract clauses
        logger.info(f"[{extraction_id}] Extracting clauses with LLM...")
        client = OpenAI(api_key=settings.openai_api_key)
        config = ExtractionConfig()
        extraction_service = ExtractionService(client, config)

        extracted = extraction_service.extract(
            parsed_pdf.full_text,
            filename=file.filename
        )

        logger.info(
            f"[{extraction_id}] Extracted {len(extracted.get_clause_names())} clauses"
        )

        # Convert to API response
        response = convert_to_api_response(
            extracted,
            extraction_id,
            file.filename
        )

        # Store result
        # Rationale: Allow retrieval later via GET endpoint
        EXTRACTIONS[extraction_id] = response

        # Cleanup temp file
        tmp_path.unlink()

        logger.info(f"[{extraction_id}] Extraction completed successfully")

        return response

    except Exception as e:
        logger.error(f"[{extraction_id}] Extraction failed: {str(e)}", exc_info=True)

        # Cleanup temp file if it exists
        try:
            if 'tmp_path' in locals():
                tmp_path.unlink()
        except:
            pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(e)}"
        )


@router.get("/{extraction_id}", response_model=ExtractResponse)
async def get_extraction(extraction_id: str):
    """
    Retrieve extraction results by ID.

    **Use case:**
    - User uploads file
    - Gets extraction_id back
    - Can retrieve results later

    **Example:**
```bash
    curl "http://localhost:8000/extract/ext_abc123"
```

    Args:
        extraction_id: Unique extraction ID

    Returns:
        ExtractResponse with results

    Raises:
        HTTPException: If extraction not found
    """
    if extraction_id not in EXTRACTIONS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Extraction {extraction_id} not found"
        )

    return EXTRACTIONS[extraction_id]


@router.post("/batch", response_model=BatchExtractResponse)
async def extract_batch(request: BatchExtractRequest):
    """
    Extract from multiple contracts (batch processing).

    **Note:** This is a simplified synchronous implementation.
    **Production:** Use async task queue (Celery, RQ, AWS Batch)

    **Process:**
    1. Queue all extraction jobs
    2. Process sequentially (or in parallel with thread pool)
    3. Return batch results

    **Cost:** ~$0.01 × N contracts
    **Time:** ~5s × N contracts

    **Example:**
```bash
    curl -X POST "http://localhost:8000/extract/batch" \\
         -H "Content-Type: application/json" \\
         -d '{
           "urls": [
             "https://example.com/contract1.pdf",
             "https://example.com/contract2.pdf"
           ]
         }'
```

    Args:
        request: BatchExtractRequest with list of URLs

    Returns:
        BatchExtractResponse with all results

    Raises:
        HTTPException: If batch processing fails
    """
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"

    logger.info(
        f"[{batch_id}] Starting batch extraction for {len(request.urls)} documents"
    )

    # In production: Queue jobs, return batch_id, process async
    # For MVP: Process synchronously

    results = []

    for i, url in enumerate(request.urls, 1):
        logger.info(f"[{batch_id}] Processing {i}/{len(request.urls)}: {url}")

        try:
            # Download PDF from URL
            # Note: In production, add timeout, size limits, virus scanning
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(str(url), timeout=30.0)
                response.raise_for_status()
                pdf_content = response.content

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_content)
                tmp_path = Path(tmp_file.name)

            # Parse and extract
            pdf_parser = PDFParser(extract_tables=request.extract_tables)
            parsed_pdf = pdf_parser.parse(tmp_path)

            client_openai = OpenAI(api_key=settings.openai_api_key)
            config = ExtractionConfig()
            extraction_service = ExtractionService(client_openai, config)

            extracted = extraction_service.extract(
                parsed_pdf.full_text,
                filename=url.path.split('/')[-1]
            )

            # Convert to API response
            extraction_id = f"ext_{uuid.uuid4().hex[:12]}"
            result = convert_to_api_response(
                extracted,
                extraction_id,
                url.path.split('/')[-1]
            )

            results.append(result)

            # Store result
            EXTRACTIONS[extraction_id] = result

            # Cleanup
            tmp_path.unlink()

            logger.info(f"[{batch_id}] ✅ Completed {i}/{len(request.urls)}")

        except Exception as e:
            logger.error(
                f"[{batch_id}] ❌ Failed {i}/{len(request.urls)}: {str(e)}"
            )
            # Continue processing other documents

    logger.info(
        f"[{batch_id}] Batch complete: {len(results)}/{len(request.urls)} successful"
    )

    return BatchExtractResponse(
        batch_id=batch_id,
        total_documents=len(request.urls),
        status="completed",
        results=results
    )


@router.delete("/{extraction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_extraction(extraction_id: str):
    """
    Delete extraction results.

    **Use case:** Clean up old extractions to save memory

    **Example:**
```bash
    curl -X DELETE "http://localhost:8000/extract/ext_abc123"
```

    Args:
        extraction_id: Unique extraction ID

    Raises:
        HTTPException: If extraction not found
    """
    if extraction_id not in EXTRACTIONS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Extraction {extraction_id} not found"
        )

    del EXTRACTIONS[extraction_id]
    logger.info(f"Deleted extraction: {extraction_id}")