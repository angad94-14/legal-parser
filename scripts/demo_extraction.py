"""
Demo script for testing extraction on a single contract.

Usage:
    poetry run python scripts/demo_extraction.py

Rationale: Quick way to test extraction without running full test suite.
Useful for:
- Debugging prompt issues
- Demonstrating to stakeholders
- Quick experiments with different contracts
"""

from pathlib import Path
import json
from openai import OpenAI

from src.extractors.extraction_service import ExtractionService
from src.utils.extraction_config import ExtractionConfig
from src.utils.config import settings
from src.utils.dataset import CUADDataset
from src.parsers.pdf_parser import PDFParser


def demo_extraction():
    """
    Run extraction on a sample contract and display results.

    Rationale: User-friendly way to see extraction in action.
    Shows all extracted clauses in readable format.
    """
    print("=" * 70)
    print("CONTRACT EXTRACTION DEMO")
    print("=" * 70)

    # Setup
    print("\n1️⃣  Setting up services...")
    client = OpenAI(api_key=settings.openai_api_key)
    config = ExtractionConfig(
        model="gpt-4o-mini",  # Cheap and fast for demo
        temperature=0.0,
    )
    extraction_service = ExtractionService(client, config)
    pdf_parser = PDFParser(extract_tables=False)
    cuad = CUADDataset()

    # Get a contract
    print("\n2️⃣  Loading sample contract...")
    sample_contracts = cuad.get_sample_contracts(n=1)
    contract_filename = sample_contracts[0]
    print(f"   📄 Contract: {contract_filename}")

    # Parse PDF
    print("\n3️⃣  Parsing PDF...")
    pdf_path = cuad.pdf_dir / contract_filename
    parsed_pdf = pdf_parser.parse(pdf_path)
    print(f"   ✅ Extracted {len(parsed_pdf.full_text):,} characters")
    print(f"   📊 Pages: {parsed_pdf.num_pages}")

    # Extract clauses
    print("\n4️⃣  Extracting clauses with LLM...")
    print(f"   🤖 Model: {config.model}")
    result = extraction_service.extract(
        parsed_pdf.full_text,
        filename=contract_filename
    )

    # Display results
    print("\n" + "=" * 70)
    print("EXTRACTION RESULTS")
    print("=" * 70)

    clause_names = result.get_clause_names()
    print(f"\n✅ Extracted {len(clause_names)} clauses:")
    print(f"   {', '.join(clause_names)}")

    print("\n" + "-" * 70)
    print("CLAUSE DETAILS")
    print("-" * 70)

    # Display each extracted clause
    for clause_name in clause_names:
        clause = getattr(result, clause_name)
        if clause:
            print(f"\n📌 {clause_name.upper().replace('_', ' ')}")
            print(f"   Confidence: {clause.confidence.value}")
            if clause.page_number:
                print(f"   Page: {clause.page_number}")
            print(f"   Answer: {clause.answer}")
            if clause.text and len(clause.text) < 200:
                print(f"   Text: {clause.text[:200]}...")

    # Display metrics
    print("\n" + "=" * 70)
    print("METRICS")
    print("=" * 70)
    metrics = extraction_service.get_metrics()
    print(f"\n💰 Cost: ${metrics['total_cost']:.4f}")
    print(f"🔢 Tokens: {metrics['total_tokens_used']:,}")
    print(f"📊 Extraction count: {metrics['extraction_count']}")

    # Save to JSON
    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{contract_filename}.json"

    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\n💾 Saved to: {output_file}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE ✅")
    print("=" * 70)


if __name__ == "__main__":
    demo_extraction()