"""
Validate extraction quality against CUAD ground truth.

Purpose:
- Extract clauses from multiple CUAD contracts
- Compare against ground truth labels
- Calculate accuracy metrics
- Identify problem areas

This is critical for:
- Verifying system works reliably
- Understanding extraction quality
- Prompt optimization
- Interview preparation (know your numbers!)
"""

from pathlib import Path
import json
from typing import Dict, List, Any
from openai import OpenAI

from src.extractors.extraction_service import ExtractionService
from src.utils.extraction_config import ExtractionConfig
from src.utils.config import settings
from src.utils.dataset import CUADDataset
from src.parsers.pdf_parser import PDFParser


class ExtractionValidator:
    """
    Validates extraction quality against CUAD ground truth.

    Design Rationale:
    - Separate validation logic from extraction logic (SRP)
    - Reusable for different validation scenarios
    - Tracks metrics for analysis

    Metrics tracked:
    - Precision: Of clauses we extracted, how many were correct?
    - Recall: Of clauses that exist, how many did we find?
    - F1 Score: Harmonic mean of precision and recall
    """

    def __init__(
            self,
            extraction_service: ExtractionService,
            pdf_parser: PDFParser,
            cuad_dataset: CUADDataset,
    ):
        """
        Initialize validator.

        Args:
            extraction_service: Service for extracting clauses
            pdf_parser: Parser for reading PDFs
            cuad_dataset: CUAD dataset with ground truth

        Rationale: Dependency injection - all components passed in.
        Makes testing easy (can mock each component).
        """
        self.extraction_service = extraction_service
        self.pdf_parser = pdf_parser
        self.cuad = cuad_dataset

        # Track results
        self.results = []

    def validate_contract(
            self,
            contract_filename: str,
            verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Validate extraction for a single contract.

        Args:
            contract_filename: Name of contract file
            verbose: Whether to print detailed results

        Returns:
            Dictionary with validation results

        Process:
        1. Parse PDF
        2. Extract clauses with LLM
        3. Load ground truth from CUAD
        4. Compare extracted vs ground truth
        5. Calculate metrics

        Rationale: This is the core validation logic.
        We compare what we extracted vs what CUAD experts labeled.
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Validating: {contract_filename}")
            print(f"{'=' * 70}")

        # Step 1: Parse PDF
        pdf_path = self.cuad.pdf_dir / contract_filename

        try:
            parsed_pdf = self.pdf_parser.parse(pdf_path)

            if verbose:
                print(f" Parsed PDF: {len(parsed_pdf.full_text):,} characters")

        except Exception as e:
            print(f" Failed to parse PDF: {str(e)}")
            return {
                'filename': contract_filename,
                'status': 'parse_failed',
                'error': str(e)
            }

        # Step 2: Extract clauses
        try:
            extracted = self.extraction_service.extract(
                parsed_pdf.full_text,
                filename=contract_filename
            )

            extracted_clauses = extracted.get_clause_names()

            if verbose:
                print(f" Extracted {len(extracted_clauses)} clauses")

        except Exception as e:
            print(f" Failed to extract: {str(e)}")
            return {
                'filename': contract_filename,
                'status': 'extraction_failed',
                'error': str(e)
            }

        # Step 3: Load ground truth
        try:
            ground_truth = self.cuad.get_ground_truth(contract_filename)

            if verbose:
                print(f" Loaded ground truth")

        except Exception as e:
            print(f" No ground truth available: {str(e)}")
            return {
                'filename': contract_filename,
                'status': 'no_ground_truth',
                'extracted_clauses': extracted_clauses,
            }

        # Step 4: Compare extraction vs ground truth
        comparison = self._compare_extraction(
            extracted,
            ground_truth,
            verbose=verbose
        )

        result = {
            'filename': contract_filename,
            'status': 'success',
            'extracted_count': len(extracted_clauses),
            'ground_truth_count': comparison['ground_truth_count'],
            'matches': comparison['matches'],
            'misses': comparison['misses'],
            'false_positives': comparison['false_positives'],
            'accuracy_by_clause': comparison['accuracy_by_clause'],
        }

        self.results.append(result)
        return result

    def _compare_extraction(
            self,
            extracted: Any,
            ground_truth: Dict[str, Any],
            verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compare extracted clauses against ground truth.

        Args:
            extracted: ExtractedContract from our system
            ground_truth: Ground truth from CUAD
            verbose: Whether to print comparison

        Returns:
            Comparison metrics

        Rationale: Clause-by-clause comparison logic.
        For each clause type, we check:
        - Did we extract it? (True/False)
        - Does ground truth have it? (True/False)
        - Match, Miss, or False Positive?
        """
        # Map our clause names to CUAD categories
        # Rationale: Our names might differ slightly from CUAD
        clause_mapping = {
            'document_name': 'Document Name',
            'parties': 'Parties',
            'agreement_date': 'Agreement Date',
            'effective_date': 'Effective Date',
            'expiration_date': 'Expiration Date',
            'governing_law': 'Governing Law',
            'payment_terms': 'Payment Terms',
            'termination_clause': 'Termination For Cause',
            'renewal_term': 'Renewal Term',
            'notice_to_terminate': 'Notice Period To Terminate Renewal',
            'liability_clause': 'Cap On Liability',
            'indemnity_clause': 'Indemnification',
            'confidentiality_clause': 'Confidentiality',
            'non_compete': 'Non-Compete',
            'ip_ownership': 'IP Ownership Assignment',
        }

        matches = []
        misses = []
        false_positives = []
        accuracy_by_clause = {}

        ground_truth_categories = ground_truth.get('categories', {})
        ground_truth_count = sum(
            1 for cat in ground_truth_categories.values()
            if cat.get('has_clause', False)
        )

        if verbose:
            print(f"\n{'─' * 70}")
            print("Clause-by-Clause Comparison:")
            print(f"{'─' * 70}")

        # Check each clause type
        for our_name, cuad_name in clause_mapping.items():
            # What we extracted
            our_clause = getattr(extracted, our_name, None)
            we_found_it = our_clause is not None

            # Ground truth
            gt_clause = ground_truth_categories.get(cuad_name, {})
            gt_has_it = gt_clause.get('has_clause', False)

            # Compare
            if we_found_it and gt_has_it:
                # TRUE POSITIVE - We found it and it exists
                matches.append(our_name)
                status = " == MATCH =="
                accuracy_by_clause[our_name] = 'match'

            elif not we_found_it and gt_has_it:
                # FALSE NEGATIVE - We missed it but it exists
                misses.append(our_name)
                status = "== MISS =="
                accuracy_by_clause[our_name] = 'miss'

            elif we_found_it and not gt_has_it:
                # FALSE POSITIVE - We found it but it doesn't exist
                false_positives.append(our_name)
                status = "== FALSE POS =="
                accuracy_by_clause[our_name] = 'false_positive'

            else:
                # TRUE NEGATIVE - We didn't find it and it doesn't exist
                status = " == ABSENT =="
                accuracy_by_clause[our_name] = 'true_negative'

            if verbose and (we_found_it or gt_has_it):
                print(f"{status} {our_name:25} | Ours: {we_found_it:5} | GT: {gt_has_it:5}")

        if verbose:
            print(f"\n{'─' * 70}")
            print(f"Summary:")
            print(f"   Matches: {len(matches)}")
            print(f"   Misses: {len(misses)}")
            print(f"   False Positives: {len(false_positives)}")

            if len(matches) + len(misses) > 0:
                recall = len(matches) / (len(matches) + len(misses))
                print(f"   Recall: {recall:.1%}")

            if len(matches) + len(false_positives) > 0:
                precision = len(matches) / (len(matches) + len(false_positives))
                print(f"   Precision: {precision:.1%}")

        return {
            'ground_truth_count': ground_truth_count,
            'matches': matches,
            'misses': misses,
            'false_positives': false_positives,
            'accuracy_by_clause': accuracy_by_clause,
        }

    def validate_multiple(
            self,
            num_contracts: int = 10,
            verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Validate extraction on multiple contracts.

        Args:
            num_contracts: Number of contracts to test
            verbose: Whether to print progress

        Returns:
            Aggregated metrics across all contracts

        Rationale: Test on multiple contracts to get statistical significance.
        One contract might be easy/hard, but 10 gives us confidence.
        """
        print(f"\n{'=' * 70}")
        print(f"VALIDATING {num_contracts} CONTRACTS")
        print(f"{'=' * 70}")

        # Get sample contracts
        sample_contracts = self.cuad.get_sample_contracts(n=num_contracts)

        # Validate each
        for i, contract_filename in enumerate(sample_contracts, 1):
            print(f"\n[{i}/{num_contracts}]")
            self.validate_contract(contract_filename, verbose=verbose)

        # Calculate aggregate metrics
        return self.get_aggregate_metrics()

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics across all validated contracts.

        Returns:
            Dictionary with aggregate metrics

        Metrics:
        - Overall recall (what % of clauses did we find?)
        - Overall precision (what % of our extractions were correct?)
        - F1 score (harmonic mean of precision and recall)
        - Per-clause accuracy (which clause types are hardest?)

        Rationale: These are standard ML metrics.
        Interviewers will ask about precision/recall/F1.
        """
        if not self.results:
            return {}

        # Filter successful results
        successful = [r for r in self.results if r['status'] == 'success']

        if not successful:
            return {'error': 'No successful extractions'}

        # Aggregate counts
        total_matches = sum(len(r['matches']) for r in successful)
        total_misses = sum(len(r['misses']) for r in successful)
        total_false_positives = sum(len(r['false_positives']) for r in successful)

        # Calculate metrics
        recall = total_matches / (total_matches + total_misses) if (total_matches + total_misses) > 0 else 0
        precision = total_matches / (total_matches + total_false_positives) if (
                                                                                           total_matches + total_false_positives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Per-clause accuracy
        clause_accuracy = {}
        clause_types = set()
        for r in successful:
            clause_types.update(r['accuracy_by_clause'].keys())

        for clause_type in clause_types:
            matches = sum(
                1 for r in successful
                if r['accuracy_by_clause'].get(clause_type) == 'match'
            )
            total = sum(
                1 for r in successful
                if r['accuracy_by_clause'].get(clause_type) in ['match', 'miss']
            )

            if total > 0:
                clause_accuracy[clause_type] = matches / total

        metrics = {
            'num_contracts': len(successful),
            'total_matches': total_matches,
            'total_misses': total_misses,
            'total_false_positives': total_false_positives,
            'recall': recall,
            'precision': precision,
            'f1_score': f1,
            'clause_accuracy': clause_accuracy,
        }

        # Get cost metrics
        service_metrics = self.extraction_service.get_metrics()
        metrics['total_cost'] = service_metrics['total_cost']
        metrics['avg_cost_per_contract'] = service_metrics['avg_cost_per_extraction']

        return metrics

    def print_summary(self):
        """
        Print summary of validation results.

        Rationale: User-friendly output for analysis.
        Shows what's working and what needs improvement.
        """
        metrics = self.get_aggregate_metrics()

        if not metrics or 'error' in metrics:
            print("\n❌ No validation results available")
            return

        print(f"\n{'=' * 70}")
        print("VALIDATION SUMMARY")
        print(f"{'=' * 70}")

        print(f"\n📊 Overall Metrics:")
        print(f"   Contracts tested: {metrics['num_contracts']}")
        print(
            f"   Recall: {metrics['recall']:.1%} (found {metrics['total_matches']}/{metrics['total_matches'] + metrics['total_misses']} clauses)")
        print(f"   Precision: {metrics['precision']:.1%} ({metrics['total_false_positives']} false positives)")
        print(f"   F1 Score: {metrics['f1_score']:.1%}")

        print(f"\n💰 Cost:")
        print(f"   Total: ${metrics['total_cost']:.4f}")
        print(f"   Per contract: ${metrics['avg_cost_per_contract']:.4f}")

        print(f"\n📋 Clause-Level Accuracy:")
        sorted_clauses = sorted(
            metrics['clause_accuracy'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for clause_name, accuracy in sorted_clauses:
            emoji = "✅" if accuracy >= 0.8 else "⚠️" if accuracy >= 0.5 else "❌"
            print(f"   {emoji} {clause_name:25} {accuracy:.1%}")

        print(f"\n{'=' * 70}")


def main():
    """
    Main validation script.

    Usage:
        poetry run python scripts/validate_extraction.py
    """
    print("CONTRACT EXTRACTION VALIDATION")
    print("This will test extraction on 10 CUAD contracts")
    print("Cost: ~$0.10 total")
    print()

    # Setup
    print("Setting up services...")
    client = OpenAI(api_key=settings.openai_api_key)
    config = ExtractionConfig(
        model="gpt-4o-mini",
        temperature=0.0,
    )
    extraction_service = ExtractionService(client, config)
    pdf_parser = PDFParser(extract_tables=False)
    cuad = CUADDataset()

    # Create validator
    validator = ExtractionValidator(
        extraction_service=extraction_service,
        pdf_parser=pdf_parser,
        cuad_dataset=cuad,
    )

    # Run validation
    validator.validate_multiple(num_contracts=3, verbose=True)

    # Print summary
    validator.print_summary()

    # Save results
    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "validation_results.json"
    with open(results_file, "w") as f:
        json.dump({
            'individual_results': validator.results,
            'aggregate_metrics': validator.get_aggregate_metrics(),
        }, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    main()