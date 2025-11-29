"""Helper functions for working with CUAD dataset"""
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd
import ast


class CUADDataset:
    """Helper class for accessing CUAD dataset files"""

    # Define the 41 clause categories
    CATEGORIES = [
        "Document Name",
        "Parties",
        "Agreement Date",
        "Effective Date",
        "Expiration Date",
        "Renewal Term",
        "Notice Period To Terminate Renewal",
        "Governing Law",
        "Most Favored Nation",
        "Competitive Restriction Exception",
        "Non-Compete",
        "Exclusivity",
        "No-Solicit Of Customers",
        "No-Solicit Of Employees",
        "Non-Disparagement",
        "Termination For Convenience",
        "Rofr/Rofo/Rofn",
        "Change Of Control",
        "Anti-Assignment",
        "Revenue/Profit Sharing",
        "Price Restrictions",
        "Minimum Commitment",
        "Volume Restriction",
        "Ip Ownership Assignment",
        "Joint Ip Ownership",
        "License Grant",
        "Non-Transferable License",
        "Affiliate License-Licensor",
        "Affiliate License-Licensee",
        "Unlimited/All-You-Can-Eat-License",
        "Irrevocable Or Perpetual License",
        "Source Code Escrow",
        "Post-Termination Services",
        "Audit Rights",
        "Uncapped Liability",
        "Cap On Liability",
        "Liquidated Damages",
        "Warranty Duration",
        "Insurance",
        "Covenant Not To Sue",
        "Third Party Beneficiary",
    ]

    def __init__(self, data_dir: Path = Path("/Users/angadb/Documents/Angad - Personal/Projects/Generative AI/CUAD_v1")):
        self.data_dir = data_dir
        self.pdf_dir = data_dir / "full_contract_pdf"
        self.txt_dir = data_dir / "full_contract_txt"
        self.master_clauses_path = data_dir / "master_clauses.csv"

        # Load master clauses on init
        self._master_clauses_df: Optional[pd.DataFrame] = None

    @property
    def master_clauses_df(self) -> pd.DataFrame:
        """Lazy load master clauses CSV"""
        if self._master_clauses_df is None:
            if self.master_clauses_path.exists():
                self._master_clauses_df = pd.read_csv(self.master_clauses_path)
            else:
                raise FileNotFoundError(f"Master clauses not found: {self.master_clauses_path}")
        return self._master_clauses_df

    def get_contract_list(self) -> List[str]:
        """Get list of all contract filenames"""
        return self.master_clauses_df['Filename'].tolist()

    def get_pdf_path(self, filename: str) -> Path:
        """Get path to PDF file"""
        if not filename.endswith('.pdf'):
            filename = filename + '.pdf'
        return self.pdf_dir / filename

    def get_txt_path(self, filename: str) -> Path:
        """Get path to TXT file"""
        if filename.endswith('.pdf'):
            filename = filename[:-4] + '.txt'
        elif not filename.endswith('.txt'):
            filename = filename + '.txt'
        return self.txt_dir / filename

    def get_contract_text(self, filename: str) -> str:
        """Read contract text from TXT file"""
        txt_path = self.get_txt_path(filename)
        if txt_path.exists():
            return txt_path.read_text(encoding='utf-8', errors='ignore')
        else:
            raise FileNotFoundError(f"Text file not found: {txt_path}")

    def parse_list_string(self, value: Any) -> List[str]:
        """Parse string that looks like a list into actual list"""


        if isinstance(value, list):
            return value

        # Try to parse as Python list literal
        try:
            if isinstance(value, str) and value.startswith('['):
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
        except:
            pass

        # Otherwise treat as single item
        return [str(value)] if value else []

    def get_ground_truth(self, filename: str) -> Dict:
        """
        Get ground truth clause labels for a contract

        Returns dict with structure:
        {
            'filename': str,
            'categories': {
                'Document Name': {
                    'text': [...],  # Raw extracted text (can be list)
                    'answer': str,  # Human-normalized answer
                    'has_clause': bool
                },
                ...
            }
        }
        """
        # Remove extension if present
        if filename.endswith('.pdf'):
            filename = filename[:-4]

        # Find row in master_clauses
        row = self.master_clauses_df[
            self.master_clauses_df['Filename'].str.replace('.pdf', '') == filename
        ]

        if row.empty:
            raise ValueError(f"Contract not found in master_clauses: {filename}")

        row_data = row.iloc[0]

        # Build structured ground truth
        ground_truth = {
            'filename': row_data['Filename'],
            'categories': {}
        }

        # Process each category
        for category in self.CATEGORIES:
            text_col = category
            answer_col = f"{category}-Answer"

            # Handle special case: some columns have space before "Answer"
            if answer_col not in row_data.index:
                answer_col = f"{category}- Answer"  # Note the space

            text_value = row_data.get(text_col)
            answer_value = row_data.get(answer_col)

            # Parse text (might be a list string)
            text_parsed = self.parse_list_string(text_value)

            # Check if clause exists
            has_clause = (
                (pd.notna(answer_value) and str(answer_value).strip() != '') or
                (len(text_parsed) > 0)
            )

            ground_truth['categories'][category] = {
                'text': text_parsed,
                'answer': answer_value if pd.notna(answer_value) else None,
                'has_clause': has_clause
            }

        return ground_truth

    def get_clause_types(self) -> List[str]:
        """Get list of all 41 clause types"""
        return self.CATEGORIES.copy()

    def get_sample_contracts(self, n: int = 5) -> List[str]:
        """Get a few sample contracts for testing"""
        return self.get_contract_list()[:n]


# Convenience instance
cuad = CUADDataset()