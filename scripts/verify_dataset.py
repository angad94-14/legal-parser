"""Verify CUAD dataset structure and content"""
from pathlib import Path
import pandas as pd
import json
from typing import Dict, List


def verify_cuad_dataset():
    """Verify the downloaded CUAD dataset"""

    contracts_dir = Path("data/contracts")

    print("=" * 70)
    print("📁 CUAD Dataset Verification")
    print("=" * 70)

    # Check directory structure
    expected_dirs = [
        "full_contract_pdf",
        "full_contract_txt",
    ]

    expected_files = [
        "master_clauses.csv",
        "legal_group.xlsx",
    ]

    print("\n1️⃣  Checking directory structure...")

    for dir_name in expected_dirs:
        dir_path = contracts_dir / dir_name
        if dir_path.exists():
            count = len(list(dir_path.glob("*")))
            print(f"  ✅ {dir_name}/ ({count} files)")
        else:
            print(f"  ❌ {dir_name}/ (missing)")

    # Check for part directories
    for part in ["part1", "part2", "part3"]:
        part_dir = contracts_dir / part
        if part_dir.exists():
            count = len(list(part_dir.glob("*.pdf")))
            print(f"  ✅ {part}/ ({count} PDFs)")

    print("\n2️⃣  Checking metadata files...")

    for file_name in expected_files:
        file_path = contracts_dir / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {file_name} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {file_name} (missing)")

    # Analyze master_clauses.csv (ground truth)
    print("\n3️⃣  Analyzing master_clauses.csv (Ground Truth)...")

    master_clauses_path = contracts_dir / "master_clauses.csv"

    if master_clauses_path.exists():
        try:
            df = pd.read_csv(master_clauses_path)

            print(f"  📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"  📄 Unique contracts: {df['Filename'].nunique()}")

            # Show clause types (columns)
            clause_columns = [col for col in df.columns if col not in ['Filename', 'Parties']]
            print(f"  🏷️  Clause types: {len(clause_columns)}")

            # Show first few clause types
            print(f"\n  Sample clause types:")
            for clause in clause_columns[:10]:
                print(f"    - {clause}")
            if len(clause_columns) > 10:
                print(f"    ... and {len(clause_columns) - 10} more")

        except Exception as e:
            print(f"  ❌ Error reading CSV: {e}")

    # Check sample PDFs and TXTs
    print("\n4️⃣  Checking sample contracts...")

    pdf_dir = contracts_dir / "full_contract_pdf"
    txt_dir = contracts_dir / "full_contract_txt"

    if pdf_dir.exists():
        pdf_files = list(pdf_dir.glob("*.pdf"))[:3]  # First 3
        print(f"  📄 Sample PDFs:")
        for pdf in pdf_files:
            size_kb = pdf.stat().st_size / 1024
            print(f"    - {pdf.name} ({size_kb:.1f} KB)")

    if txt_dir.exists():
        txt_files = list(txt_dir.glob("*.txt"))[:3]  # First 3
        print(f"  📝 Sample TXTs:")
        for txt in txt_files:
            size_kb = txt.stat().st_size / 1024
            # Read first 100 chars
            sample_text = txt.read_text(encoding='utf-8', errors='ignore')[:100]
            print(f"    - {txt.name} ({size_kb:.1f} KB)")
            print(f"      Preview: {sample_text}...")

    # Create a summary
    print("\n" + "=" * 70)
    print("✅ Dataset Summary")
    print("=" * 70)

    stats = get_dataset_stats(contracts_dir)

    print(f"📊 Total Contracts: ~{stats['total_contracts']}")
    print(f"📄 PDF Files: {stats['pdf_count']}")
    print(f"📝 TXT Files: {stats['txt_count']}")
    print(f"🏷️  Clause Types: {stats['clause_types']}")

    print("\n✅ CUAD dataset verified and ready!")
    print("\n🚀 Next: Build PDF/Text parser (Step 2)")

    return stats


def get_dataset_stats(contracts_dir: Path) -> Dict:
    """Get basic statistics about the dataset"""

    stats = {
        'total_contracts': 0,
        'pdf_count': 0,
        'txt_count': 0,
        'clause_types': 0,
    }

    # Count PDFs
    pdf_dir = contracts_dir / "full_contract_pdf"
    if pdf_dir.exists():
        stats['pdf_count'] = len(list(pdf_dir.glob("*.pdf")))

    # Count TXTs
    txt_dir = contracts_dir / "full_contract_txt"
    if txt_dir.exists():
        stats['txt_count'] = len(list(txt_dir.glob("*.txt")))

    # Count clause types from master_clauses.csv
    master_clauses_path = contracts_dir / "master_clauses.csv"
    if master_clauses_path.exists():
        try:
            df = pd.read_csv(master_clauses_path)
            stats['total_contracts'] = df['Filename'].nunique()
            # Clause columns (excluding metadata columns)
            clause_columns = [col for col in df.columns if col not in ['Filename', 'Parties']]
            stats['clause_types'] = len(clause_columns)
        except:
            pass

    return stats


if __name__ == "__main__":
    verify_cuad_dataset()