"""Inspect master_clauses.csv structure to understand ground truth format"""
import pandas as pd
from pathlib import Path


def inspect_master_clauses():
    """Analyze the structure of master_clauses.csv"""

    csv_path = Path("/Users/angadb/Documents/Angad - Personal/Projects/Generative AI/CUAD_v1/master_clauses.csv")

    if not csv_path.exists():
        print("❌ master_clauses.csv not found")
        return

    print("=" * 80)
    print("📊 MASTER CLAUSES CSV STRUCTURE")
    print("=" * 80)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Basic info
    print(f"\n📏 Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"📄 Contracts: {df.shape[0] - 1}")  # Minus header

    # Show all column names
    print(f"\n📋 All Columns ({len(df.columns)}):")
    print("-" * 80)

    for i, col in enumerate(df.columns, 1):
        print(f"{i:3d}. {col}")

    # Identify category columns
    print("\n" + "=" * 80)
    print("🏷️  IDENTIFIED CATEGORIES")
    print("=" * 80)

    # The first column should be Filename
    print(f"\n📁 Filename column: '{df.columns[0]}'")

    # Remaining columns are category-related
    category_columns = df.columns[1:].tolist()

    # Try to identify the 41 categories
    # They might be named like "Document Name", "Parties", "Agreement Date", etc.
    print(f"\n📝 Category columns ({len(category_columns)}):")

    # Print in groups of 3 for readability
    for i in range(0, len(category_columns), 3):
        group = category_columns[i:i + 3]
        print(f"  {', '.join(group)}")

    # Sample a contract to see data structure
    print("\n" + "=" * 80)
    print("📄 SAMPLE CONTRACT DATA")
    print("=" * 80)

    # Get first contract (row 0)
    sample_contract = df.iloc[0]

    print(f"\n📁 Contract: {sample_contract['Filename']}")
    print("\n📝 Sample clause extractions:")
    print("-" * 80)

    # Show first 5 non-empty categories
    shown = 0
    for col in category_columns[:20]:  # Check first 20 columns
        value = sample_contract[col]
        if pd.notna(value) and str(value).strip():
            print(f"\n🏷️  {col}:")
            value_str = str(value)
            if len(value_str) > 200:
                print(f"   {value_str[:200]}...")
            else:
                print(f"   {value_str}")

            shown += 1
            if shown >= 5:
                break

    # Check for paired columns (text + answer)
    print("\n" + "=" * 80)
    print("🔍 ANALYZING COLUMN PATTERNS")
    print("=" * 80)

    # Look for patterns in column names
    unique_categories = set()
    for col in category_columns:
        # Try to extract base category name
        # Some might be "[Category]" and "[Category]_answer" or similar
        base_name = col.replace("_text", "").replace("_answer", "").replace(" (text)", "").replace(" (answer)", "")
        unique_categories.add(base_name)

    print(f"\n📊 Estimated unique categories: {len(unique_categories)}")
    print("\nFirst 10 unique categories:")
    for i, cat in enumerate(list(unique_categories)[:10], 1):
        print(f"  {i}. {cat}")

    # Check data completeness
    print("\n" + "=" * 80)
    print("📈 DATA COMPLETENESS")
    print("=" * 80)

    for col in category_columns[:10]:  # Check first 10
        non_null = df[col].notna().sum()
        completeness = (non_null / len(df)) * 100
        print(f"  {col[:50]:50s}: {non_null:3d}/{len(df)} ({completeness:5.1f}%)")


if __name__ == "__main__":
    inspect_master_clauses()