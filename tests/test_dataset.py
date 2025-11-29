"""Test CUAD dataset helper functions"""
import pytest
from pathlib import Path
from src.utils.dataset import CUADDataset


def test_cuad_dataset_init():
    """Test dataset initialization"""
    dataset = CUADDataset()
    assert dataset.data_dir.exists()
    assert dataset.pdf_dir.exists()
    assert dataset.txt_dir.exists()


def test_get_contract_list():
    """Test getting contract list"""
    dataset = CUADDataset()
    contracts = dataset.get_contract_list()

    assert len(contracts) > 0
    assert len(contracts) == 510
    assert all(isinstance(c, str) for c in contracts)


def test_get_clause_types():
    """Test getting clause types"""
    dataset = CUADDataset()
    clause_types = dataset.get_clause_types()

    assert len(clause_types) == 41  # CUAD has 41 clause types
    assert "Document Name" in clause_types
    assert "Governing Law" in clause_types
    assert "Cap On Liability" in clause_types


def test_get_contract_text():
    """Test reading contract text"""
    dataset = CUADDataset()

    # Get first contract
    first_contract = dataset.get_sample_contracts(n=1)[0]

    # Read text
    text = dataset.get_contract_text(first_contract)

    assert len(text) > 0
    assert isinstance(text, str)


def test_get_ground_truth():
    """Test getting ground truth labels"""
    dataset = CUADDataset()

    # Get first contract
    first_contract = dataset.get_sample_contracts(n=1)[0]

    # Get ground truth
    ground_truth = dataset.get_ground_truth(first_contract)

    assert isinstance(ground_truth, dict)
    assert 'filename' in ground_truth
    assert 'categories' in ground_truth
    assert isinstance(ground_truth['categories'], dict)

    # Check structure of a category
    if 'Document Name' in ground_truth['categories']:
        doc_name = ground_truth['categories']['Document Name']
        assert 'text' in doc_name
        assert 'answer' in doc_name
        assert 'has_clause' in doc_name
        assert isinstance(doc_name['text'], list)


def test_parse_list_string():
    """Test parsing list strings"""
    dataset = CUADDataset()

    # Test actual list
    result = dataset.parse_list_string(['item1', 'item2'])
    assert result == ['item1', 'item2']

    # Test string representation of list
    result = dataset.parse_list_string("['item1', 'item2']")
    assert result == ['item1', 'item2']

    # Test single string
    result = dataset.parse_list_string("single item")
    assert result == ['single item']

    # Test NaN
    result = dataset.parse_list_string(None)
    assert result == []


def test_ground_truth_structure():
    """Test that ground truth has expected structure"""
    dataset = CUADDataset()

    # Get a contract we know has data
    contract = "CybergyHoldingsInc_20140520_10-Q_EX-10.27_8605784_EX-10.27_Affiliate Agreement.pdf"

    ground_truth = dataset.get_ground_truth(contract)

    # Check Document Name
    doc_name = ground_truth['categories']['Document Name']
    assert doc_name['has_clause'] == True
    assert doc_name['answer'] == "MARKETING AFFILIATE AGREEMENT"
    assert isinstance(doc_name['text'], list)
    assert len(doc_name['text']) > 0

    # Check Parties
    parties = ground_truth['categories']['Parties']
    assert parties['has_clause'] == True
    assert parties['answer'] is not None
    assert isinstance(parties['text'], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])