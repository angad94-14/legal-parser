"""Test PDF parser functionality"""
import pytest
from pathlib import Path
from src.parsers.pdf_parser import PDFParser, parse_pdf
from src.utils.dataset import CUADDataset


@pytest.fixture
def cuad_dataset():
    """Fixture to get CUAD dataset"""
    return CUADDataset()


@pytest.fixture
def sample_pdf_path(cuad_dataset):
    """Get path to a sample PDF"""
    contracts = cuad_dataset.get_sample_contracts(n=1)
    print(contracts, '===================')
    pdf_path = cuad_dataset.pdf_dir / contracts[0]
    return pdf_path


def test_pdf_parser_init():
    """Test parser initialization"""
    parser = PDFParser()
    assert parser.extract_tables == True

    parser_no_tables = PDFParser(extract_tables=False)
    assert parser_no_tables.extract_tables == False


def test_parse_pdf_basic(sample_pdf_path):
    """Test basic PDF parsing"""
    parser = PDFParser()
    result = parser.parse(sample_pdf_path)

    # Check basic structure
    assert result.filename == sample_pdf_path.name
    assert result.num_pages > 0
    assert len(result.pages) == result.num_pages
    assert len(result.full_text) > 0


def test_parse_pdf_pages(sample_pdf_path):
    """Test that pages are parsed correctly"""
    parser = PDFParser()
    result = parser.parse(sample_pdf_path)

    # Check each page
    for page in result.pages:
        assert page.page_number > 0
        assert isinstance(page.text, str)
        assert isinstance(page.tables, list)
        assert isinstance(page.metadata, dict)


def test_parse_pdf_metadata(sample_pdf_path):
    """Test metadata extraction"""
    parser = PDFParser()
    result = parser.parse(sample_pdf_path)

    # Metadata should be a dict (even if empty)
    assert isinstance(result.metadata, dict)


def test_parse_pdf_tables(sample_pdf_path):
    """Test table extraction"""
    parser = PDFParser(extract_tables=True)
    result = parser.parse(sample_pdf_path)

    # Tables should be extracted
    assert isinstance(result.tables, list)

    # If tables exist, check structure
    if result.tables:
        table = result.tables[0]
        assert 'page' in table
        assert 'data' in table
        assert 'num_rows' in table
        assert 'num_cols' in table


def test_parse_text_only(sample_pdf_path):
    """Test fast text-only extraction"""
    parser = PDFParser()
    text = parser.parse_text_only(sample_pdf_path)

    assert isinstance(text, str)
    assert len(text) > 0


def test_parse_nonexistent_file():
    """Test parsing non-existent file raises error"""
    parser = PDFParser()
    fake_path = Path("nonexistent.pdf")

    with pytest.raises(FileNotFoundError):
        parser.parse(fake_path)


def test_parsed_document_to_dict(sample_pdf_path):
    """Test converting ParsedDocument to dictionary"""
    parser = PDFParser()
    result = parser.parse(sample_pdf_path)

    result_dict = result.to_dict()

    assert isinstance(result_dict, dict)
    assert 'filename' in result_dict
    assert 'num_pages' in result_dict
    assert 'full_text' in result_dict
    assert 'pages' in result_dict
    assert 'tables' in result_dict


def test_convenience_function(sample_pdf_path):
    """Test convenience parse_pdf function"""
    result = parse_pdf(sample_pdf_path)

    assert result.num_pages > 0
    assert len(result.full_text) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])