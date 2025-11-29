"""
PDF parsing module for extracting text, tables, and structure from contracts.

Handles:
- Text extraction with layout preservation
- Table detection and extraction
- Multi-column layouts
- Page metadata
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import fitz  # PyMuPDF
import pdfplumber
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    """Represents a parsed page from a PDF"""
    page_number: int
    text: str
    tables: List[List[List[str]]]  # List of tables, each table is list of rows
    metadata: Dict[str, Any]


@dataclass
class ParsedDocument:
    """Represents a complete parsed PDF document"""
    filename: str
    num_pages: int
    pages: List[ParsedPage]
    full_text: str
    metadata: Dict[str, Any]
    tables: List[Dict[str, Any]]  # All tables with page numbers

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'filename': self.filename,
            'num_pages': self.num_pages,
            'full_text': self.full_text,
            'metadata': self.metadata,
            'pages': [
                {
                    'page_number': p.page_number,
                    'text': p.text,
                    'num_tables': len(p.tables),
                }
                for p in self.pages
            ],
            'tables': self.tables,
        }


class PDFParser:
    """
    PDF parser with support for text, tables, and structure extraction.

    Uses PyMuPDF for fast text extraction and pdfplumber for table detection.
    """

    def __init__(self, extract_tables: bool = True):
        """
        Initialize PDF parser.

        Args:
            extract_tables: Whether to extract tables (slower but more accurate)
        """
        self.extract_tables = extract_tables

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """
        Parse a PDF file and extract all content.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ParsedDocument with all extracted content

        Raises:
            FileNotFoundError: If PDF doesn't exist
            ValueError: If PDF is corrupted or can't be read
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Parsing PDF: {pdf_path.name}")

        try:
            # Extract metadata and basic info
            metadata = self._extract_metadata(pdf_path)

            # Parse each page
            pages = self._parse_pages(pdf_path)

            # Extract all tables with page references
            all_tables = []
            for page in pages:
                for table_idx, table in enumerate(page.tables):
                    all_tables.append({
                        'page': page.page_number,
                        'table_index': table_idx,
                        'data': table,
                        'num_rows': len(table),
                        'num_cols': len(table[0]) if table else 0,
                    })

            # Combine all text
            full_text = "\n\n".join(page.text for page in pages)

            parsed_doc = ParsedDocument(
                filename=pdf_path.name,
                num_pages=len(pages),
                pages=pages,
                full_text=full_text,
                metadata=metadata,
                tables=all_tables,
            )

            logger.info(f"Successfully parsed {pdf_path.name}: {len(pages)} pages, {len(all_tables)} tables")
            return parsed_doc

        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path.name}: {str(e)}")
            raise ValueError(f"Failed to parse PDF: {str(e)}")

    def _extract_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract PDF metadata (author, creation date, etc.)"""
        metadata = {}

        try:
            with fitz.open(pdf_path) as doc:
                pdf_metadata = doc.metadata
                metadata = {
                    'title': pdf_metadata.get('title', ''),
                    'author': pdf_metadata.get('author', ''),
                    'subject': pdf_metadata.get('subject', ''),
                    'creator': pdf_metadata.get('creator', ''),
                    'producer': pdf_metadata.get('producer', ''),
                    'creation_date': pdf_metadata.get('creationDate', ''),
                    'modification_date': pdf_metadata.get('modDate', ''),
                }
        except Exception as e:
            logger.warning(f"Could not extract metadata: {str(e)}")

        return metadata

    def _parse_pages(self, pdf_path: Path) -> List[ParsedPage]:
        """Parse all pages in the PDF"""
        pages = []

        # Use PyMuPDF for text extraction (fast)
        with fitz.open(pdf_path) as doc:
            # Use pdfplumber for table extraction (more accurate)
            with pdfplumber.open(pdf_path) as plumber_pdf:
                for page_num in range(len(doc)):
                    mupdf_page = doc[page_num]
                    plumber_page = plumber_pdf.pages[page_num]

                    # Extract text with PyMuPDF
                    text = mupdf_page.get_text()

                    # Extract tables with pdfplumber
                    tables = []
                    if self.extract_tables:
                        try:
                            extracted_tables = plumber_page.extract_tables()
                            if extracted_tables:
                                tables = extracted_tables
                        except Exception as e:
                            logger.warning(f"Could not extract tables from page {page_num + 1}: {str(e)}")

                    # Page metadata
                    page_metadata = {
                        'width': mupdf_page.rect.width,
                        'height': mupdf_page.rect.height,
                        'rotation': mupdf_page.rotation,
                    }

                    pages.append(ParsedPage(
                        page_number=page_num + 1,
                        text=text,
                        tables=tables,
                        metadata=page_metadata,
                    ))

        return pages

    def parse_text_only(self, pdf_path: Path) -> str:
        """
        Fast text-only extraction without tables or structure.

        Useful for simple documents or when speed is critical.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        full_text = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text()
                full_text.append(text)

        return "\n\n".join(full_text)


# Convenience function for quick parsing
def parse_pdf(pdf_path: Path, extract_tables: bool = True) -> ParsedDocument:
    """
    Parse a PDF file.

    Args:
        pdf_path: Path to PDF file
        extract_tables: Whether to extract tables

    Returns:
        ParsedDocument with extracted content
    """
    parser = PDFParser(extract_tables=extract_tables)
    return parser.parse(pdf_path)