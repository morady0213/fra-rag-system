"""
PDF Processing Module.

Handles extraction of text from PDF documents:
- Digital PDFs: Uses PyMuPDF (fitz) for fast, accurate extraction
- Scanned PDFs: Placeholder for PaddleOCR integration

Designed to be modular so PDF parsing can be easily swapped.
"""

import io
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Protocol
from dataclasses import dataclass
from loguru import logger

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not installed. Run: pip install PyMuPDF")

# ============================================================================
# PADDLEOCR PLACEHOLDER
# Uncomment and configure when OCR is needed for scanned documents
# ============================================================================
# try:
#     from paddleocr import PaddleOCR
#     PADDLEOCR_AVAILABLE = True
# except ImportError:
#     PADDLEOCR_AVAILABLE = False
#     logger.warning("PaddleOCR not installed. Run: pip install paddleocr paddlepaddle")

PADDLEOCR_AVAILABLE = False  # Set to True when PaddleOCR is configured


@dataclass
class PDFPage:
    """Represents a single page from a PDF document."""
    page_number: int
    text: str
    images: List[bytes] = None
    is_scanned: bool = False
    
    def __post_init__(self):
        if self.images is None:
            self.images = []


@dataclass  
class PDFDocument:
    """Represents a processed PDF document."""
    filename: str
    filepath: str
    pages: List[PDFPage]
    metadata: Dict[str, Any]
    total_pages: int
    
    @property
    def full_text(self) -> str:
        """Get concatenated text from all pages."""
        return "\n\n".join(page.text for page in self.pages if page.text)
    
    @property
    def is_scanned(self) -> bool:
        """Check if any page appears to be scanned."""
        return any(page.is_scanned for page in self.pages)


class PDFParserInterface(Protocol):
    """Protocol defining the interface for PDF parsers."""
    
    def parse(self, filepath: Union[str, Path]) -> PDFDocument:
        """Parse a PDF file and return structured content."""
        ...
    
    def parse_bytes(self, pdf_bytes: bytes, filename: str) -> PDFDocument:
        """Parse PDF from bytes."""
        ...


class PyMuPDFParser:
    """
    PDF parser using PyMuPDF (fitz).
    
    PyMuPDF is excellent for:
    - Fast extraction from digital PDFs
    - Preserving text layout and structure
    - Handling Arabic text with proper RTL support
    """
    
    def __init__(self):
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")
    
    def parse(self, filepath: Union[str, Path]) -> PDFDocument:
        """
        Parse a PDF file from disk.
        
        Args:
            filepath: Path to the PDF file
            
        Returns:
            PDFDocument with extracted content
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"PDF file not found: {filepath}")
        
        logger.info(f"Parsing PDF: {filepath.name}")
        
        doc = fitz.open(filepath)
        return self._process_document(doc, filepath.name, str(filepath))
    
    def parse_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> PDFDocument:
        """
        Parse PDF from bytes.
        
        Args:
            pdf_bytes: PDF content as bytes
            filename: Name for the document
            
        Returns:
            PDFDocument with extracted content
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return self._process_document(doc, filename, "")
    
    def _process_document(
        self, 
        doc: "fitz.Document", 
        filename: str, 
        filepath: str
    ) -> PDFDocument:
        """
        Process a PyMuPDF document object.
        
        Args:
            doc: fitz.Document object
            filename: Document filename
            filepath: Full file path
            
        Returns:
            PDFDocument with extracted content
        """
        pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text
            # Using "text" mode which preserves reading order
            text = page.get_text("text")
            
            # Check if page might be scanned (no text but has images)
            is_scanned = len(text.strip()) < 50 and len(page.get_images()) > 0
            
            # Extract images if page appears scanned (for potential OCR)
            images = []
            if is_scanned:
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    images.append(base_image["image"])
            
            pages.append(PDFPage(
                page_number=page_num + 1,
                text=text,
                images=images,
                is_scanned=is_scanned,
            ))
        
        # Extract metadata
        metadata = doc.metadata or {}
        
        doc.close()
        
        return PDFDocument(
            filename=filename,
            filepath=filepath,
            pages=pages,
            metadata=metadata,
            total_pages=len(pages),
        )


class PaddleOCRParser:
    """
    OCR parser using PaddleOCR.
    
    PaddleOCR provides excellent Arabic text recognition.
    This is a placeholder implementation - uncomment and configure
    when OCR is needed for scanned documents.
    
    Prerequisites:
        pip install paddlepaddle paddleocr
    """
    
    def __init__(self, lang: str = "ar"):
        """
        Initialize PaddleOCR.
        
        Args:
            lang: Language code ('ar' for Arabic)
        """
        # ================================================================
        # PADDLEOCR INITIALIZATION (COMMENTED OUT)
        # Uncomment when ready to use OCR
        # ================================================================
        # if not PADDLEOCR_AVAILABLE:
        #     raise ImportError(
        #         "PaddleOCR is required. Install with: "
        #         "pip install paddlepaddle paddleocr"
        #     )
        # 
        # self.ocr = PaddleOCR(
        #     use_angle_cls=True,  # Enable angle classification
        #     lang=lang,           # Arabic language
        #     use_gpu=False,       # Set True if GPU available
        #     show_log=False,
        # )
        
        self.lang = lang
        logger.warning("PaddleOCR is not configured. OCR will not work.")
    
    def ocr_image(self, image_bytes: bytes) -> str:
        """
        Perform OCR on an image.
        
        Args:
            image_bytes: Image content as bytes
            
        Returns:
            Extracted text from image
        """
        # ================================================================
        # OCR IMPLEMENTATION (COMMENTED OUT)
        # Uncomment when ready to use
        # ================================================================
        # import numpy as np
        # from PIL import Image
        # 
        # # Convert bytes to numpy array
        # image = Image.open(io.BytesIO(image_bytes))
        # image_array = np.array(image)
        # 
        # # Run OCR
        # result = self.ocr.ocr(image_array, cls=True)
        # 
        # # Extract text from results
        # texts = []
        # if result and result[0]:
        #     for line in result[0]:
        #         if line[1]:  # Check if text exists
        #             texts.append(line[1][0])
        # 
        # return "\n".join(texts)
        
        logger.warning("OCR not available - returning empty string")
        return ""
    
    def process_scanned_pdf(self, pdf_doc: PDFDocument) -> PDFDocument:
        """
        Process scanned pages in a PDF document with OCR.
        
        Args:
            pdf_doc: PDFDocument with scanned pages
            
        Returns:
            Updated PDFDocument with OCR text
        """
        for page in pdf_doc.pages:
            if page.is_scanned and page.images:
                ocr_texts = []
                for image_bytes in page.images:
                    text = self.ocr_image(image_bytes)
                    if text:
                        ocr_texts.append(text)
                
                if ocr_texts:
                    page.text = "\n".join(ocr_texts)
        
        return pdf_doc


class PDFProcessor:
    """
    Main PDF processing class.
    
    Combines digital PDF extraction and OCR capabilities.
    Automatically detects whether pages need OCR.
    
    This class is designed to be the primary interface for PDF processing.
    It can be easily extended or have its parsers swapped out.
    """
    
    def __init__(
        self,
        use_ocr: bool = False,
        ocr_lang: str = "ar",
    ):
        """
        Initialize the PDF processor.
        
        Args:
            use_ocr: Enable OCR for scanned documents
            ocr_lang: Language for OCR ('ar' for Arabic)
        """
        self.pdf_parser = PyMuPDFParser() if PYMUPDF_AVAILABLE else None
        self.ocr_parser = None
        
        if use_ocr and PADDLEOCR_AVAILABLE:
            self.ocr_parser = PaddleOCRParser(lang=ocr_lang)
        
        if not self.pdf_parser:
            raise RuntimeError("No PDF parser available. Install PyMuPDF.")
    
    def process_file(self, filepath: Union[str, Path]) -> PDFDocument:
        """
        Process a PDF file.
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            PDFDocument with extracted content
        """
        # Parse the PDF
        doc = self.pdf_parser.parse(filepath)
        
        # Apply OCR to scanned pages if enabled
        if self.ocr_parser and doc.is_scanned:
            logger.info(f"Document contains scanned pages, applying OCR...")
            doc = self.ocr_parser.process_scanned_pdf(doc)
        
        return doc
    
    def process_bytes(
        self, 
        pdf_bytes: bytes, 
        filename: str = "document.pdf"
    ) -> PDFDocument:
        """
        Process PDF from bytes.
        
        Args:
            pdf_bytes: PDF content as bytes
            filename: Name for the document
            
        Returns:
            PDFDocument with extracted content
        """
        doc = self.pdf_parser.parse_bytes(pdf_bytes, filename)
        
        if self.ocr_parser and doc.is_scanned:
            doc = self.ocr_parser.process_scanned_pdf(doc)
        
        return doc
    
    def process_directory(
        self, 
        directory: Union[str, Path],
        pattern: str = "*.pdf",
    ) -> List[PDFDocument]:
        """
        Process all PDFs in a directory.
        
        Args:
            directory: Directory path
            pattern: Glob pattern for PDF files
            
        Returns:
            List of PDFDocument objects
        """
        directory = Path(directory)
        documents = []
        
        pdf_files = list(directory.glob(pattern))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        for pdf_path in pdf_files:
            try:
                doc = self.process_file(pdf_path)
                documents.append(doc)
                logger.info(
                    f"Processed: {pdf_path.name} "
                    f"({doc.total_pages} pages, {len(doc.full_text)} chars)"
                )
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
        
        return documents
    
    def extract_text(self, filepath: Union[str, Path]) -> str:
        """
        Simple method to extract full text from a PDF.
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            Extracted text as a single string
        """
        doc = self.process_file(filepath)
        return doc.full_text


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_text_from_pdf(filepath: Union[str, Path]) -> str:
    """
    Convenience function to extract text from a PDF file.
    
    Args:
        filepath: Path to PDF file
        
    Returns:
        Extracted text
    """
    processor = PDFProcessor()
    return processor.extract_text(filepath)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        processor = PDFProcessor()
        doc = processor.process_file(pdf_path)
        
        print(f"Filename: {doc.filename}")
        print(f"Total Pages: {doc.total_pages}")
        print(f"Is Scanned: {doc.is_scanned}")
        print(f"Text Length: {len(doc.full_text)} characters")
        print("\n--- First 500 characters ---")
        print(doc.full_text[:500])
    else:
        print("Usage: python ocr_processor.py <pdf_file>")
