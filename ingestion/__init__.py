"""
Ingestion module for FRA RAG System.
Contains text processing, OCR, and chunking utilities.
"""

from .arabic_utils import normalize_text, ArabicTextNormalizer
from .ocr_processor import PDFProcessor
from .chunking import ArabicTextChunker

__all__ = [
    "normalize_text",
    "ArabicTextNormalizer", 
    "PDFProcessor",
    "ArabicTextChunker",
]
