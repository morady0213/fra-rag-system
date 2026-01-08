"""
Arabic-Aware Text Chunking Module.

Provides text chunking strategies optimized for Arabic legal documents.
Uses RecursiveCharacterTextSplitter with Arabic-specific separators
to avoid breaking legal articles and clauses mid-sentence.

Key Features:
- Arabic-friendly separators (المادة, قرار, etc.)
- Respects legal document structure
- Configurable chunk size and overlap
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from loguru import logger

from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP, ARABIC_SEPARATORS
from ingestion.arabic_utils import normalize_text


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    start_char: int = 0
    end_char: int = 0
    
    @property
    def length(self) -> int:
        """Get the length of the chunk in characters."""
        return len(self.content)


@dataclass
class ChunkedDocument:
    """Represents a document that has been split into chunks."""
    source: str
    chunks: List[TextChunk]
    total_chunks: int
    original_length: int
    
    def get_texts(self) -> List[str]:
        """Get just the text content of all chunks."""
        return [chunk.content for chunk in self.chunks]
    
    def get_metadatas(self) -> List[Dict[str, Any]]:
        """Get metadata for all chunks."""
        return [chunk.metadata for chunk in self.chunks]


class ArabicTextChunker:
    """
    Text chunker optimized for Arabic legal and regulatory documents.
    
    Uses RecursiveCharacterTextSplitter with Arabic-specific separators:
    - Paragraph breaks (\\n\\n)
    - Line breaks (\\n)
    - Legal article markers (المادة - "Article")
    - Decision markers (قرار - "Decision")
    - Clause markers (البند - "Clause")
    - Chapter markers (الفصل - "Chapter")
    - Space and character-level fallback
    
    The order of separators is important - we try to split on larger
    semantic units first, falling back to smaller units as needed.
    """
    
    # Arabic legal document separators
    # Order matters: try larger semantic units first
    DEFAULT_ARABIC_SEPARATORS = [
        "\n\n",      # Double newline (paragraph break) - highest priority
        "\n",        # Single newline
        "المادة",    # "Article" - common in laws
        "قرار",      # "Decision/Decree" - common in regulatory docs
        "البند",     # "Clause" - common in regulations
        "الفصل",    # "Chapter" - document structure
        "الباب",     # "Section/Part" - document structure
        "أولاً",     # "First" - enumeration
        "ثانياً",    # "Second" - enumeration
        "ثالثاً",    # "Third" - enumeration
        ":",         # Colon - often precedes definitions
        ".",         # Period - sentence boundary
        " ",         # Space - word boundary
        "",          # Character-level - last resort
    ]
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: Optional[List[str]] = None,
        normalize_arabic: bool = True,
        length_function: Optional[callable] = None,
    ):
        """
        Initialize the Arabic text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            separators: Custom list of separators (default: Arabic legal separators)
            normalize_arabic: Whether to normalize Arabic text before chunking
            length_function: Custom function to calculate text length
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_ARABIC_SEPARATORS
        self.normalize_arabic = normalize_arabic
        
        # Create the text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=length_function or len,
            keep_separator=True,  # Keep separators in the chunks
            is_separator_regex=False,
        )
        
        logger.info(
            f"ArabicTextChunker initialized: "
            f"size={chunk_size}, overlap={chunk_overlap}, "
            f"separators={len(self.separators)}"
        )
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "unknown",
    ) -> ChunkedDocument:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            metadata: Additional metadata to attach to each chunk
            source: Source identifier (filename, URL, etc.)
            
        Returns:
            ChunkedDocument containing all chunks with metadata
        """
        if not text:
            return ChunkedDocument(
                source=source,
                chunks=[],
                total_chunks=0,
                original_length=0,
            )
        
        original_length = len(text)
        
        # Optionally normalize Arabic text
        if self.normalize_arabic:
            text = normalize_text(text)
        
        # Split the text
        texts = self.splitter.split_text(text)
        
        # Create chunk objects with metadata
        chunks = []
        current_pos = 0
        
        for i, chunk_text in enumerate(texts):
            # Calculate position in original text (approximate)
            start_char = current_pos
            end_char = start_char + len(chunk_text)
            
            # Build chunk metadata
            chunk_metadata = {
                "source": source,
                "chunk_index": i,
                "total_chunks": len(texts),
                "start_char": start_char,
                "end_char": end_char,
            }
            
            # Merge with provided metadata
            if metadata:
                chunk_metadata.update(metadata)
            
            chunks.append(TextChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_index=i,
                start_char=start_char,
                end_char=end_char,
            ))
            
            # Update position (accounting for overlap)
            current_pos = end_char - self.chunk_overlap
        
        logger.debug(
            f"Chunked '{source}': {original_length} chars → {len(chunks)} chunks"
        )
        
        return ChunkedDocument(
            source=source,
            chunks=chunks,
            total_chunks=len(chunks),
            original_length=original_length,
        )
    
    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "text",
        metadata_keys: Optional[List[str]] = None,
    ) -> List[ChunkedDocument]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dictionaries
            text_key: Key for the text field in each document
            metadata_keys: Keys to extract as metadata from each document
            
        Returns:
            List of ChunkedDocument objects
        """
        chunked_docs = []
        
        for doc in documents:
            text = doc.get(text_key, "")
            source = doc.get("source", doc.get("filename", "unknown"))
            
            # Extract metadata
            metadata = {}
            if metadata_keys:
                for key in metadata_keys:
                    if key in doc:
                        metadata[key] = doc[key]
            
            chunked = self.chunk_text(text, metadata, source)
            chunked_docs.append(chunked)
        
        total_chunks = sum(doc.total_chunks for doc in chunked_docs)
        logger.info(
            f"Chunked {len(documents)} documents → {total_chunks} total chunks"
        )
        
        return chunked_docs


def chunk_arabic_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    source: str = "unknown",
) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk Arabic text.
    
    Returns a list of dictionaries suitable for vector store ingestion.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        source: Source identifier
        
    Returns:
        List of dicts with 'text' and 'metadata' keys
    """
    chunker = ArabicTextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunked_doc = chunker.chunk_text(text, source=source)
    
    return [
        {
            "text": chunk.content,
            "metadata": chunk.metadata,
        }
        for chunk in chunked_doc.chunks
    ]


# ============================================================================
# SPECIALIZED CHUNKERS
# ============================================================================

class LegalDocumentChunker(ArabicTextChunker):
    """
    Specialized chunker for Egyptian legal documents.
    
    Adds additional separators common in Egyptian legislation
    and regulatory decisions.
    """
    
    LEGAL_SEPARATORS = [
        "\n\n",
        "\n",
        "المادة",        # Article
        "قرار رقم",      # Decision number
        "قرار",          # Decision
        "البند",         # Clause
        "الفصل",         # Chapter
        "الباب",         # Part/Section
        "مادة",          # Article (variant)
        "بند",           # Clause (variant)
        "أولاً:",        # First:
        "ثانياً:",       # Second:
        "ثالثاً:",       # Third:
        "رابعاً:",       # Fourth:
        "خامساً:",       # Fifth:
        "– ",           # Arabic dash separator
        "- ",           # Latin dash separator
        ":",
        ".",
        "،",            # Arabic comma
        " ",
        "",
    ]
    
    def __init__(
        self,
        chunk_size: int = 1200,  # Larger for legal docs
        chunk_overlap: int = 200,
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.LEGAL_SEPARATORS,
        )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Sample Arabic legal text
    sample_text = """
    المادة الأولى
    تسري أحكام هذا القانون على جميع الأنشطة المالية غير المصرفية في جمهورية مصر العربية.
    
    المادة الثانية
    يُقصد في تطبيق أحكام هذا القانون بالكلمات والعبارات التالية المعني المبين قرين كل منها:
    الهيئة: الهيئة العامة للرقابة المالية.
    الوزير المختص: وزير الاستثمار.
    
    المادة الثالثة
    تختص الهيئة بالرقابة والإشراف على الأسواق والأدوات المالية غير المصرفية.
    
    قرار رقم ١٢٣ لسنة ٢٠٢٤
    بشأن تنظيم أعمال الرقابة على شركات التمويل.
    
    البند الأول: نطاق التطبيق
    تسري أحكام هذا القرار على جميع شركات التمويل المرخص لها.
    """
    
    # Create chunker and process text
    chunker = ArabicTextChunker(chunk_size=300, chunk_overlap=50)
    result = chunker.chunk_text(sample_text, source="sample_legal_doc.pdf")
    
    print(f"Original length: {result.original_length} characters")
    print(f"Total chunks: {result.total_chunks}")
    print("\n" + "=" * 60 + "\n")
    
    for chunk in result.chunks:
        print(f"Chunk {chunk.chunk_index + 1}/{result.total_chunks}")
        print(f"Length: {chunk.length} chars")
        print(f"Content:\n{chunk.content[:200]}...")
        print("-" * 40)
