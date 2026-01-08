# 🏛️ FRA RAG System

**Retrieval-Augmented Generation System for the Egyptian Financial Regulatory Authority (FRA)**

نظام الاسترجاع المعزز بالتوليد للهيئة العامة للرقابة المالية المصرية

---

## Overview

A complete, modular Python RAG system designed specifically for Arabic legal and regulatory documents from [fra.gov.eg](https://fra.gov.eg). The system combines web scraping, document processing, semantic search, and LLM-powered answer generation.

### Key Features

- **Arabic-Optimized**: Strict Arabic text normalization (Alef unification, diacritics removal, etc.)
- **Hybrid Scraping**: Firecrawl for web content + Scrapy for PDF downloads
- **SOTA Embeddings**: BAAI/bge-m3 for multilingual semantic search
- **Legal Document Aware**: Arabic-specific chunking separators (المادة, قرار, etc.)
- **Grok Integration**: xAI's Grok for context-grounded answer generation

---

## Project Structure

```
fra-rag-system/
├── config.py                 # Centralized configuration
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
│
├── scrapers/                 # Web scraping module
│   ├── firecrawl_spider.py   # Firecrawl API for web content
│   └── pdf_spider.py         # Scrapy spider for PDF downloads
│
├── ingestion/                # Document processing module
│   ├── arabic_utils.py       # Arabic text normalization
│   ├── ocr_processor.py      # PDF parsing (PyMuPDF + PaddleOCR stub)
│   └── chunking.py           # Arabic-aware text chunking
│
├── rag_engine/               # Retrieval module
│   ├── vector_store.py       # ChromaDB with BGE-M3 embeddings
│   └── retriever.py          # High-level retrieval interface
│
├── llm_client/               # LLM module
│   └── grok_client.py        # xAI Grok API client
│
└── data/                     # Data directories
    ├── sample_docs/          # Place documents here for ingestion
    ├── raw_pdfs/             # Downloaded PDFs from scraper
    ├── processed/            # Processed markdown files
    └── chroma_db/            # Vector database (auto-created)
```

---

## Installation

### 1. Clone/Create the project

```bash
cd fra-rag-system
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys
# XAI_API_KEY=your_xai_api_key
# FIRECRAWL_API_KEY=your_firecrawl_api_key (optional)
```

---

## Usage

### Quick Start

```bash
# 1. Place documents in data/sample_docs/
# 2. Run the system
python main.py
```

### Command Line Options

```bash
# Interactive mode (default)
python main.py

# Force re-index documents
python main.py --ingest

# Single query mode
python main.py --query "ما هي اختصاصات الهيئة العامة للرقابة المالية؟"

# Show statistics
python main.py --stats

# Custom number of retrieved documents
python main.py --query "سؤالك" --k 10
```

### Programmatic Usage

```python
from main import FRARAGSystem

# Initialize the system
system = FRARAGSystem()

# Ingest documents (if needed)
if not system.is_indexed():
    system.ingest_documents()

# Query the system
answer = system.query("ما هي إجراءات الترخيص؟")
print(answer)
```

---

## Module Documentation

### 1. Scrapers Module (`scrapers/`)

#### Firecrawl Spider

```python
from scrapers import FirecrawlScraper

scraper = FirecrawlScraper()

# Map website structure
urls = scraper.map_website()

# Crawl and extract content
pages = scraper.crawl_website(max_pages=50)

# Scrape specific sections
results = scraper.scrape_specific_sections()
```

#### PDF Spider (Scrapy)

```python
from scrapers import run_pdf_spider

# Run the spider to download PDFs
run_pdf_spider()
```

Or via command line:
```bash
cd scrapers
scrapy runspider pdf_spider.py
```

### 2. Ingestion Module (`ingestion/`)

#### Arabic Text Normalization

```python
from ingestion import normalize_text

text = "القَرَارُ رَقْم ١٢٣"
normalized = normalize_text(text)
# Output: "القرار رقم ١٢٣"
```

**Normalization rules:**
- Alef unification: `أ, إ, آ, ٱ → ا`
- Yeh unification: `ى → ي`
- Tatweel removal: `ـ` removed
- Diacritics removal: All tashkeel removed
- Whitespace normalization

#### PDF Processing

```python
from ingestion import PDFProcessor

processor = PDFProcessor()

# Process a single PDF
doc = processor.process_file("document.pdf")
print(doc.full_text)

# Process all PDFs in a directory
docs = processor.process_directory("data/raw_pdfs/")
```

#### Text Chunking

```python
from ingestion import ArabicTextChunker

chunker = ArabicTextChunker(
    chunk_size=1000,
    chunk_overlap=200,
)

result = chunker.chunk_text(text, source="document.pdf")
for chunk in result.chunks:
    print(chunk.content)
```

**Arabic-aware separators:**
- `\n\n` - Paragraph breaks
- `\n` - Line breaks
- `المادة` - Article
- `قرار` - Decision
- `البند` - Clause
- `الفصل` - Chapter

### 3. RAG Engine (`rag_engine/`)

#### Vector Store

```python
from rag_engine import VectorStore

store = VectorStore()

# Add documents
docs = [
    {"text": "محتوى الوثيقة...", "metadata": {"source": "doc.pdf"}},
]
store.add_documents(docs)

# Search
results = store.search("استعلام البحث", k=5)
```

#### Retriever

```python
from rag_engine import Retriever

retriever = Retriever()

# Retrieve relevant documents
response = retriever.retrieve("ما هي اختصاصات الهيئة؟", k=5)

for result in response.results:
    print(f"Source: {result.source}")
    print(f"Score: {result.score}")
    print(f"Content: {result.content[:200]}...")
```

### 4. LLM Client (`llm_client/`)

```python
from llm_client import GrokClient

client = GrokClient()

result = client.generate(
    query="ما هي إجراءات الترخيص؟",
    context="السياق من الوثائق...",
    sources=["doc1.pdf", "doc2.pdf"],
)

print(result.answer)
```

---

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `XAI_API_KEY` | xAI (Grok) API key | Yes |
| `FIRECRAWL_API_KEY` | Firecrawl API key | For scraping |
| `GROK_MODEL` | Model name (default: grok-beta) | No |
| `CHROMA_PERSIST_DIR` | Vector DB path | No |
| `EMBEDDING_MODEL` | Embedding model | No |

### Configuration File (`config.py`)

Key settings:
- `CHUNK_SIZE`: 1000 characters
- `CHUNK_OVERLAP`: 200 characters
- `DEFAULT_TOP_K`: 5 documents
- `ARABIC_SEPARATORS`: Legal document separators

---

## Arabic NLP Details

### Why Arabic Normalization Matters

Arabic text has many variations that should be treated as equivalent:
- **Alef variants**: أ, إ, آ are all normalized to ا
- **Yeh/Alef Maksura**: ى is normalized to ي
- **Diacritics**: Short vowel marks are removed for matching
- **Tatweel**: Decorative elongation character is removed

### BGE-M3 for Arabic

We use `BAAI/bge-m3` because:
- State-of-the-art multilingual embeddings
- Excellent Arabic language support
- Supports up to 8192 tokens
- Dense, sparse, and multi-vector retrieval

---

## OCR Support (Optional)

For scanned PDFs, uncomment the PaddleOCR integration in `ocr_processor.py`:

```bash
# Install PaddleOCR
pip install paddlepaddle paddleocr
```

Then update `ocr_processor.py` to enable OCR.

---

## Troubleshooting

### "No documents found"

Place documents in `data/sample_docs/` or PDFs in `data/raw_pdfs/`.

### "LLM not available"

Set the `XAI_API_KEY` environment variable.

### Slow embedding generation

First run downloads the BGE-M3 model (~2GB). Subsequent runs use cached model.

### Memory issues

Reduce `CHUNK_SIZE` or process fewer documents at once.

---

## License

MIT License

---

## Acknowledgments

- [Firecrawl](https://firecrawl.dev/) - Web scraping API
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) - Embedding model
- [xAI Grok](https://x.ai/) - LLM API
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing
