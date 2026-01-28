"""
Configuration module for FRA RAG System.
Loads environment variables and provides centralized configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# API KEYS
# ============================================================================
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "fra_documents")
QDRANT_PATH = os.getenv("QDRANT_PATH", str(DATA_DIR / "qdrant_db"))  # Local storage path

# Ensure directories exist
for dir_path in [SAMPLE_DOCS_DIR, RAW_PDFS_DIR, PROCESSED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SCRAPING CONFIGURATION
# ============================================================================
FRA_BASE_URL = "https://fra.gov.eg"
FRA_LEGISLATION_URL = f"{FRA_BASE_URL}/ar/legislation"

# ============================================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# ============================================================================
# LLM CONFIGURATION (Grok / xAI)
# ============================================================================
XAI_API_ENDPOINT = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4-1-fast-non-reasoning")

# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Arabic-specific separators for legal documents
# These separators respect Arabic document structure (articles, decisions)
ARABIC_SEPARATORS = [
    "\n\n",      # Double newline (paragraph break)
    "\n",        # Single newline
    "المادة",    # "Article" - common in legal texts
    "قرار",      # "Decision/Decree" - common in regulatory documents
    "البند",     # "Clause"
    "الفصل",    # "Chapter"
    " ",         # Space
    "",          # Character-level fallback
]

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================
DEFAULT_TOP_K = 3  # Number of documents to retrieve (reduced for better grounding)

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
