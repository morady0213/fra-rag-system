"""
Scrapers module for FRA RAG System.
Contains web scraping utilities for fra.gov.eg
"""

from .firecrawl_spider import FirecrawlScraper
from .pdf_spider import run_pdf_spider

__all__ = ["FirecrawlScraper", "run_pdf_spider"]
