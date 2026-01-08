"""
Scrapy Spider for PDF Downloads from FRA Website.

This spider specifically targets the Legislation section of fra.gov.eg
to identify and download PDF files (laws, decrees, regulations).

Features:
- Handles Arabic filenames correctly (UTF-8 encoding)
- Validates PDF links before downloading
- Organizes downloads by category
- Respects rate limits and robots.txt
"""

import os
import re
import hashlib
from pathlib import Path
from urllib.parse import urljoin, unquote, urlparse
from typing import Optional

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.http import Response
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import FRA_BASE_URL, FRA_LEGISLATION_URL, RAW_PDFS_DIR


class FRAPdfSpider(scrapy.Spider):
    """
    Scrapy spider for downloading PDF documents from FRA website.
    
    Targets the Legislation section and downloads:
    - Laws (قوانين)
    - Decrees (قرارات)
    - Regulations (لوائح)
    - Circulars (تعميمات)
    """
    
    name = "fra_pdf_spider"
    allowed_domains = ["fra.gov.eg"]
    
    # Starting URLs - Legislation section and its subsections
    start_urls = [
        f"{FRA_BASE_URL}/ar/legislation",
        f"{FRA_BASE_URL}/ar/legislation/laws",
        f"{FRA_BASE_URL}/ar/legislation/decrees",
        f"{FRA_BASE_URL}/ar/legislation/regulations",
    ]
    
    # Custom settings for the spider
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "CONCURRENT_REQUESTS": 2,  # Be gentle with the server
        "DOWNLOAD_DELAY": 1,  # 1 second delay between requests
        "COOKIES_ENABLED": True,
        "DEFAULT_REQUEST_HEADERS": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ar,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
        },
        # Handle Arabic encoding properly
        "FEED_EXPORT_ENCODING": "utf-8",
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pdf_dir = RAW_PDFS_DIR
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_pdfs = set()  # Track downloaded files to avoid duplicates
        self.pdf_metadata = []  # Store metadata about downloaded PDFs
        
        logger.info(f"PDF Spider initialized. Output dir: {self.pdf_dir}")
    
    def parse(self, response: Response):
        """
        Parse the page and extract PDF links.
        
        Args:
            response: Scrapy response object
        """
        logger.info(f"Parsing: {response.url}")
        
        # Find all PDF links on the page
        # PDFs can be linked via <a href="...pdf"> or data attributes
        pdf_links = response.css('a[href$=".pdf"]::attr(href)').getall()
        pdf_links += response.css('a[href*=".pdf"]::attr(href)').getall()
        
        # Also check for links that might have PDF in the URL but not end with .pdf
        # (some sites use query parameters)
        all_links = response.css('a::attr(href)').getall()
        for link in all_links:
            if link and ('pdf' in link.lower() or 'download' in link.lower()):
                if link not in pdf_links:
                    pdf_links.append(link)
        
        # Deduplicate
        pdf_links = list(set(pdf_links))
        
        logger.info(f"Found {len(pdf_links)} potential PDF links on {response.url}")
        
        # Process each PDF link
        for pdf_url in pdf_links:
            absolute_url = urljoin(response.url, pdf_url)
            
            # Skip if already downloaded
            if absolute_url in self.downloaded_pdfs:
                continue
            
            yield scrapy.Request(
                url=absolute_url,
                callback=self.validate_and_download_pdf,
                meta={
                    "source_page": response.url,
                    "original_link": pdf_url,
                },
                errback=self.handle_error,
            )
        
        # Follow pagination links if any
        next_pages = response.css('a.page-link::attr(href)').getall()
        next_pages += response.css('a[rel="next"]::attr(href)').getall()
        next_pages += response.css('.pagination a::attr(href)').getall()
        
        for next_page in set(next_pages):
            if next_page:
                yield response.follow(next_page, callback=self.parse)
        
        # Follow links to subsections within legislation
        section_links = response.css('a[href*="/legislation/"]::attr(href)').getall()
        for section_link in set(section_links):
            if section_link and section_link not in self.start_urls:
                yield response.follow(section_link, callback=self.parse)
    
    def validate_and_download_pdf(self, response: Response):
        """
        Validate that the response is a PDF and save it.
        
        Args:
            response: Scrapy response object
        """
        content_type = response.headers.get("Content-Type", b"").decode("utf-8", errors="ignore")
        
        # Check if it's actually a PDF
        is_pdf = (
            "application/pdf" in content_type.lower() or
            response.url.lower().endswith(".pdf") or
            response.body[:4] == b"%PDF"
        )
        
        if not is_pdf:
            logger.debug(f"Skipping non-PDF: {response.url} (Content-Type: {content_type})")
            return
        
        # Generate filename
        filename = self._generate_filename(response)
        filepath = self.pdf_dir / filename
        
        # Skip if already exists
        if filepath.exists():
            logger.debug(f"File already exists: {filename}")
            return
        
        # Save the PDF
        try:
            with open(filepath, "wb") as f:
                f.write(response.body)
            
            self.downloaded_pdfs.add(response.url)
            
            # Store metadata
            metadata = {
                "filename": filename,
                "url": response.url,
                "source_page": response.meta.get("source_page", ""),
                "size_bytes": len(response.body),
            }
            self.pdf_metadata.append(metadata)
            
            logger.info(f"Downloaded: {filename} ({len(response.body)} bytes)")
            
            yield metadata
            
        except Exception as e:
            logger.error(f"Error saving PDF {response.url}: {e}")
    
    def _generate_filename(self, response: Response) -> str:
        """
        Generate a safe filename for the PDF, preserving Arabic characters.
        
        Args:
            response: Scrapy response object
            
        Returns:
            Safe filename string
        """
        # Try to get filename from Content-Disposition header
        content_disp = response.headers.get("Content-Disposition", b"").decode("utf-8", errors="ignore")
        
        filename = None
        
        if content_disp:
            # Parse Content-Disposition for filename
            # Handle both filename= and filename*=UTF-8'' formats
            match = re.search(r"filename\*?=(?:UTF-8'')?([^;\n]+)", content_disp, re.IGNORECASE)
            if match:
                filename = unquote(match.group(1).strip('"\''))
        
        if not filename:
            # Extract from URL
            parsed_url = urlparse(response.url)
            filename = unquote(parsed_url.path.split("/")[-1])
        
        if not filename or not filename.endswith(".pdf"):
            # Generate filename from URL hash
            url_hash = hashlib.md5(response.url.encode()).hexdigest()[:8]
            filename = f"document_{url_hash}.pdf"
        
        # Clean filename while preserving Arabic characters
        # Remove/replace characters that are invalid in Windows filenames
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")
        
        # Ensure the filename isn't too long (Windows has 255 char limit)
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:195] + ext
        
        return filename
    
    def handle_error(self, failure):
        """Handle request errors."""
        logger.error(f"Request failed: {failure.request.url} - {failure.value}")
    
    def closed(self, reason):
        """Called when spider is closed."""
        logger.info(f"Spider closed: {reason}")
        logger.info(f"Total PDFs downloaded: {len(self.downloaded_pdfs)}")
        
        # Save metadata to JSON file
        if self.pdf_metadata:
            import json
            metadata_file = self.pdf_dir / "pdf_metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.pdf_metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Metadata saved to: {metadata_file}")


def run_pdf_spider(
    output_dir: Optional[Path] = None,
    start_urls: Optional[list] = None,
) -> None:
    """
    Run the PDF spider.
    
    Args:
        output_dir: Directory to save PDFs (default: RAW_PDFS_DIR)
        start_urls: Custom starting URLs (optional)
    """
    # Configure the crawler process
    process = CrawlerProcess(
        settings={
            "LOG_LEVEL": "INFO",
            "LOG_FORMAT": "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        }
    )
    
    # Configure spider with custom settings if provided
    spider_kwargs = {}
    if output_dir:
        spider_kwargs["pdf_dir"] = output_dir
    if start_urls:
        FRAPdfSpider.start_urls = start_urls
    
    # Start the spider
    process.crawl(FRAPdfSpider, **spider_kwargs)
    process.start()


def main():
    """Main function to run the PDF spider."""
    logger.info("Starting FRA PDF Spider...")
    run_pdf_spider()


if __name__ == "__main__":
    main()
