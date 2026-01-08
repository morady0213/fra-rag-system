"""
Firecrawl Spider for FRA Website.

Uses Firecrawl API to:
1. Map the website structure (sitemap discovery)
2. Extract clean markdown content from informational pages

Firecrawl handles JavaScript rendering and provides clean markdown output,
which is ideal for Arabic content extraction.
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger

try:
    from firecrawl import FirecrawlApp
except ImportError:
    logger.warning("firecrawl-py not installed. Run: pip install firecrawl-py")
    FirecrawlApp = None

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import FIRECRAWL_API_KEY, FRA_BASE_URL, PROCESSED_DIR


class FirecrawlScraper:
    """
    Scraper using Firecrawl API for extracting content from fra.gov.eg.
    
    Firecrawl is particularly useful for:
    - Handling JavaScript-rendered content
    - Extracting clean markdown from complex pages
    - Respecting robots.txt and rate limits
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Firecrawl scraper.
        
        Args:
            api_key: Firecrawl API key. If not provided, reads from environment.
        """
        self.api_key = api_key or FIRECRAWL_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "Firecrawl API key not found. "
                "Set FIRECRAWL_API_KEY environment variable or pass api_key parameter."
            )
        
        if FirecrawlApp is None:
            raise ImportError("firecrawl-py package not installed")
            
        self.app = FirecrawlApp(api_key=self.api_key)
        self.output_dir = PROCESSED_DIR / "markdown"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FirecrawlScraper initialized. Output dir: {self.output_dir}")
    
    def map_website(self, url: str = FRA_BASE_URL) -> List[str]:
        """
        Map the website to discover all accessible URLs.
        
        Args:
            url: Starting URL for mapping (default: FRA base URL)
            
        Returns:
            List of discovered URLs
        """
        logger.info(f"Mapping website structure starting from: {url}")
        
        try:
            # Use Firecrawl's map feature to discover URLs
            map_result = self.app.map_url(url)
            
            if isinstance(map_result, dict) and "links" in map_result:
                urls = map_result["links"]
            elif isinstance(map_result, list):
                urls = map_result
            else:
                urls = []
            
            logger.info(f"Discovered {len(urls)} URLs")
            
            # Save the sitemap for reference
            sitemap_path = self.output_dir / "sitemap.json"
            with open(sitemap_path, "w", encoding="utf-8") as f:
                json.dump({"base_url": url, "urls": urls}, f, ensure_ascii=False, indent=2)
            
            return urls
            
        except Exception as e:
            logger.error(f"Error mapping website: {e}")
            return []
    
    def scrape_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape a single page and extract markdown content.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with page metadata and markdown content
        """
        logger.info(f"Scraping page: {url}")
        
        try:
            # Scrape with markdown output format
            result = self.app.scrape_url(
                url,
                params={
                    "formats": ["markdown", "html"],
                    "onlyMainContent": True,  # Extract main content only
                }
            )
            
            return {
                "url": url,
                "markdown": result.get("markdown", ""),
                "html": result.get("html", ""),
                "metadata": result.get("metadata", {}),
            }
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def crawl_website(
        self,
        url: str = FRA_BASE_URL,
        max_pages: int = 100,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Crawl the website and extract content from multiple pages.
        
        Args:
            url: Starting URL for crawling
            max_pages: Maximum number of pages to crawl
            include_patterns: URL patterns to include (regex)
            exclude_patterns: URL patterns to exclude (regex)
            
        Returns:
            List of scraped page data
        """
        logger.info(f"Starting crawl from {url} (max {max_pages} pages)")
        
        # Default patterns for FRA website
        if include_patterns is None:
            include_patterns = [
                r".*fra\.gov\.eg/ar/.*",  # Arabic pages
            ]
        
        if exclude_patterns is None:
            exclude_patterns = [
                r".*\.(pdf|jpg|png|gif|zip)$",  # Skip binary files
                r".*/login.*",  # Skip login pages
                r".*/search.*",  # Skip search pages
            ]
        
        try:
            # Use Firecrawl's crawl feature
            crawl_result = self.app.crawl_url(
                url,
                params={
                    "limit": max_pages,
                    "scrapeOptions": {
                        "formats": ["markdown"],
                        "onlyMainContent": True,
                    },
                    "includePaths": include_patterns,
                    "excludePaths": exclude_patterns,
                },
                poll_interval=5,  # Check status every 5 seconds
            )
            
            # Process results
            pages = []
            if isinstance(crawl_result, dict) and "data" in crawl_result:
                for item in crawl_result["data"]:
                    page_data = {
                        "url": item.get("metadata", {}).get("sourceURL", ""),
                        "markdown": item.get("markdown", ""),
                        "title": item.get("metadata", {}).get("title", ""),
                    }
                    pages.append(page_data)
                    
                    # Save each page as markdown file
                    self._save_page_markdown(page_data)
            
            logger.info(f"Crawl complete. Extracted {len(pages)} pages")
            return pages
            
        except Exception as e:
            logger.error(f"Error during crawl: {e}")
            return []
    
    def _save_page_markdown(self, page_data: Dict[str, Any]) -> None:
        """
        Save extracted page content as a markdown file.
        
        Args:
            page_data: Dictionary containing url, markdown, and title
        """
        url = page_data.get("url", "")
        markdown = page_data.get("markdown", "")
        title = page_data.get("title", "untitled")
        
        if not markdown:
            return
        
        # Create a safe filename from the URL
        # Handle Arabic characters in URLs properly
        safe_name = url.replace("https://", "").replace("http://", "")
        safe_name = safe_name.replace("/", "_").replace("?", "_").replace("&", "_")
        safe_name = safe_name[:100]  # Limit filename length
        
        if not safe_name.endswith(".md"):
            safe_name += ".md"
        
        filepath = self.output_dir / safe_name
        
        # Add metadata header to markdown
        content = f"""---
title: {title}
source_url: {url}
---

{markdown}
"""
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.debug(f"Saved: {filepath}")
    
    def scrape_specific_sections(self) -> Dict[str, List[Dict]]:
        """
        Scrape specific important sections of the FRA website.
        
        Returns:
            Dictionary mapping section names to their scraped content
        """
        # Key sections of the FRA website
        sections = {
            "about": f"{FRA_BASE_URL}/ar/about",
            "services": f"{FRA_BASE_URL}/ar/services",
            "legislation": f"{FRA_BASE_URL}/ar/legislation",
            "news": f"{FRA_BASE_URL}/ar/news",
            "non_banking": f"{FRA_BASE_URL}/ar/non-banking-financial-activities",
            "capital_market": f"{FRA_BASE_URL}/ar/capital-market",
            "insurance": f"{FRA_BASE_URL}/ar/insurance",
        }
        
        results = {}
        
        for section_name, section_url in sections.items():
            logger.info(f"Scraping section: {section_name}")
            page_data = self.scrape_page(section_url)
            
            if page_data:
                results[section_name] = [page_data]
                self._save_page_markdown({
                    "url": section_url,
                    "markdown": page_data.get("markdown", ""),
                    "title": section_name,
                })
        
        return results


def main():
    """Main function to demonstrate Firecrawl scraper usage."""
    try:
        scraper = FirecrawlScraper()
        
        # Option 1: Map the website structure
        # urls = scraper.map_website()
        # print(f"Found {len(urls)} URLs")
        
        # Option 2: Crawl and extract content
        # pages = scraper.crawl_website(max_pages=50)
        
        # Option 3: Scrape specific sections
        results = scraper.scrape_specific_sections()
        
        for section, pages in results.items():
            print(f"\n{section}: {len(pages)} pages scraped")
            
    except Exception as e:
        logger.error(f"Scraper failed: {e}")
        raise


if __name__ == "__main__":
    main()
