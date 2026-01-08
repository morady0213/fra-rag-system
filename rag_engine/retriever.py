"""
Retriever Module for RAG System.

Provides high-level retrieval interface that wraps the vector store
and adds additional features like query preprocessing and result formatting.

Features:
- Query normalization for Arabic
- Configurable retrieval parameters
- Result formatting for LLM consumption
- Source attribution
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DEFAULT_TOP_K
from rag_engine.vector_store import VectorStore
from ingestion.arabic_utils import normalize_text


@dataclass
class RetrievalResult:
    """Represents a single retrieval result."""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]
    chunk_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
        }


@dataclass
class RetrievalResponse:
    """Represents the complete retrieval response."""
    query: str
    results: List[RetrievalResult]
    total_results: int
    
    @property
    def context(self) -> str:
        """Get concatenated context from all results."""
        contexts = []
        for i, result in enumerate(self.results, 1):
            source_info = f"[المصدر {i}: {result.source}]"
            contexts.append(f"{source_info}\n{result.content}")
        return "\n\n---\n\n".join(contexts)
    
    @property
    def sources(self) -> List[str]:
        """Get list of unique sources."""
        return list(set(r.source for r in self.results))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "sources": self.sources,
        }


class Retriever:
    """
    High-level retriever for Arabic documents.
    
    Wraps VectorStore with additional functionality:
    - Arabic query normalization
    - Score thresholding
    - Result formatting for LLM prompts
    - Source deduplication
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        normalize_queries: bool = True,
        score_threshold: float = 0.0,
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store: VectorStore instance (creates new if not provided)
            normalize_queries: Whether to normalize Arabic queries
            score_threshold: Minimum score for results (0.0 to 1.0)
        """
        self.vector_store = vector_store or VectorStore()
        self.normalize_queries = normalize_queries
        self.score_threshold = score_threshold
        
        logger.info(
            f"Retriever initialized. "
            f"Normalize: {normalize_queries}, Threshold: {score_threshold}"
        )
    
    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        where: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResponse:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query (Arabic or English)
            k: Number of documents to retrieve
            where: Optional metadata filter
            
        Returns:
            RetrievalResponse with ranked results
        """
        if not query:
            return RetrievalResponse(query="", results=[], total_results=0)
        
        # Normalize query for better Arabic matching
        processed_query = query
        if self.normalize_queries:
            processed_query = normalize_text(query)
        
        logger.info(f"Retrieving for: {processed_query[:100]}...")
        
        # Search vector store
        raw_results = self.vector_store.search(
            query=processed_query,
            k=k,
            where=where,
        )
        
        # Process and filter results
        results = []
        for raw in raw_results:
            score = raw.get("score", 0)
            
            # Apply score threshold
            if score < self.score_threshold:
                continue
            
            # Extract source from metadata
            metadata = raw.get("metadata", {})
            source = metadata.get("source", metadata.get("filename", "unknown"))
            
            result = RetrievalResult(
                content=raw.get("content", ""),
                source=source,
                score=score,
                metadata=metadata,
                chunk_index=metadata.get("chunk_index", 0),
            )
            results.append(result)
        
        logger.info(f"Retrieved {len(results)} results (threshold: {self.score_threshold})")
        
        return RetrievalResponse(
            query=query,
            results=results,
            total_results=len(results),
        )
    
    def retrieve_with_context(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        max_context_length: int = 4000,
    ) -> Dict[str, Any]:
        """
        Retrieve documents and format as context for LLM.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            max_context_length: Maximum characters for context
            
        Returns:
            Dictionary with 'context' and 'sources' keys
        """
        response = self.retrieve(query, k=k)
        
        # Build context string, respecting max length
        context_parts = []
        current_length = 0
        included_sources = []
        
        for i, result in enumerate(response.results, 1):
            # Format this result
            source_header = f"[المصدر {i}: {result.source}]"
            part = f"{source_header}\n{result.content}"
            
            # Check if adding this would exceed limit
            if current_length + len(part) + 10 > max_context_length:
                break
            
            context_parts.append(part)
            current_length += len(part) + 10  # Account for separators
            included_sources.append({
                "source": result.source,
                "score": result.score,
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        return {
            "context": context,
            "sources": included_sources,
            "query": query,
            "total_retrieved": len(response.results),
            "total_used": len(context_parts),
        }
    
    def format_for_prompt(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
    ) -> str:
        """
        Retrieve and format results specifically for LLM prompts.
        
        Returns formatted Arabic context with clear source attribution.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            Formatted context string in Arabic
        """
        response = self.retrieve(query, k=k)
        
        if not response.results:
            return "لم يتم العثور على معلومات ذات صلة."  # No relevant information found
        
        # Build formatted context
        lines = ["السياق من الوثائق الرسمية:"]  # Context from official documents:
        lines.append("=" * 50)
        
        for i, result in enumerate(response.results, 1):
            lines.append(f"\n📄 المصدر {i}: {result.source}")
            lines.append(f"درجة الصلة: {result.score:.2%}")
            lines.append("-" * 30)
            lines.append(result.content)
            lines.append("")
        
        return "\n".join(lines)


def retrieve(
    query: str,
    k: int = DEFAULT_TOP_K,
    vector_store: Optional[VectorStore] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to retrieve documents.
    
    Args:
        query: Search query
        k: Number of results
        vector_store: Optional VectorStore instance
        
    Returns:
        List of result dictionaries
    """
    retriever = Retriever(vector_store=vector_store)
    response = retriever.retrieve(query, k=k)
    return [r.to_dict() for r in response.results]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    retriever = Retriever()
    
    # Sample query in Arabic
    query = "ما هي إجراءات الحصول على ترخيص من الهيئة العامة للرقابة المالية؟"
    
    # Retrieve documents
    response = retriever.retrieve(query, k=3)
    
    print(f"Query: {response.query}")
    print(f"Total Results: {response.total_results}")
    print(f"Sources: {response.sources}")
    print("\n" + "=" * 60 + "\n")
    
    for i, result in enumerate(response.results, 1):
        print(f"Result {i}:")
        print(f"  Score: {result.score:.4f}")
        print(f"  Source: {result.source}")
        print(f"  Content: {result.content[:200]}...")
        print()
    
    # Get formatted context for LLM
    print("\n" + "=" * 60)
    print("Formatted for LLM Prompt:")
    print("=" * 60 + "\n")
    print(retriever.format_for_prompt(query))
