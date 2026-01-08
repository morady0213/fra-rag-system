"""
Hybrid Retriever with BM25 + Vector Search and Reranking.

Combines:
- BM25 keyword search for exact term matching
- Vector search for semantic similarity
- Cross-encoder reranking for precision
- Response caching for performance
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

import numpy as np
from rank_bm25 import BM25Okapi
import diskcache

# For reranking
from sentence_transformers import CrossEncoder

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR


@dataclass
class RetrievalResult:
    """Single retrieval result with score."""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]
    retrieval_method: str  # 'vector', 'bm25', or 'hybrid'


class ResponseCache:
    """Disk-based cache for embeddings and LLM responses."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or DATA_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate caches for different purposes
        self.embedding_cache = diskcache.Cache(str(self.cache_dir / "embeddings"))
        self.llm_cache = diskcache.Cache(str(self.cache_dir / "llm_responses"))
        self.retrieval_cache = diskcache.Cache(str(self.cache_dir / "retrievals"))
        
        logger.info(f"Cache initialized at {self.cache_dir}")
    
    def _hash_key(self, text: str) -> str:
        """Generate hash key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        key = self._hash_key(text)
        return self.embedding_cache.get(key)
    
    def set_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding."""
        key = self._hash_key(text)
        self.embedding_cache.set(key, embedding)
    
    def get_llm_response(self, query: str, context_hash: str) -> Optional[str]:
        """Get cached LLM response."""
        key = self._hash_key(f"{query}:{context_hash}")
        return self.llm_cache.get(key)
    
    def set_llm_response(self, query: str, context_hash: str, response: str):
        """Cache LLM response."""
        key = self._hash_key(f"{query}:{context_hash}")
        self.llm_cache.set(key, response, expire=3600)  # 1 hour expiry
    
    def get_retrieval(self, query: str, k: int) -> Optional[List[Dict]]:
        """Get cached retrieval results."""
        key = self._hash_key(f"{query}:k{k}")
        return self.retrieval_cache.get(key)
    
    def set_retrieval(self, query: str, k: int, results: List[Dict]):
        """Cache retrieval results."""
        key = self._hash_key(f"{query}:k{k}")
        self.retrieval_cache.set(key, results, expire=1800)  # 30 min expiry
    
    def clear(self):
        """Clear all caches."""
        self.embedding_cache.clear()
        self.llm_cache.clear()
        self.retrieval_cache.clear()
        logger.info("All caches cleared")


class BM25Index:
    """BM25 index for keyword-based retrieval."""
    
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._is_built = False
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the index."""
        for doc in documents:
            content = doc.get("content", doc.get("text", ""))
            self.documents.append(doc)
            # Simple tokenization for Arabic/English
            tokens = self._tokenize(content)
            self.tokenized_docs.append(tokens)
        
        self._is_built = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Simple whitespace tokenization + basic normalization
        import re
        # Remove punctuation and normalize
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.lower().split()
        # Filter very short tokens
        return [t for t in tokens if len(t) > 1]
    
    def build(self):
        """Build the BM25 index."""
        if self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs)
            self._is_built = True
            logger.info(f"BM25 index built with {len(self.documents)} documents")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search the BM25 index."""
        if not self._is_built or not self.bm25:
            self.build()
        
        if not self.bm25:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append((self.documents[idx], float(scores[idx])))
        
        return results
    
    def clear(self):
        """Clear the index."""
        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None
        self._is_built = False


class Reranker:
    """Cross-encoder reranker for improving retrieval precision."""
    
    # Multilingual reranker model
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MULTILINGUAL_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    
    def __init__(self, model_name: Optional[str] = None, use_multilingual: bool = True):
        """Initialize the reranker."""
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.MULTILINGUAL_MODEL if use_multilingual else self.DEFAULT_MODEL
        
        self.model: Optional[CrossEncoder] = None
        self._is_loaded = False
    
    def _load_model(self):
        """Lazy load the reranker model."""
        if not self._is_loaded:
            logger.info(f"Loading reranker model: {self.model_name}")
            try:
                self.model = CrossEncoder(self.model_name)
                self._is_loaded = True
                logger.info("Reranker model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}. Using fallback scoring.")
                self._is_loaded = False
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: User query
            documents: List of document dicts with 'content' key
            top_k: Number of top documents to return
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not documents:
            return []
        
        self._load_model()
        
        if not self.model:
            # Fallback: return documents with original order
            return [(doc, doc.get("score", 0.5)) for doc in documents[:top_k]]
        
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            content = doc.get("content", doc.get("text", ""))[:512]  # Truncate for efficiency
            pairs.append([query, content])
        
        # Get reranking scores
        try:
            scores = self.model.predict(pairs)
            
            # Combine with documents and sort
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return scored_docs[:top_k]
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return [(doc, doc.get("score", 0.5)) for doc in documents[:top_k]]


class HybridRetriever:
    """
    Hybrid retriever combining vector search and BM25.
    
    Features:
    - Vector similarity search (semantic)
    - BM25 keyword search (lexical)
    - Score fusion (RRF - Reciprocal Rank Fusion)
    - Cross-encoder reranking
    - Response caching
    """
    
    def __init__(
        self,
        vector_store,
        use_reranker: bool = True,
        use_cache: bool = True,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: VectorStore instance for semantic search
            use_reranker: Whether to use cross-encoder reranking
            use_cache: Whether to cache results
            vector_weight: Weight for vector search scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
        """
        self.vector_store = vector_store
        self.bm25_index = BM25Index()
        self.reranker = Reranker() if use_reranker else None
        self.cache = ResponseCache() if use_cache else None
        
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        self._bm25_synced = False
        
        logger.info(f"HybridRetriever initialized (reranker={use_reranker}, cache={use_cache})")
    
    def sync_bm25_index(self):
        """Sync BM25 index with vector store documents."""
        if self._bm25_synced:
            return
        
        try:
            # Get all documents from vector store
            all_docs = self.vector_store.get_all_documents()
            if all_docs:
                self.bm25_index.clear()
                self.bm25_index.add_documents(all_docs)
                self.bm25_index.build()
                self._bm25_synced = True
                logger.info(f"BM25 index synced with {len(all_docs)} documents")
        except Exception as e:
            logger.warning(f"Could not sync BM25 index: {e}")
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        use_hybrid: bool = True,
        use_rerank: bool = True,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents using hybrid search.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            use_hybrid: Use hybrid (vector + BM25) or vector only
            use_rerank: Apply reranking
            
        Returns:
            List of RetrievalResult objects
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get_retrieval(query, k)
            if cached:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return [RetrievalResult(**r) for r in cached]
        
        # Get more candidates for reranking
        fetch_k = k * 3 if use_rerank else k
        
        # Vector search
        vector_results = self._vector_search(query, fetch_k)
        
        if use_hybrid:
            # Sync BM25 index if needed
            self.sync_bm25_index()
            
            # BM25 search
            bm25_results = self._bm25_search(query, fetch_k)
            
            # Fuse results
            fused_results = self._fuse_results(vector_results, bm25_results, fetch_k)
        else:
            fused_results = vector_results
        
        # Rerank if enabled
        if use_rerank and self.reranker and fused_results:
            docs = [{"content": r.content, "source": r.source, "metadata": r.metadata} 
                    for r in fused_results]
            reranked = self.reranker.rerank(query, docs, top_k=k)
            
            final_results = [
                RetrievalResult(
                    content=doc["content"],
                    source=doc["source"],
                    score=float(score),
                    metadata=doc["metadata"],
                    retrieval_method="hybrid+rerank"
                )
                for doc, score in reranked
            ]
        else:
            final_results = fused_results[:k]
        
        # Cache results
        if self.cache and final_results:
            cache_data = [
                {
                    "content": r.content,
                    "source": r.source,
                    "score": r.score,
                    "metadata": r.metadata,
                    "retrieval_method": r.retrieval_method
                }
                for r in final_results
            ]
            self.cache.set_retrieval(query, k, cache_data)
        
        return final_results
    
    def _vector_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Perform vector similarity search."""
        try:
            results = self.vector_store.search(query, k=k)
            return [
                RetrievalResult(
                    content=r.get("content", r.get("text", "")),
                    source=r.get("source", "unknown"),
                    score=r.get("score", 0.0),
                    metadata=r.get("metadata", {}),
                    retrieval_method="vector"
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def _bm25_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Perform BM25 keyword search."""
        try:
            results = self.bm25_index.search(query, k=k)
            
            # Normalize BM25 scores to 0-1 range
            if results:
                max_score = max(score for _, score in results) or 1
                return [
                    RetrievalResult(
                        content=doc.get("content", doc.get("text", "")),
                        source=doc.get("source", "unknown"),
                        score=score / max_score,
                        metadata=doc.get("metadata", {}),
                        retrieval_method="bm25"
                    )
                    for doc, score in results
                ]
            return []
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []
    
    def _fuse_results(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        k: int
    ) -> List[RetrievalResult]:
        """
        Fuse vector and BM25 results using Reciprocal Rank Fusion (RRF).
        """
        rrf_k = 60  # RRF constant
        
        # Calculate RRF scores
        doc_scores: Dict[str, float] = {}
        doc_data: Dict[str, RetrievalResult] = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            key = f"{result.source}:{hash(result.content[:100])}"
            rrf_score = self.vector_weight / (rrf_k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0) + rrf_score
            doc_data[key] = result
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            key = f"{result.source}:{hash(result.content[:100])}"
            rrf_score = self.bm25_weight / (rrf_k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0) + rrf_score
            if key not in doc_data:
                doc_data[key] = result
        
        # Sort by fused score
        sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        return [
            RetrievalResult(
                content=doc_data[key].content,
                source=doc_data[key].source,
                score=doc_scores[key],
                metadata=doc_data[key].metadata,
                retrieval_method="hybrid"
            )
            for key in sorted_keys[:k]
        ]
    
    def retrieve_with_context(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Retrieve and format context for LLM.
        
        Returns dict with 'context' string and 'sources' list.
        """
        results = self.retrieve(query, k=k)
        
        if not results:
            return {"context": "", "sources": []}
        
        # Build context string
        context_parts = []
        sources = []
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"[المصدر {i}: {result.source}]\n{result.content}")
            sources.append({
                "source": result.source,
                "score": result.score,
                "content": result.content,
                "method": result.retrieval_method
            })
        
        return {
            "context": "\n\n---\n\n".join(context_parts),
            "sources": sources
        }
