"""
BGE-M3 Embeddings with Dense and Sparse Vectors.

This module provides BGE-M3 embeddings that support:
1. Dense embeddings (1024 dimensions) - for semantic similarity
2. Sparse embeddings (lexical weights) - for keyword matching

Using both vectors together provides hybrid retrieval benefits.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger


@dataclass
class BGEM3Embedding:
    """Container for BGE-M3 embedding results."""
    dense: np.ndarray  # Dense embedding vector
    sparse: Dict[int, float]  # Sparse embedding (token_id -> weight)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dense": self.dense.tolist(),
            "sparse": self.sparse
        }


class BGEM3Embedder:
    """
    BGE-M3 Embedder with Dense and Sparse support.
    
    Uses FlagEmbedding library for full BGE-M3 functionality.
    Falls back to sentence-transformers dense-only if FlagEmbedding unavailable.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._model = None
        self._fallback_model = None
        self._use_flag_embedding = False
        self._dense_dim = 1024
        
        self._load_model()
        self._initialized = True
    
    def _load_model(self):
        """Load BGE-M3 model with sparse support if available."""
        # Try FlagEmbedding first (full sparse support)
        try:
            from FlagEmbedding import BGEM3FlagModel
            logger.info("Loading BGE-M3 with FlagEmbedding (sparse vectors enabled)")
            self._model = BGEM3FlagModel(
                'BAAI/bge-m3',
                use_fp16=True,
                device='cpu'  # Use CPU to avoid GPU memory issues
            )
            self._use_flag_embedding = True
            logger.info("BGE-M3 loaded with sparse vector support")
            return
        except ImportError:
            logger.warning("FlagEmbedding not available, trying sentence-transformers")
        except Exception as e:
            logger.warning(f"FlagEmbedding failed: {e}, trying sentence-transformers")
        
        # Fallback to sentence-transformers (dense only)
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading BGE-M3 with sentence-transformers (dense only)")
            self._fallback_model = SentenceTransformer('BAAI/bge-m3')
            self._dense_dim = self._fallback_model.get_sentence_embedding_dimension()
            logger.info(f"BGE-M3 loaded (dense only), dim={self._dense_dim}")
        except Exception as e:
            logger.error(f"Failed to load any embedding model: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        """Get dense embedding dimension."""
        return self._dense_dim
    
    @property
    def has_sparse(self) -> bool:
        """Check if sparse embeddings are available."""
        return self._use_flag_embedding
    
    def encode(
        self,
        texts: List[str],
        return_sparse: bool = True,
        batch_size: int = 12,
    ) -> List[BGEM3Embedding]:
        """
        Encode texts to dense and optionally sparse embeddings.
        
        Args:
            texts: List of texts to encode
            return_sparse: Whether to compute sparse embeddings
            batch_size: Batch size for encoding
            
        Returns:
            List of BGEM3Embedding objects
        """
        if not texts:
            return []
        
        if self._use_flag_embedding and self._model:
            return self._encode_with_flag(texts, return_sparse, batch_size)
        else:
            return self._encode_with_st(texts, batch_size)
    
    def _encode_with_flag(
        self,
        texts: List[str],
        return_sparse: bool,
        batch_size: int
    ) -> List[BGEM3Embedding]:
        """Encode with FlagEmbedding (full sparse support)."""
        results = []
        
        # Encode with FlagEmbedding
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,  # Skip ColBERT for efficiency
        )
        
        dense_vecs = output['dense_vecs']
        sparse_vecs = output.get('lexical_weights', [{}] * len(texts)) if return_sparse else [{}] * len(texts)
        
        for i in range(len(texts)):
            dense = dense_vecs[i] if isinstance(dense_vecs, list) else dense_vecs[i]
            sparse = sparse_vecs[i] if i < len(sparse_vecs) else {}
            
            # Convert sparse to dict format
            if hasattr(sparse, 'items'):
                sparse_dict = {int(k): float(v) for k, v in sparse.items()}
            else:
                sparse_dict = {}
            
            results.append(BGEM3Embedding(
                dense=np.array(dense),
                sparse=sparse_dict
            ))
        
        return results
    
    def _encode_with_st(self, texts: List[str], batch_size: int) -> List[BGEM3Embedding]:
        """Encode with sentence-transformers (dense only)."""
        embeddings = self._fallback_model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
        )
        
        return [
            BGEM3Embedding(dense=emb, sparse={})
            for emb in embeddings
        ]
    
    def encode_dense(self, texts: List[str]) -> np.ndarray:
        """Encode texts to dense embeddings only."""
        if self._use_flag_embedding and self._model:
            output = self._model.encode(texts, return_dense=True, return_sparse=False)
            return np.array(output['dense_vecs'])
        else:
            return self._fallback_model.encode(texts, convert_to_numpy=True)
    
    def compute_sparse_scores(
        self,
        query_sparse: Dict[int, float],
        doc_sparse_list: List[Dict[int, float]]
    ) -> List[float]:
        """
        Compute sparse similarity scores between query and documents.
        
        Uses dot product of overlapping token weights.
        """
        scores = []
        for doc_sparse in doc_sparse_list:
            score = 0.0
            for token_id, query_weight in query_sparse.items():
                if token_id in doc_sparse:
                    score += query_weight * doc_sparse[token_id]
            scores.append(score)
        return scores


def get_bge_m3_embedder() -> BGEM3Embedder:
    """Get singleton BGE-M3 embedder instance."""
    return BGEM3Embedder()


# Test
if __name__ == "__main__":
    embedder = get_bge_m3_embedder()
    
    test_texts = [
        "الحد الأدنى لرأس المال لشركة التمويل الاستهلاكي",
        "ما هي شروط الترخيص للشركات المالية؟"
    ]
    
    print(f"Has sparse support: {embedder.has_sparse}")
    print(f"Embedding dimension: {embedder.dimension}")
    
    results = embedder.encode(test_texts, return_sparse=True)
    
    for i, (text, emb) in enumerate(zip(test_texts, results)):
        print(f"\nText {i+1}: {text[:50]}...")
        print(f"  Dense shape: {emb.dense.shape}")
        print(f"  Sparse tokens: {len(emb.sparse)}")
        if emb.sparse:
            top_tokens = sorted(emb.sparse.items(), key=lambda x: -x[1])[:5]
            print(f"  Top sparse: {top_tokens}")
