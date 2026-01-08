"""
Vector Store Module using Qdrant.

Provides persistent vector storage with BAAI/bge-m3 embeddings.
BGE-M3 is state-of-the-art for multilingual embeddings, 
including excellent Arabic support.

Features:
- Persistent storage using Qdrant (local or server mode)
- BGE-M3 embeddings via sentence-transformers
- Batch ingestion with progress tracking
- Metadata filtering support
"""

import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from loguru import logger

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, QDRANT_PATH, EMBEDDING_MODEL


@dataclass
class Document:
    """Represents a document to be stored in the vector store."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class EmbeddingModel:
    """
    Wrapper for the embedding model.
    
    Uses sentence-transformers with BAAI/bge-m3 model.
    BGE-M3 is chosen because:
    - State-of-the-art multilingual performance
    - Excellent Arabic language support
    - Dense, sparse, and multi-vector retrieval
    - 8192 token context length
    """
    
    _instance = None
    _model = None
    
    def __new__(cls, model_name: str = EMBEDDING_MODEL):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._model_name = model_name
        return cls._instance
    
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
        return self._model
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


def get_embedding_function(model_name: str = EMBEDDING_MODEL) -> EmbeddingModel:
    """
    Get the embedding model instance.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        EmbeddingModel instance
    """
    return EmbeddingModel(model_name)


class VectorStore:
    """
    Qdrant-based vector store for Arabic documents.
    
    Provides:
    - Document indexing with embeddings
    - Semantic similarity search
    - Metadata filtering
    - Persistent storage (local file or Qdrant server)
    """
    
    DEFAULT_COLLECTION_NAME = "fra_documents"
    
    def __init__(
        self,
        collection_name: str = None,
        embedding_model: str = EMBEDDING_MODEL,
        host: str = None,
        port: int = None,
        path: str = None,
        use_server: bool = False,
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: HuggingFace model name for embeddings
            host: Qdrant server host (if use_server=True)
            port: Qdrant server port (if use_server=True)
            path: Local storage path (if use_server=False)
            use_server: If True, connect to Qdrant server; else use local storage
        """
        self.collection_name = collection_name or QDRANT_COLLECTION
        self.embedding_model_name = embedding_model
        
        # Initialize embedding model
        self.embedding_model = EmbeddingModel(embedding_model)
        
        # Initialize Qdrant client
        if use_server:
            # Connect to Qdrant server
            self.host = host or QDRANT_HOST
            self.port = port or QDRANT_PORT
            logger.info(f"Connecting to Qdrant server at {self.host}:{self.port}")
            self.client = QdrantClient(host=self.host, port=self.port)
            self.persist_path = f"{self.host}:{self.port}"
        else:
            # Use local file storage
            self.persist_path = path or QDRANT_PATH
            Path(self.persist_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing Qdrant with local storage at: {self.persist_path}")
            self.client = QdrantClient(path=self.persist_path)
        
        # Get embedding dimension
        self.vector_size = self.embedding_model.dimension
        
        # Create collection if it doesn't exist
        self._ensure_collection()
        
        doc_count = self._get_document_count()
        logger.info(
            f"Collection '{self.collection_name}' ready. "
            f"Documents: {doc_count}"
        )
    
    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
    
    def _get_document_count(self) -> int:
        """Get the number of documents in the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception:
            return 0
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of dicts with 'text' and optional 'metadata' keys
            batch_size: Number of documents to process at once
            
        Returns:
            Number of documents added
        """
        if not documents:
            logger.warning("No documents to add")
            return 0
        
        total_added = 0
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            texts = [doc.get("text", "") for doc in batch]
            
            # Generate embeddings
            logger.debug(f"Generating embeddings for batch {i // batch_size + 1}...")
            embeddings = self.embedding_model.encode(texts)
            
            # Prepare points
            points = []
            for j, (doc, embedding) in enumerate(zip(batch, embeddings)):
                # Generate unique ID
                doc_id = doc.get("id", str(uuid.uuid4()))
                
                # Prepare payload (metadata + text content)
                metadata = doc.get("metadata", {})
                payload = self._clean_metadata(metadata)
                payload["text"] = doc.get("text", "")
                # Ensure source is at top level of payload for easy retrieval
                if "source" not in payload and "source" in metadata:
                    payload["source"] = metadata["source"]
                
                points.append(PointStruct(
                    id=doc_id if isinstance(doc_id, (int, str)) else str(doc_id),
                    vector=embedding,
                    payload=payload,
                ))
            
            try:
                # Upsert points to collection
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                total_added += len(batch)
                logger.debug(f"Added batch {i // batch_size + 1}: {len(batch)} docs")
                
            except Exception as e:
                logger.error(f"Error adding batch: {e}")
                # Try adding documents one by one
                for point in points:
                    try:
                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=[point],
                        )
                        total_added += 1
                    except Exception as inner_e:
                        logger.error(f"Failed to add doc {point.id}: {inner_e}")
        
        logger.info(f"Added {total_added} documents to collection")
        return total_added
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata for Qdrant compatibility.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Cleaned metadata dictionary
        """
        clean = {}
        
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, (list, tuple)):
                # Convert lists to comma-separated strings
                clean[key] = ", ".join(str(v) for v in value)
            else:
                # Convert other types to string
                clean[key] = str(value)
        
        return clean
    
    def search(
        self,
        query: str,
        k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            k: Number of results to return
            where: Metadata filter conditions (Qdrant filter format)
            where_document: Not used in Qdrant (kept for API compatibility)
            
        Returns:
            List of matching documents with scores
        """
        if not query:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Build filter if provided
            query_filter = None
            if where:
                conditions = []
                for key, value in where.items():
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )
                query_filter = models.Filter(must=conditions)
            
            # Search using query_points (qdrant-client >= 1.7)
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=k,
                query_filter=query_filter,
                with_payload=True,
            )
            
            # Format results
            documents = []
            for result in search_result.points:
                payload = result.payload or {}
                metadata = {k: v for k, v in payload.items() if k != "text"}
                doc = {
                    "id": result.id,
                    "content": payload.get("text", ""),
                    "source": payload.get("source", metadata.get("source", "unknown")),
                    "metadata": metadata,
                    "score": result.score,  # Qdrant returns similarity score directly
                    "distance": 1 - result.score,  # Convert to distance for compatibility
                }
                documents.append(doc)
            
            logger.debug(f"Search returned {len(documents)} results for: {query[:50]}...")
            return documents
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def delete_collection(self) -> None:
        """Delete the entire collection and recreate it."""
        logger.warning(f"Deleting collection: {self.collection_name}")
        
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            logger.warning(f"Error deleting collection: {e}")
        
        # Recreate empty collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "collection_name": self.collection_name,
            "document_count": self._get_document_count(),
            "persist_path": self.persist_path,
            "embedding_model": self.embedding_model_name,
            "vector_size": self.vector_size,
        }
    
    def document_exists(self, doc_id: str) -> bool:
        """Check if a document ID exists in the collection."""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id],
            )
            return len(result) > 0
        except Exception:
            return False
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id],
                with_payload=True,
            )
            
            if results:
                payload = results[0].payload or {}
                return {
                    "id": results[0].id,
                    "content": payload.get("text", ""),
                    "metadata": {k: v for k, v in payload.items() if k != "text"},
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None
    
    def get_all_documents(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """
        Get all documents from the collection for BM25 indexing.
        
        Args:
            limit: Maximum number of documents to retrieve
            
        Returns:
            List of document dicts with content, source, and metadata
        """
        try:
            # Scroll through all documents
            documents = []
            offset = None
            
            while True:
                results, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                )
                
                for point in results:
                    payload = point.payload or {}
                    documents.append({
                        "id": point.id,
                        "content": payload.get("text", ""),
                        "text": payload.get("text", ""),
                        "source": payload.get("source", "unknown"),
                        "metadata": {k: v for k, v in payload.items() if k not in ["text"]},
                    })
                
                if offset is None or len(documents) >= limit:
                    break
            
            logger.info(f"Retrieved {len(documents)} documents for indexing")
            return documents
            
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []


def create_vector_store(
    collection_name: str = None,
    use_server: bool = False,
) -> VectorStore:
    """
    Factory function to create a vector store instance.
    
    Args:
        collection_name: Name of the collection
        use_server: If True, connect to Qdrant server
        
    Returns:
        Configured VectorStore instance
    """
    return VectorStore(
        collection_name=collection_name,
        use_server=use_server,
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    store = VectorStore()
    
    # Sample Arabic documents
    sample_docs = [
        {
            "text": "الهيئة العامة للرقابة المالية هي الجهة المسؤولة عن تنظيم الأسواق المالية غير المصرفية في مصر.",
            "metadata": {"source": "about.md", "section": "introduction"},
        },
        {
            "text": "تختص الهيئة بالإشراف على أنشطة التأمين وإعادة التأمين في جمهورية مصر العربية.",
            "metadata": {"source": "insurance.md", "section": "responsibilities"},
        },
        {
            "text": "يجب على شركات التمويل الحصول على ترخيص من الهيئة قبل مزاولة نشاطها.",
            "metadata": {"source": "licensing.md", "section": "requirements"},
        },
    ]
    
    # Add documents
    added = store.add_documents(sample_docs)
    print(f"Added {added} documents")
    
    # Search
    results = store.search("ما هي مسؤوليات الهيئة العامة للرقابة المالية؟", k=2)
    
    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Content: {result['content'][:100]}...")
        print(f"   Source: {result['metadata'].get('source', 'N/A')}")
    
    # Stats
    print(f"\nStats: {store.get_stats()}")
