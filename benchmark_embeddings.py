"""
Benchmark script to compare different embedding models for Arabic retrieval.

Tests:
1. BAAI/bge-m3 (current - multilingual)
2. aubmindlab/bert-base-arabertv2 (AraBERT - Arabic-specific)
3. CAMeL-Lab/bert-base-arabic-camelbert-ca (CAMeLBERT - Arabic-specific)

Usage:
    python benchmark_embeddings.py
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger

# Embedding models to test
EMBEDDING_MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "arabert": "aubmindlab/bert-base-arabertv2", 
    "camelbert": "CAMeL-Lab/bert-base-arabic-camelbert-ca",
}

# Test queries (Arabic legal/regulatory)
TEST_QUERIES = [
    "ما هو الحد الأدنى لرأس المال لشركة تمويل استهلاكي؟",
    "ما هي شروط الترخيص لشركات السمسرة؟",
    "ما الفرق بين التمويل الاستهلاكي والتمويل متناهي الصغر؟",
    "ما هي إجراءات تأسيس شركة توريق؟",
    "ما هي متطلبات الإفصاح للشركات المقيدة؟",
]

# Sample documents (simplified for benchmark)
TEST_DOCS = [
    {
        "id": "doc1",
        "content": "الحد الأدنى لرأس المال المدفوع لشركة التمويل الاستهلاكي هو 50 مليون جنيه مصري",
        "source": "قرار-تمويل-استهلاكي.docx"
    },
    {
        "id": "doc2", 
        "content": "يجب على شركات السمسرة الحصول على ترخيص من الهيئة العامة للرقابة المالية",
        "source": "لائحة-السمسرة.docx"
    },
    {
        "id": "doc3",
        "content": "التمويل متناهي الصغر يختلف عن التمويل الاستهلاكي في الحد الأقصى للقرض",
        "source": "مقارنة-التمويل.docx"
    },
    {
        "id": "doc4",
        "content": "إجراءات تأسيس شركة التوريق تتطلب موافقة مبدئية من الهيئة",
        "source": "دليل-التوريق.docx"
    },
    {
        "id": "doc5",
        "content": "الإفصاح عن القوائم المالية إلزامي للشركات المقيدة في البورصة",
        "source": "قواعد-الإفصاح.docx"
    },
]

# Expected relevance (query_idx -> doc_ids)
GROUND_TRUTH = {
    0: ["doc1"],  # Capital question -> doc1
    1: ["doc2"],  # Brokerage -> doc2
    2: ["doc3"],  # Comparison -> doc3
    3: ["doc4"],  # Securitization -> doc4
    4: ["doc5"],  # Disclosure -> doc5
}


@dataclass
class BenchmarkResult:
    model_name: str
    hit_rate: float
    mrr: float
    avg_latency_ms: float
    embedding_dim: int


def load_model(model_name: str):
    """Load embedding model."""
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading model: {model_name}")
    start = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - start
    logger.info(f"Model loaded in {load_time:.2f}s, dim={model.get_sentence_embedding_dimension()}")
    return model


def compute_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity."""
    # Normalize
    query_norm = query_emb / np.linalg.norm(query_emb)
    doc_norms = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    # Cosine similarity
    return np.dot(doc_norms, query_norm)


def evaluate_model(model_key: str, model_path: str) -> BenchmarkResult:
    """Evaluate a single embedding model."""
    try:
        model = load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load {model_key}: {e}")
        return BenchmarkResult(model_key, 0.0, 0.0, 0.0, 0)
    
    # Encode all documents
    doc_texts = [d["content"] for d in TEST_DOCS]
    doc_ids = [d["id"] for d in TEST_DOCS]
    doc_embs = model.encode(doc_texts, convert_to_numpy=True)
    
    hits = 0
    reciprocal_ranks = []
    latencies = []
    
    for q_idx, query in enumerate(TEST_QUERIES):
        start = time.time()
        query_emb = model.encode([query], convert_to_numpy=True)[0]
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        
        # Compute similarities
        similarities = compute_similarity(query_emb, doc_embs)
        
        # Rank documents
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_ids = [doc_ids[i] for i in ranked_indices]
        
        # Check hit rate @ 1
        expected = GROUND_TRUTH.get(q_idx, [])
        if ranked_ids[0] in expected:
            hits += 1
        
        # Calculate MRR
        for rank, doc_id in enumerate(ranked_ids, 1):
            if doc_id in expected:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
        
        logger.debug(f"Query {q_idx}: Top match = {ranked_ids[0]}, Expected = {expected}")
    
    hit_rate = hits / len(TEST_QUERIES)
    mrr = np.mean(reciprocal_ranks)
    avg_latency = np.mean(latencies)
    
    return BenchmarkResult(
        model_name=model_key,
        hit_rate=hit_rate,
        mrr=mrr,
        avg_latency_ms=avg_latency,
        embedding_dim=model.get_sentence_embedding_dimension()
    )


def main():
    """Run benchmark on all models."""
    print("\n" + "=" * 70)
    print("           ARABIC EMBEDDING MODEL BENCHMARK")
    print("=" * 70)
    
    results = []
    
    for model_key, model_path in EMBEDDING_MODELS.items():
        print(f"\n📊 Testing: {model_key} ({model_path})")
        print("-" * 50)
        
        result = evaluate_model(model_key, model_path)
        results.append(result)
        
        print(f"   Hit Rate @ 1: {result.hit_rate:.1%}")
        print(f"   MRR:          {result.mrr:.3f}")
        print(f"   Avg Latency:  {result.avg_latency_ms:.1f} ms")
        print(f"   Dimensions:   {result.embedding_dim}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("                    SUMMARY")
    print("=" * 70)
    print(f"{'Model':<15} | {'Hit Rate':<10} | {'MRR':<8} | {'Latency':<10} | {'Dim':<6}")
    print("-" * 70)
    
    for r in results:
        print(f"{r.model_name:<15} | {r.hit_rate:>8.1%} | {r.mrr:>6.3f} | {r.avg_latency_ms:>8.1f}ms | {r.embedding_dim:>5}")
    
    # Recommendation
    print("\n" + "=" * 70)
    best = max(results, key=lambda x: x.mrr) if results else None
    if best and best.mrr > 0:
        print(f"✅ RECOMMENDATION: {best.model_name} (MRR: {best.mrr:.3f})")
    print("=" * 70 + "\n")
    
    # Save results
    results_path = Path("data/reports/embedding_benchmark.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump([{
            "model": r.model_name,
            "hit_rate": r.hit_rate,
            "mrr": r.mrr,
            "latency_ms": r.avg_latency_ms,
            "dimensions": r.embedding_dim
        } for r in results], f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
