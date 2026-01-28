# FRA RAG System - Technical Report for LLM Architect

**Date:** January 25, 2026  
**Purpose:** Detailed analysis of current system performance issues and recommendations for improvement  
**Domain:** Arabic Legal/Regulatory RAG System for Egyptian Financial Regulatory Authority (FRA)

---

## Executive Summary

The FRA RAG system currently achieves an **overall score of 52.93%** on a 15-question evaluation dataset. Key issues include:

| Metric | Current Score | Target | Gap |
|--------|---------------|--------|-----|
| **Faithfulness** | 40.33% | 85%+ | -44.67% |
| **Hit Rate @ 5** | 66.67% | 90%+ | -23.33% |
| **Answer Correctness** | 32.26% | 80%+ | -47.74% |
| **Anti-Hallucination** | 69.33% | 95%+ | -25.67% |

The system struggles primarily with **faithfulness** (LLM adds information not in context) and **retrieval precision** (relevant documents not always in top 5).

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRA RAG SYSTEM                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │  Query Input │───▶│ Query Router │───▶│ Complexity Detection │   │
│  └──────────────┘    └──────────────┘    └──────────────────────┘   │
│                              │                                        │
│         ┌────────────────────┼────────────────────┐                  │
│         ▼                    ▼                    ▼                  │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐          │
│  │   Simple    │      │ Comparison  │      │  Multi-Hop  │          │
│  │  Retrieval  │      │Decomposition│      │ ReAct Agent │          │
│  └─────────────┘      └─────────────┘      └─────────────┘          │
│         │                    │                    │                  │
│         └────────────────────┼────────────────────┘                  │
│                              ▼                                        │
│                    ┌──────────────────┐                              │
│                    │ Hybrid Retriever │                              │
│                    │  (Vector + BM25) │                              │
│                    └──────────────────┘                              │
│                              │                                        │
│                              ▼                                        │
│                    ┌──────────────────┐                              │
│                    │    Cross-Encoder │                              │
│                    │     Reranker     │                              │
│                    └──────────────────┘                              │
│                              │                                        │
│                              ▼                                        │
│                    ┌──────────────────┐                              │
│                    │   Grok LLM       │                              │
│                    │   Generation     │                              │
│                    └──────────────────┘                              │
│                              │                                        │
│                              ▼                                        │
│                    ┌──────────────────┐                              │
│                    │  Answer + Cites  │                              │
│                    └──────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Details

### 2.1 Document Ingestion Pipeline

| Component | Technology | Configuration |
|-----------|------------|---------------|
| **PDF Processing** | PyMuPDF + pytesseract (OCR) | Arabic OCR support |
| **DOCX Processing** | python-docx | Direct text extraction |
| **Text Chunking** | LangChain RecursiveCharacterTextSplitter | 1000 chars, 200 overlap |
| **Arabic Separators** | Custom | `المادة`, `قرار`, `البند`, `الفصل` |
| **Metadata Extraction** | Regex-based | Entity types, topics, regulatory flags |

**Current Issues:**
- Chunks may break mid-article despite Arabic separators
- No hierarchical document structure preservation
- Metadata extraction is regex-only (no LLM enhancement)

### 2.2 Embedding Model

| Parameter | Value |
|-----------|-------|
| **Model** | `BAAI/bge-m3` |
| **Dimension** | 1024 |
| **Context Length** | 8192 tokens |
| **Type** | Dense embeddings |

**Current Issues:**
- Using only dense embeddings, not utilizing BGE-M3's sparse/multi-vector capabilities
- No fine-tuning on Arabic legal domain

### 2.3 Vector Store

| Parameter | Value |
|-----------|-------|
| **Database** | Qdrant (local mode) |
| **Collection** | `fra_documents` |
| **Document Count** | 112 chunks |
| **Distance Metric** | Cosine similarity |

**Current Issues:**
- Small corpus (112 chunks from ~50 DOCX files)
- No metadata indexing for filtered search

### 2.4 Hybrid Retrieval

| Component | Technology | Configuration |
|-----------|------------|---------------|
| **Vector Search** | Qdrant + BGE-M3 | Top-K = 5 |
| **Keyword Search** | BM25Okapi | Whitespace tokenization |
| **Score Fusion** | Linear combination | Vector weight: 0.5, BM25 weight: 0.5 |
| **Reranker** | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Multilingual cross-encoder |

**Current Issues:**
- BM25 tokenization is naive (whitespace-only, no Arabic morphology)
- No Arabic stemming or lemmatization
- Reranker not specifically trained on Arabic legal text
- Fixed 50/50 fusion weights (not tuned)

### 2.5 Query Router

| Query Type | Detection Method | Strategy |
|------------|------------------|----------|
| **Simple** | No comparison/procedural patterns | Standard retrieval |
| **Comparison** | Patterns: `الفرق بين`, `قارن بين`, etc. | Decomposition into sub-queries |
| **Procedural** | Patterns: `خطوات`, `إجراءات`, etc. | Standard retrieval |
| **Multi-part** | Patterns: `وكذلك`, `بالإضافة إلى`, etc. | Decomposition |

**Current Issues:**
- Pattern-based detection is fragile
- No LLM-based query understanding
- Decomposition may miss nuances

### 2.6 LLM Generation

| Parameter | Value |
|-----------|-------|
| **Model** | `grok-4-1-fast-non-reasoning` |
| **Provider** | xAI |
| **Max Tokens** | 4096 |
| **Temperature** | 0.3 |

**System Prompt Summary (Arabic):**
1. Answer ONLY based on provided context
2. Include citations with article numbers and direct quotes
3. If no answer in context, explicitly state "لا يوجد نص صريح"
4. Anti-hallucination rule: Never add info not literally in context
5. Response format: Summary → Details → Sources

**Current Issues:**
- Model still hallucinates despite strict prompting
- Long responses dilute faithfulness score
- Model adds interpretations not present in source

### 2.7 ReAct Agent (Optional)

| Parameter | Value |
|-----------|-------|
| **Max Iterations** | 5 |
| **Actions** | RETRIEVE, CALCULATE, COMPARE, ANSWER |
| **Language** | Arabic |

**Current Issues:**
- Not enabled by default
- May increase latency significantly
- Limited action vocabulary

---

## 3. Evaluation Results (Detailed)

### 3.1 Overall Metrics

| Category | Metric | Score | Interpretation |
|----------|--------|-------|----------------|
| **Retrieval** | Hit Rate @ 5 | 66.67% | Found relevant doc in top 5 for 67% of queries |
| | MRR | 54.44% | First relevant doc appears around position 2 |
| | Precision @ 5 | 26.67% | Only 1.3 of 5 docs are relevant |
| | Recall @ 5 | 86.67% | Good - retrieves 87% of relevant docs |
| | NDCG @ 5 | 55.85% | Moderate ranking quality |
| **Generation** | Faithfulness | 40.33% | **CRITICAL** - 60% of claims not grounded |
| | Answer Relevancy | 92.89% | Good - answers address questions |
| | Context Utilization | 100.00% | All context is used |
| | Answer Correctness | 32.26% | Low - differs from ground truth |
| **Arabic** | Citation Accuracy | 50.00% | Half of citations match sources |
| | Anti-Hallucination | 69.33% | **NEEDS IMPROVEMENT** |
| | Number Accuracy | 90.00% | Good - numeric values correct |
| **Performance** | Avg Latency | 6,648 ms | ~6.6 seconds per query |
| | P95 Latency | 15,843 ms | ~16 seconds worst case |

### 3.2 Performance by Question Type

| Type | Count | Faithfulness | Relevancy | Issue |
|------|-------|--------------|-----------|-------|
| **factual** | 5 | 49.08% | 92.29% | Model adds extra context |
| **comparison** | 2 | 55.19% | 95.00% | Best performing |
| **conditional** | 3 | 36.57% | 89.92% | Model infers instead of quoting |
| **procedural** | 2 | 47.78% | 88.57% | Missing step details |
| **aggregation** | 1 | 27.27% | 100.00% | Can't find all items |
| **edge_case** | 2 | 8.33% | 97.50% | Should refuse, but explains |

### 3.3 Performance by Difficulty

| Difficulty | Count | Faithfulness | Issue |
|------------|-------|--------------|-------|
| **easy** | 1 | 26.67% | Paradoxically worst (model over-explains) |
| **medium** | 8 | 50.06% | Best performance |
| **hard** | 6 | 29.64% | Multi-hop reasoning fails |

### 3.4 Performance by Entity Type

| Entity | Faithfulness | Issue |
|--------|--------------|-------|
| تمويل_استهلاكي (Consumer Finance) | 25.10% | Worst - missing ground truth in corpus |
| سمسرة (Brokerage) | 38.52% | Model adds procedures not in docs |
| تأمين (Insurance) | 49.42% | Better grounding |
| توريق (Securitization) | 51.71% | Best - documents are detailed |
| تمويل_متناهي_الصغر (Microfinance) | 55.56% | Good document coverage |

---

## 4. Root Cause Analysis

### 4.1 Low Faithfulness (40.33%) - PRIMARY ISSUE

**Symptoms:**
- Model adds information not present in retrieved context
- Model uses phrases like "عادةً" (usually), "غالباً" (often) despite being forbidden
- Model provides interpretations rather than direct quotes

**Root Causes:**
1. **Insufficient grounding constraint**: System prompt is long; model may lose focus
2. **Context overflow**: 5 retrieved chunks may be too much; model synthesizes/hallucinates
3. **Model knowledge leakage**: Grok's training data includes Arabic legal texts; model uses prior knowledge
4. **Missing explicit refusal training**: Model doesn't know when to say "لم أجد"

**Evidence from Evaluation:**
```
Question: "ما هو الحد الأدنى لرأس المال لشركة تمويل استهلاكي؟"
Ground Truth: "50 مليون جنيه مصري"
Model Response: "**لا يوجد نص صريح في المستندات المتاحة**..."
Faithfulness: 26.67%
```
The model correctly refuses BUT then adds tangential information, lowering faithfulness.

### 4.2 Low Hit Rate (66.67%)

**Symptoms:**
- Relevant document not in top 5 for 33% of queries
- Questions about specific regulations miss the right document

**Root Causes:**
1. **Semantic gap**: Query terms don't match document terms (e.g., "رأس المال" vs "الحد الأدنى للتمويل")
2. **No query expansion**: Single query embedding may miss relevant docs
3. **BM25 tokenization**: Arabic morphology not handled (no stemming)
4. **Small corpus**: Only 112 chunks; sparse coverage

### 4.3 Low Answer Correctness (32.26%)

**Symptoms:**
- Generated answers differ significantly from ground truth
- Model provides more detail than ground truth expects

**Root Causes:**
1. **Ground truth too short**: GT answers are 1-2 sentences; model generates paragraphs
2. **Evaluation metric mismatch**: Semantic similarity penalizes verbosity
3. **Different answer style**: Model uses structured format; GT is plain text

### 4.4 Edge Case Handling (8.33% Faithfulness)

**Symptoms:**
- Model tries to answer out-of-scope questions instead of refusing
- Example: Traffic violation question answered with tangential tax info

**Root Causes:**
1. **No scope boundary**: Model doesn't know FRA domain boundaries
2. **Retrieval returns irrelevant docs**: Vector search returns "similar" but wrong docs
3. **Model tries to be helpful**: Overrides refusal instinct

---

## 5. Technical Recommendations

### 5.1 High Priority - Faithfulness Improvement

#### 5.1.1 Shorter, Stricter System Prompt
```
Current: ~800 tokens of instructions
Recommended: ~200 tokens focused ONLY on:
1. Copy text from sources
2. Cite every claim
3. Say "لم أجد" if not present
```

#### 5.1.2 Context Reduction
```python
# Current
DEFAULT_TOP_K = 5

# Recommended
DEFAULT_TOP_K = 3  # Fewer docs = less hallucination
```

#### 5.1.3 Chain-of-Verification (CoVe)
Add a verification step after generation:
```
1. Generate answer
2. Extract claims from answer
3. For each claim, check if it appears in sources
4. Remove ungrounded claims
5. Return verified answer
```

#### 5.1.4 Constrained Decoding
Use Grok's function calling or structured output to force:
```json
{
  "answer_found": true/false,
  "direct_quotes": ["quote1", "quote2"],
  "interpretation": null,  // Force empty
  "source_articles": ["المادة 5", "المادة 7"]
}
```

### 5.2 Medium Priority - Retrieval Improvement

#### 5.2.1 Arabic-Aware BM25
```python
# Current: Whitespace tokenization
tokens = text.lower().split()

# Recommended: Use CAMeL Tools or Farasa
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.morphology.analyzer import Analyzer

tokens = simple_word_tokenize(text)
stems = [analyzer.analyze(t)[0]['stem'] for t in tokens]
```

#### 5.2.2 Query Expansion
Before retrieval, expand query with:
```python
def expand_query(query: str) -> List[str]:
    return [
        query,
        rephrase_query(query),  # LLM rephrase
        extract_keywords(query),  # Key terms only
        add_synonyms(query),  # Arabic synonyms
    ]
```

#### 5.2.3 Hybrid Score Tuning
```python
# Current: Fixed weights
final_score = 0.5 * vector_score + 0.5 * bm25_score

# Recommended: Learned weights per query type
weights = {
    "factual": (0.7, 0.3),  # More semantic
    "procedural": (0.4, 0.6),  # More keyword
    "comparison": (0.5, 0.5),
}
```

#### 5.2.4 ColBERT/Late Interaction
Replace dense retrieval with ColBERT for token-level matching:
```
Model: "colbert-ir/colbertv2.0" or Arabic fine-tuned version
Benefit: Better term matching for legal terminology
```

### 5.3 Low Priority - Corpus Improvement

#### 5.3.1 Expand Document Corpus
```
Current: 112 chunks from ~50 DOCX files
Target: 1000+ chunks from:
- Full FRA regulations
- Historical decisions
- FAQ documents
```

#### 5.3.2 Hierarchical Chunking
```python
# Current: Flat 1000-char chunks
# Recommended: Preserve document structure

ChunkLevel.ARTICLE  # Full article as context
ChunkLevel.CLAUSE   # Index at clause level
ChunkLevel.PARENT   # Retrieve parent on match
```

#### 5.3.3 Ground Truth Enrichment
```json
{
  "question": "ما هو الحد الأدنى لرأس المال؟",
  "ground_truth_short": "50 مليون جنيه",
  "ground_truth_detailed": "الحد الأدنى لرأس المال المدفوع لشركة التمويل الاستهلاكي هو 50 مليون جنيه مصري وفقاً للمادة 5 من قرار مجلس إدارة الهيئة رقم 179 لسنة 2020.",
  "source_article": "المادة 5",
  "source_document": "قرار-179-2020.docx"
}
```

---

## 6. Proposed Architecture Changes

### 6.1 Option A: Minimal Changes (1-2 weeks)

```
Changes:
1. Shorten system prompt to 200 tokens
2. Reduce TOP_K from 5 to 3
3. Add post-generation verification step
4. Integrate Arabic stemmer for BM25

Expected Improvement:
- Faithfulness: 40% → 60%
- Hit Rate: 67% → 75%
```

### 6.2 Option B: Moderate Overhaul (1 month)

```
Changes:
1. All Option A changes
2. Replace BM25 with CAMeL-tokenized version
3. Add query expansion with LLM
4. Implement Chain-of-Verification
5. Add scope detection (is question in FRA domain?)
6. Tune hybrid weights per query type

Expected Improvement:
- Faithfulness: 40% → 75%
- Hit Rate: 67% → 85%
- Anti-Hallucination: 69% → 90%
```

### 6.3 Option C: Full Rebuild (2-3 months)

```
Changes:
1. Replace retrieval with ColBERT
2. Fine-tune embedding model on FRA corpus
3. Implement RAG-Fusion (multiple query variants)
4. Add semantic caching
5. Implement hierarchical retrieval (article → clause)
6. Train custom Arabic legal NER
7. Add confidence scoring with abstention

Expected Improvement:
- Faithfulness: 40% → 90%
- Hit Rate: 67% → 95%
- Overall Score: 53% → 85%
```

---

## 7. Immediate Action Items

### Week 1
- [ ] Shorten system prompt to 200 tokens (focus on grounding)
- [ ] Reduce TOP_K to 3
- [ ] Add explicit scope check before answering

### Week 2
- [ ] Implement post-generation claim verification
- [ ] Add Arabic stemmer to BM25 (CAMeL Tools)
- [ ] Expand evaluation dataset to 50 questions

### Week 3
- [ ] Add query expansion with LLM
- [ ] Tune hybrid retrieval weights
- [ ] Re-evaluate and measure improvements

---

## 8. Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `llm_client/grok_client.py:59-117` | Shorten system prompt | HIGH |
| `config.py:76` | Change DEFAULT_TOP_K to 3 | HIGH |
| `rag_engine/hybrid_retriever.py:116-124` | Add Arabic tokenizer | MEDIUM |
| `main.py:300-350` | Add claim verification step | HIGH |
| `evaluation/pipeline.py` | Add per-claim faithfulness scoring | MEDIUM |

---

## 9. Contact & Questions

For questions about this report, the system can be tested at:
```
python app.py  # Starts Gradio UI at localhost:7860
python evaluate.py --run  # Runs full evaluation
```

Key metric to optimize: **Faithfulness** - currently 40.33%, target 85%+

---

*Report generated for LLM Architect review*
